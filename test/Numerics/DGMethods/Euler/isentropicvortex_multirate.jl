using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.BalanceLaws
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.GenericCallbacks
using ClimateMachine.Mesh.Geometry
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.Orientations
using ClimateMachine.SystemSolvers
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates
using ClimateMachine.VTK

using CLIMAParameters
using CLIMAParameters.Planet: kappa_d
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test

include("isentropicvortex_setup.jl")

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

const output_vtk = false

function main()
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 4
    numlevels = integration_testing ? 4 : 1

    expected_error = Dict()
    expected_error[Float64, SSPRK33ShuOsher, 1] = 2.3222373077778794e+01
    expected_error[Float64, SSPRK33ShuOsher, 2] = 5.2782503174265516e+00
    expected_error[Float64, SSPRK33ShuOsher, 3] = 1.2281763287878383e-01
    expected_error[Float64, SSPRK33ShuOsher, 4] = 2.3761870907666096e-03

    expected_error[Float64, ARK2GiraldoKellyConstantinescu, 1] =
        2.3245216640111998e+01
    expected_error[Float64, ARK2GiraldoKellyConstantinescu, 2] =
        5.2626584944153949e+00
    expected_error[Float64, ARK2GiraldoKellyConstantinescu, 3] =
        1.2324230746483673e-01
    expected_error[Float64, ARK2GiraldoKellyConstantinescu, 4] =
        3.8777995619211627e-03

    @testset "$(@__FILE__)" begin
        for FT in (Float64,), dims in 2
            for FastMethod in (SSPRK33ShuOsher, ARK2GiraldoKellyConstantinescu)
                @info @sprintf """Configuration
                                  ArrayType  = %s
                                  FastMethod = %s
                                  FT     = %s
                                  dims       = %d
                                  """ ArrayType "$FastMethod" "$FT" dims

                setup = IsentropicVortexSetup{FT}()
                errors = Vector{FT}(undef, numlevels)

                for level in 1:numlevels
                    numelems =
                        ntuple(dim -> dim == 3 ? 1 : 2^(level - 1) * 5, dims)
                    errors[level] = test_run(
                        mpicomm,
                        ArrayType,
                        polynomialorder,
                        numelems,
                        setup,
                        FT,
                        FastMethod,
                        dims,
                        level,
                    )

                    @test errors[level] ≈ expected_error[FT, FastMethod, level]
                end

                rates = @. log2(
                    first(errors[1:(numlevels - 1)]) /
                    first(errors[2:numlevels]),
                )
                numlevels > 1 && @info "Convergence rates\n" * join(
                    [
                        "rate for levels $l → $(l + 1) = $(rates[l])"
                        for l in 1:(numlevels - 1)
                    ],
                    "\n",
                )
            end
        end
    end
end

function test_run(
    mpicomm,
    ArrayType,
    polynomialorder,
    numelems,
    setup,
    FT,
    FastMethod,
    dims,
    level,
)
    brickrange = ntuple(dims) do dim
        range(
            -setup.domain_halflength;
            length = numelems[dim] + 1,
            stop = setup.domain_halflength,
        )
    end

    topology = BrickTopology(
        mpicomm,
        brickrange;
        periodicity = ntuple(_ -> true, dims),
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
    )

    problem =
        AtmosProblem(boundaryconditions = (), init_state_prognostic = setup)

    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        problem = problem,
        orientation = NoOrientation(),
        ref_state = IsentropicVortexReferenceState{FT}(setup),
        turbulence = ConstantDynamicViscosity(FT(0)),
        moisture = DryModel(),
        source = (),
    )
    # The linear model has the fast time scales
    fast_model = AtmosAcousticLinearModel(model)
    # The nonlinear model has the slow time scales

    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )
    fast_dg = DGModel(
        fast_model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        state_auxiliary = dg.state_auxiliary,
    )
    slow_dg = remainder_DGModel(dg, (fast_dg,))

    timeend = FT(2 * setup.domain_halflength / setup.translation_speed)
    # determine the slow time step
    elementsize = minimum(step.(brickrange))
    slow_dt =
        8 * elementsize / soundspeed_air(model.param_set, setup.T∞) /
        polynomialorder^2
    nsteps = ceil(Int, timeend / slow_dt)
    slow_dt = timeend / nsteps

    # arbitrary and not needed for stability, just for testing
    fast_dt = slow_dt / 3

    Q = init_ode_state(dg, FT(0), setup)

    slow_ode_solver = LSRK144NiegemannDiehlBusch(slow_dg, Q; dt = slow_dt)

    # check if FastMethod is ARK, is there a better way ?
    if FastMethod == ARK2GiraldoKellyConstantinescu

        linearsolver = GeneralizedMinimalResidual(Q; M = 10, rtol = 1e-10)
        # splitting the fast part into full and linear but the fast part
        # is already linear so full_dg == linear_dg == fast_dg
        fast_ode_solver = FastMethod(
            fast_dg,
            fast_dg,
            LinearBackwardEulerSolver(linearsolver; isadjustable = true),
            Q;
            dt = fast_dt,
            paperversion = true,
        )
    else
        fast_ode_solver = FastMethod(fast_dg, Q; dt = fast_dt)
    end

    ode_solver = MultirateRungeKutta((slow_ode_solver, fast_ode_solver))

    eng0 = norm(Q)
    dims == 2 && (numelems = (numelems..., 0))
    @info @sprintf """Starting refinement level %d
                      numelems  = (%d, %d, %d)
                      slow_dt   = %.16e
                      fast_dt   = %.16e
                      norm(Q₀)  = %.16e
                      """ level numelems... slow_dt fast_dt eng0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            runtime = Dates.format(
                convert(DateTime, now() - starttime[]),
                dateformat"HH:MM:SS",
            )
            @info @sprintf """Update
                              simtime = %.16e
                              runtime = %s
                              norm(Q) = %.16e
                              """ gettime(ode_solver) runtime energy
        end
    end
    callbacks = (cbinfo,)

    if output_vtk
        # create vtk dir
        vtkdir =
            "vtk_isentropicvortex_multirate" *
            "_poly$(polynomialorder)_dims$(dims)_$(ArrayType)_$(FT)" *
            "_$(FastMethod)_level$(level)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model)

        # setup the output callback
        outputtime = timeend
        cbvtk = EveryXSimulationSteps(floor(outputtime / slow_dt)) do
            vtkstep += 1
            Qe = init_ode_state(dg, gettime(ode_solver), setup)
            do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model)
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(Q, ode_solver; timeend = timeend, callbacks = callbacks)

    # final statistics
    Qe = init_ode_state(dg, timeend, setup)
    engf = norm(Q)
    engfe = norm(Qe)
    errf = euclidean_distance(Q, Qe)
    @info @sprintf """Finished refinement level %d
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    norm(Q - Qe)            = %.16e
    norm(Q - Qe) / norm(Qe) = %.16e
    """ level engf engf / eng0 engf - eng0 errf errf / engfe
    errf
end

function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    Qe,
    model,
    testname = "isentropicvortex_mutirate",
)
    ## name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/%s_mpirank%04d_step%04d",
        vtkdir,
        testname,
        MPI.Comm_rank(mpicomm),
        vtkstep
    )

    statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
    exactnames = statenames .* "_exact"

    writevtk(filename, Q, dg, statenames, Qe, exactnames)

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## name of the pvtu file
        pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

        ## name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
        end

        writepvtu(
            pvtuprefix,
            prefixes,
            (statenames..., exactnames...),
            eltype(Q),
        )

        @info "Done writing VTK: $pvtuprefix"
    end
end

main()
