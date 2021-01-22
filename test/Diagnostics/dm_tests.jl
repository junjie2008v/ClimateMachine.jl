using Dates
using FileIO
using MPI
using NCDatasets
using Printf
using Random
using StaticArrays
using Test

using ClimateMachine
ClimateMachine.init()
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.DiagnosticsMachine
using ClimateMachine.GenericCallbacks
using ClimateMachine.VariableTemplates
using ClimateMachine.Writers

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function init_dm_test!(problem, bl, state, aux, localgeo, t)
    (x, y, z) = localgeo.coord

    FT = eltype(state)

    z = FT(z)

    ρ = one(FT)
    w = FT(10 + 0.5 * sin(2 * π * ((x / 1500) + (y / 1500))))
    u = (5 + 2 * sin(2 * π * ((x / 1500) + (y / 1500))))
    v = FT(5 + 2 * sin(2 * π * ((x / 1500) + (y / 1500))))

    state.ρ = ρ
    state.ρu = SVector(ρ * u, ρ * v, ρ * w)
end

function config_dm_test(FT, N, resolution, xmax, ymax, zmax)
    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK54CarpenterKennedy,
    )
    config = ClimateMachine.AtmosLESConfiguration(
        "DiagnosticsMachine test",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_dm_test!;
        solver_type = ode_solver,
    )

    return config
end

function main()
    FT = Float64

    # DG polynomial order
    N = 4

    # Domain resolution and size
    Δh = FT(50)
    Δv = FT(20)
    resolution = (Δh, Δh, Δv)

    xmax = FT(1500)
    ymax = FT(1500)
    zmax = FT(1500)

    t0 = FT(0)
    dt = FT(0.01)
    timeend = dt

    driver_config = config_dm_test(FT, N, resolution, xmax, ymax, zmax)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
        init_on_cpu = true,
    )
    dgn_config = config_diagnostics(driver_config)

    mpicomm = solver_config.mpicomm
    dg = solver_config.dg
    Q = solver_config.Q
    solver = solver_config.solver

    outdir = mktempdir()
    currtime = ODESolvers.gettime(solver)
    starttime = replace(string(now()), ":" => ".")
    Diagnostics.init(mpicomm, param_set, dg, Q, starttime, outdir)
    GenericCallbacks.init!(
        dgn_config.groups[1],
        nothing,
        nothing,
        nothing,
        currtime,
    )

    ClimateMachine.invoke!(solver_config)

    # Check results
    mpirank = MPI.Comm_rank(mpicomm)
    if mpirank == 0
        dgngrp = dgn_config.groups[1]
        nm = @sprintf(
            "%s_%s_%s.nc",
            replace(dgngrp.out_prefix, " " => "_"),
            dgngrp.name,
            starttime,
        )
        ds = NCDataset(joinpath(outdir, nm), "r")
        ds_u = ds["u"][:]
        ds_cov_w_u = ds["cov_w_u"][:]
        N = size(ds_u, 1)
        err = 0
        err1 = 0
        for i in 1:N
            u = ds_u[i]
            cov_w_u = ds_cov_w_u[i]
            err += (cov_w_u - 0.5)^2
            err1 += (u - 5)^2
        end
        close(ds)
        err = sqrt(err / N)
        @test err1 <= 1e-16
        @test err <= 2e-15
    end
end

main()
