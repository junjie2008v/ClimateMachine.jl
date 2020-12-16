# Return `true` if the specified symbol is a type name that is a subtype
# of `BalanceLaw` and `false` otherwise.
isa_bl(sym::Symbol) = any(
    bl -> endswith(bl, "." * String(sym)),
    map(bl -> String(Symbol(bl)), subtypes(BalanceLaw)),
)
isa_bl(ex) = false

uppers_in(s) = foldl((f, c) -> isuppercase(c) ? f * c : f, s, init = "")

# Generate the common definitions used in many places.
function generate_common_defs()
    quote
        mpicomm = Settings.mpicomm
        dg = Settings.dg
        Q = Settings.Q
        mpirank = MPI.Comm_rank(mpicomm)
        bl = dg.balance_law
        grid = dg.grid
        grid_info = basic_grid_info(dg)
        topl_info = basic_topology_info(grid.topology)
        Nqk = grid_info.Nqk
        Nqh = grid_info.Nqh
        npoints = prod(grid_info.Nq)
        nrealelem = topl_info.nrealelem
        nvertelem = topl_info.nvertelem
        nhorzelem = topl_info.nhorzrealelem
        FT = eltype(Q)
    end
end

# Generate the `dims` dictionary for `Writers.init_data`.
function generate_init_dims(::NoInterpolation, cfg, dvtype_dvars_map)
    dimslst = Any[]
    for dvtype in keys(dvtype_dvars_map)
        dimnames = dv_dg_dimnames(cfg, dvtype)
        dimranges = dv_dg_dimranges(cfg, dvtype)
        for (dimname, dimrange) in zip(dimnames, dimranges)
            lhs = :($dimname)
            rhs = :(collect($dimrange), Dict())
            push!(dimslst, :($lhs => $rhs))
        end
    end

    quote
        OrderedDict($(Expr(:tuple, dimslst...))...)
    end
end
function generate_init_dims(::InterpolationType, cfg, dvtype_dvars_map)
    quote
        dims = dimensions(interpol)
        if interpol isa InterpolationCubedSphere
            # Adjust `level` on the sphere.
            level_val = dims["level"]
            dims["level"] = (
                level_val[1] .- FT(planet_radius(Settings.param_set)),
                level_val[2],
            )
        end
        dims
    end
end

get_dimnames(::NoInterpolation, cfg, dvtype) = dv_dg_dimnames(cfg, dvtype)
get_dimnames(::InterpolationType, cfg, dvtype) = :(tuple(collect(keys(dims))))

# Generate the `vars` dictionary for `Writers.init_data`.
function generate_init_vars(intrp, cfg, dvtype_dvars_map)
    varslst = Any[]
    for (dvtype, dvlst) in dvtype_dvars_map
        for dvar in dvlst
            lhs = :($(dv_name(cfg, dvar)))
            dimnames = get_dimnames(intrp, cfg, dvtype)
            rhs = :($dimnames, FT, $(dv_attrib(cfg, dvar)))
            push!(varslst, :($lhs => $rhs))
        end
    end

    quote
        # TODO: add code to filter this based on what's actually in `bl`.
        OrderedDict($(Expr(:tuple, varslst...))...)
    end
end

# Generate `Diagnostics.$(name)_init(...)` which will initialize the
# `DiagnosticsGroup` when called.
function generate_init_fun(
    name,
    config_type,
    params_type,
    init_fun,
    interpolate,
    dvtype_dvars_map,
)
    init_name = Symbol(name, "_init")
    cfg = getfield(ConfigTypes, config_type)
    intrp = getfield(@__MODULE__, interpolate)
    quote
        function $init_name(dgngrp, curr_time)
            $(generate_common_defs())

            $(init_fun)(dgngrp, curr_time)

            if dgngrp.onetime
                collect_onetime(mpicomm, dg, Q)
            end

            if mpirank == 0
                dims = $(generate_init_dims(intrp(), cfg(), dvtype_dvars_map))
                vars = $(generate_init_vars(intrp(), cfg(), dvtype_dvars_map))

                # create the output file
                dprefix = @sprintf(
                    "%s_%s",
                    dgngrp.out_prefix,
                    dgngrp.name,
                )
                dfilename = joinpath(Settings.output_dir, dprefix)
                init_data(dgngrp.writer, dfilename, dims, vars)
            end

            return nothing
        end
    end
end

# Generate code snippet for copying arrays to the CPU if needed. Ideally,
# this will be removed when diagnostics are made to run on GPU.
function generate_array_copies()
    quote
        # get needed arrays onto the CPU
        if array_device(Q) isa CPU
            state_data = Q.realdata
            gradflux_data = dg.state_gradient_flux.realdata
            aux_data = dg.state_auxiliary.realdata
            vgeo = grid.vgeo
        else
            state_data = Array(Q.realdata)
            gradflux_data = Array(dg.state_gradient_flux.realdata)
            aux_data = Array(dg.state_auxiliary.realdata)
            vgeo = Array(grid.vgeo)
        end
    end
end

# Generate code to create the necessary arrays for the diagnostics
# variables.
function generate_create_vars_arrays(
    ::CollectOnInterpolatedGrid,
    cfg,
    dvtype_dvars_map,
)
end
function generate_create_vars_arrays(
    ::InterpolationType,
    cfg,
    dvtype_dvars_map,
)
    cva_exs = []
    for (dvtype, dvlst) in dvtype_dvars_map
        dvt_short = uppers_in(split(String(Symbol(dvtype)), ".")[end])
        arr_name = Symbol("vars_", dvt_short, "_dg_array")
        npoints = dv_dg_points_length(cfg, dvtype)
        nvars = length(dvlst)
        nelems = dv_dg_elems_length(cfg, dvtype)
        cva_ex = quote
            $arr_name = Array{FT}(undef, $npoints, $nvars, $nelems)
        end
        push!(cva_exs, cva_ex)
    end
    return Expr(:block, (cva_exs...))
end

# Generate calls to the implementations for the `DiagnosticVar`s in this
# group and store the results.
function generate_collect_calls(
    ::InterpolationType,
    cfg,
    dvtype_dvars_map,
)
    all_cc_exs = []
    for (dvtype, dvlst) in dvtype_dvars_map
        dvt = split(String(Symbol(dvtype)), ".")[end]
        dvt_short = uppers_in(dvt)
        arr_name = Symbol("vars_", dvt_short, "_dg_array")
        nvars = length(dvlst)
        pt = dv_dg_points_index(cfg, dvtype)
        elem = dv_dg_elems_index(cfg, dvtype)
        var_impl = Symbol("dv_", dvt)

        cc_exs = []
        for dvar in dvlst
            impl_args = dv_args(cfg, dvar)
            AT1 = impl_args[1][2] # the type of the first argument
            if isa_bl(AT1)
                impl_extra_params = ()
            else
                AT2 = impl_args[2][2] # the type of the second argument
                @assert isa_bl(AT2)
                AN1 = impl_args[1][1] # the name of the first argument
                impl_extra_params = (Symbol("bl.", AN1),)
            end
            cc_ex = quote
                dv_op(
                    $cfg,
                    $dvtype,
                    getindex($arr_name, $pt, v, $elem),
                    $(var_impl)(
                        $cfg,
                        $dvar,
                        $(impl_extra_params...),
                        bl,
                        states,
                        curr_time,
                        cache,
                    ),
                    MH,
                )
            end
            push!(cc_exs, cc_ex)
        end
        dvtype_cc_ex = quote
            for v in 1:$nvars
                $(cc_exs...)
            end
        end
        push!(all_cc_exs, dvtype_cc_ex)
    end

    return Expr(:block, (all_cc_exs...))
end

# Generate the nested loops to traverse the DG grid within which we extract
# the various states and then generate the individual collection calls.
function generate_dg_collection(
    ::CollectOnInterpolatedGrid,
    cfg,
    dvtype_dvars_map,
)
end
function generate_dg_collection(
    intrp::InterpolationType,
    cfg,
    dvtype_dvars_map,
)
    quote
        cache = Dict{Symbol, Any}()
        for eh in 1:nhorzelem, ev in 1:nvertelem
            e = ev + (eh - 1) * nvertelem
            for k in 1:Nqk, j in 1:Nq, i in 1:Nq
                ijk = i + Nq * ((j - 1) + Nq * (k - 1))
                evk = Nqk * (ev - 1) + k
                MH = vgeo[ijk, grid.MHid, e]
                z = AtmosCollected.zvals[evk]
                states = States(
                    extract_state(bl, state_data, ijk, e, Prognostic()),
                    extract_state(
                        bl,
                        gradflux_data,
                        ijk,
                        e,
                        GradientFlux(),
                    ),
                    extract_state(bl, aux_data, ijk, e, Auxiliary()),
                )
                $(generate_collect_calls(intrp, cfg, dvtype_dvars_map))
            end
        end
        empty!(cache)
    end
end

function generate_interpolation(dvars)
    if interpolate != :NoInterpolation
        quote
            all_state_data = nothing
            all_gradflux_data = nothing
            all_aux_data = nothing

            if interpolate && !isempty(dvars_ig)
                istate_array = similar(
                    Q.realdata,
                    interpol.Npl,
                    number_states(bl, Prognostic()),
                )
                interpolate_local!(interpol, Q.realdata, istate_array)
                igradflux_array = similar(
                    Q.realdata,
                    interpol.Npl,
                    number_states(bl, GradientFlux()),
                )
                interpolate_local!(
                    interpol,
                    dg.state_gradient_flux.realdata,
                    igradflux_array,
                )
                iaux_array = similar(
                    Q.realdata,
                    interpol.Npl,
                    number_states(bl, Auxiliary()),
                )
                interpolate_local!(
                    interpol,
                    dg.state_auxiliary.realdata,
                    iaux_array,
                )

                _ρu, _ρv, _ρw = 2, 3, 4
                project_cubed_sphere!(interpol, istate_array, (_ρu, _ρv, _ρw))

                # FIXME: accumulating to rank 0 is not scalable
                all_state_data = accumulate_interpolated_data(
                    mpicomm,
                    interpol,
                    istate_array,
                )
                all_gradflux_data = accumulate_interpolated_data(
                    mpicomm,
                    interpol,
                    igradflux_array,
                )
                all_aux_data =
                    accumulate_interpolated_data(mpicomm, interpol, iaux_array)
            end
        end
    else
        quote end
    end
end

# Generate `Diagnostics.$(name)_collect(...)` which when called,
# performs a collection of all the diagnostic variables in the group
# and writes them out.
function generate_collect_fun(
    name,
    config_type,
    params_type,
    init_fun,
    interpolate,
    dvtype_dvars_map,
)
    collect_name = Symbol(name, "_collect")
    cfg = getfield(ConfigTypes, config_type)
    intrp = getfield(@__MODULE__, interpolate)
    quote
        function $collect_name(dgngrp, curr_time)
            $(generate_common_defs())
            $(generate_array_copies())
            interpol = dgngrp.interpol
            $(generate_create_vars_arrays(intrp(), cfg(), dvtype_dvars_map))

            # Traverse the DG grid and collect diagnostics as needed.
            $(generate_dg_collection(intrp(), cfg(), dvtype_dvars_map))

            # Perform any reductions necessary.
            $(generate_dg_reductions(intrp(), cfg(), dvtype_dvars_map))

            #=
            # Interpolate and accumulate if needed.
            $(generate_interpolation(intrp(), cfg(), dvtype_dvars_map))

            # Traverse the interpolated grid and collect diagnostics if needed.
            $(generate_i_collection(dvars))
            if interpolate
                ivars_array = similar(Q.realdata, interpol.Npl, n_grp_vars)
                interpolate_local!(interpol, vars_array, ivars_array)
                all_ivars_data = accumulate_interpolated_data(
                    mpicomm,
                    interpol,
                    ivars_array,
                )
            end

            # TODO: density averaging.

            if mpirank == 0
                dims = dimensions(interpol)
                (nx, ny, nz) = map(k -> dims[k][1], collect(keys(dims)))
                for x in 1:nx, y in 1:ny, z in 1:nz
                    istate = Vars{vars_state(bl, Prognostic(), FT)}(view(
                        all_state_data,
                        x,
                        y,
                        z,
                        :,
                    ))
                    igradflux = Vars{vars_state(bl, GradientFlux(), FT)}(view(
                        all_gradflux_data,
                        x,
                        y,
                        z,
                        :,
                    ))
                    iaux = Vars{vars_state(bl, Auxiliary(), FT)}(view(
                        all_aux_data,
                        x,
                        y,
                        z,
                        :,
                    ))
                    states = States(istate, igradflux, iaux)
                    vars = Vars{grp_vars}(view(ivars_array, x, y, z, :))
                    $(generate_collect_calls(name, config_type, dvars_ig))
                end

                varvals = OrderedDict()
                varnames = map(
                    s -> startswith(s, "moisture.") ? s[10:end] : s, # XXX: FIXME
                    flattenednames(grp_vars),
                )
                for (vari, varname) in enumerate(varnames)
                    varvals[varname] = vars_array[
                        ntuple(_ -> Colon(), ndims(vars_array))...,
                        vari,
                    ] # XXX: FIXME
                end
                append_data(dgngrp.writer, varvals, curr_time)
            end
            =#

            MPI.Barrier(mpicomm)
            return nothing
        end
    end
end

# Generate `Diagnostics.$(name)_fini(...)`, which does nothing right now.
function generate_fini_fun(name, args...)
    fini_name = Symbol(name, "_fini")
    quote
        function $fini_name(dgngrp, curr_time) end
    end
end

# Generate `setup_$(name)(...)` which will create the `DiagnosticsGroup`
# for $name when called.
function generate_setup_fun(
    name,
    config_type,
    params_type,
    init_fun,
    interpolate,
    dvtype_dvars_map,
)
    init_name = Symbol(name, "_init")
    collect_name = Symbol(name, "_collect")
    fini_name = Symbol(name, "_fini")

    setup_name = Symbol("setup_", name)
    intrp = getfield(@__MODULE__, interpolate)

    no_intrp_err = quote end
    some_intrp_err = quote end
    if intrp isa NoInterpolation
        some_intrp_err = quote
            throw(ArgumentError(
                "$($name) specifies interpolation, but no " *
                "`InterpolationTopology` has been provided.",
            ))
        end
    else
        no_intrp_err = quote
            @warn "$($name) does not specify interpolation, but an " *
                  "`InterpolationTopology` has been provided; ignoring."
            interpol = nothing
        end
    end
    quote
        function $setup_name(
            ::$config_type,
            params::$params_type,
            interval::String,
            out_prefix::String,
            writer = NetCDFWriter(),
            interpol = nothing,
        ) where {
            $config_type <: ClimateMachineConfigType,
            $params_type <: Union{Nothing, DiagnosticsGroupParams},
        }
            if isnothing(interpol)
                $(some_intrp_err)
            else
                $(no_intrp_err)
            end

            return DiagnosticsGroup(
                $name,
                $init_name,
                $collect_name,
                $fini_name,
                interval,
                out_prefix,
                writer,
                interpol,
                $(intrp isa NoInterpolation),
                params,
            )
        end
    end
end
