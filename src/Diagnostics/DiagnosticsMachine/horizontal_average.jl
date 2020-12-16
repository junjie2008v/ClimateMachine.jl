"""
    HorizontalAverage

A horizontal reduction into a single vertical dimension.
"""
abstract type HorizontalAverage <: DiagnosticVar end
dv_HorizontalAverage(
    ::ClimateMachineConfigType,
    ::HorizontalAverage,
    ::BalanceLaw,
    ::States,
    ::AbstractFloat,
) = nothing

# replace these with a `dv_array_dims` that takes `nvars` and returns the dims for the array
# or create the array? Use `Array`? `similar`?
function dv_dg_points_length(::ClimateMachineConfigType, ::Type{HorizontalAverage})
    :(Nqk)
end
function dv_dg_points_index(::ClimateMachineConfigType, ::Type{HorizontalAverage})
    :(k)
end

function dv_dg_elems_length(::ClimateMachineConfigType, ::Type{HorizontalAverage})
    :(nvertelem)
end
function dv_dg_elems_index(::ClimateMachineConfigType, ::Type{HorizontalAverage})
    :(ev)
end

function dv_dg_dimnames(::ClimateMachineConfigType, ::Type{HorizontalAverage})
    ("z",)
end
function dv_dg_dimranges(::ClimateMachineConfigType, ::Type{HorizontalAverage})
    (:(AtmosCollected.zvals),)
end

function dv_op(
    ::ClimateMachineConfigType,
    ::Type{HorizontalAverage},
    x,
    y,
    scale_with = 1,
)
    x += y * scale_with
end

macro horizontal_average(impl, config_type, name)
    iex = quote
        $(generate_dv_interface(:HorizontalAverage, config_type, name))
        $(generate_dv_function(:HorizontalAverage, config_type, name, impl))
    end
    esc(MacroTools.prewalk(unblock, iex))
end

macro horizontal_average(
    impl,
    config_type,
    name,
    units,
    long_name,
    standard_name,
)
    iex = quote
        $(generate_dv_interface(
            :HorizontalAverage,
            config_type,
            name,
            units,
            long_name,
            standard_name,
        ))
        $(generate_dv_function(:HorizontalAverage, config_type, name, impl))
    end
    esc(MacroTools.prewalk(unblock, iex))
end
