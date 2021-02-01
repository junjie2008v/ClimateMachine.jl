using ClimateMachine
using ClimateMachine.Thermodynamics
using ClimateMachine.TemperatureProfiles
using UnPack
using CLIMAParameters
using RootSolvers
using CLIMAParameters.Planet
using Plots
import ClimateMachine.Thermodynamics
Thermodynamics.print_warning() = false
TD = Thermodynamics

struct EarthParameterSet <: AbstractEarthParameterSet end;
const param_set = EarthParameterSet();
FT = Float64;

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "test", "Common", "Thermodynamics", "profiles.jl"))
include(joinpath(clima_dir, "docs", "plothelpers.jl"));
profiles = PhaseEquilProfiles(param_set, Array{FT});
@unpack ρ, e_int, q_tot = profiles
prof_pts = (ρ, e_int, q_tot)

dims = (10, 10, 10);
ρ = range(min(ρ...), stop = max(ρ...), length = dims[1]);
e_int = range(min(e_int...), stop = max(e_int...), length = dims[2]);
q_tot = range(min(q_tot...), stop = max(q_tot...), length = dims[3]);

ρ_all = Array{FT}(undef, prod(dims));
e_int_all = Array{FT}(undef, prod(dims));
q_tot_all = Array{FT}(undef, prod(dims));

linear_indices = LinearIndices((1:dims[1], 1:dims[2], 1:dims[3]));

folder = "sat_adjust"
folder_energy = "sat_adjust_energy"
mkpath(folder)
mkpath(folder_energy)

numerical_methods =
    (SecantMethod, NewtonsMethod, NewtonsMethodAD, RegulaFalsiMethod)

ts = Dict(
    NM => Array{Union{ThermodynamicState, Nothing}}(undef, prod(dims))
    for NM in numerical_methods
)
ts_no_err = Dict(
    NM => Array{ThermodynamicState}(undef, prod(dims))
    for NM in numerical_methods
)

@inbounds for i in linear_indices.indices[1]
    @inbounds for j in linear_indices.indices[2]
        @inbounds for k in linear_indices.indices[3]
            @inbounds for NM in numerical_methods
                n = linear_indices[i, j, k]
                ρ_all[n] = ρ[i]
                e_int_all[n] = e_int[j]
                q_tot_all[n] = q_tot[k]

                Thermodynamics.error_on_non_convergence() = false
                ts_no_err[NM][n] = TD.PhaseEquil_dev_only(
                    param_set,
                    e_int[j],
                    ρ[i],
                    q_tot[k];
                    sat_adjust_method = NM,
                    maxiter = 10,
                )
                Thermodynamics.error_on_non_convergence() = true
                # @show n/prod(linear_indices.indices)*100
                try
                    ts[NM][n] = TD.PhaseEquil_dev_only(
                        param_set,
                        e_int[j],
                        ρ[i],
                        q_tot[k];
                        sat_adjust_method = NM,
                        maxiter = 10,
                    )
                catch
                    ts[NM][n] = nothing
                end
            end
        end
    end
end

# folder = "sat_adjust_analysis"
folder = @__DIR__
mkpath(folder)

let
    casename(converged) = converged ? "converged" : "non-converged"
    # Full 3D scatter plot
    function plot3D(ts_no_err, ts, NM; energy_axis, include_prof, converged)
        mask = converged ? ts .≠ nothing : ts .== nothing

        c_name = converged ? "converged" : "non_converged"
        label = converged ? "converged" : "non-converged"
        e_name = energy_axis ? "energy" : "temperature"
        e_label = energy_axis ? "Internal energy" : "Temperature"
        nm_name = nameof(NM)
        filename = "3DSpace_$(e_name)_$(c_name)_$nm_name.svg"

        ρ_mask = ρ_all[mask]
        q_tot_mask = q_tot_all[mask]
        if energy_axis
            e_int_mask = e_int_all[mask]
            pts = (ρ_mask, e_int_all[mask], q_tot_mask)
        else
            T_mask = air_temperature.(ts_no_err[mask])
            pts = (ρ_mask, T_mask, q_tot_mask)
        end
        Plots.plot(
            pts...,
            color = "blue",
            seriestype = :scatter,
            markersize = 7,
            label = casename(converged),
        )
        if include_prof
            Plots.plot!(
                prof_pts...,
                color = "red",
                seriestype = :scatter,
                markersize = 7,
                label = "tested thermo profiles",
            )
        end
        plot!(
            xlabel = "Density",
            ylabel = e_label,
            zlabel = "Total specific humidity",
            title = "3D input to PhaseEquil",
            xlims = (min(ρ_all...), max(ρ_all...)),
            ylims = energy_axis ? (min(e_int_all...), max(e_int_all...)) :
                    (min(T_mask...), max(T_mask...)),
            zlims = (min(q_tot_all...), max(q_tot_all...)),
        )
        savefig(joinpath(folder, filename))
    end

    # 2D binned scatter plots
    function plot2D_slices(ts_no_err, ts, NM; energy_axis, converged)
        mask = converged ? ts .≠ nothing : ts .== nothing
        ρ_mask = ρ_all[mask]
        if energy_axis
            var = e_int_all[mask]
        else
            var = air_temperature.(ts_no_err[mask])
        end
        q_tot_mask = q_tot_all[mask]
        c_name = converged ? "converged" : "non_converged"
        label = converged ? "converged" : "non-converged"
        short_name = converged ? "C" : "NC"
        e_name = energy_axis ? "energy" : "temperature"
        nm_name = nameof(NM)
        filename = "2DSlice_$(e_name)_$(c_name)_$nm_name.svg"
        filename = joinpath(folder, filename)
        save_binned_surface_plots(
            ρ_mask,
            var,
            q_tot_mask,
            short_name,
            filename;
            xlims = (min(ρ_mask...), max(ρ_mask...)),
            ylims = energy_axis ? (min(e_int_all...), max(e_int_all...)) :
                    (min(var...), max(var...)),
            label = label,
            ref_points = prof_pts,
        )
    end

    #! format: off
    for NM in numerical_methods
    # Energy axis
    plot3D(ts_no_err[NM], ts[NM], NM; energy_axis = true, include_prof = true, converged = false);
    plot3D(ts_no_err[NM], ts[NM], NM; energy_axis = true, include_prof = true, converged = true);

    plot2D_slices(ts_no_err[NM], ts[NM], NM; energy_axis = true, converged=true);
    plot2D_slices(ts_no_err[NM], ts[NM], NM; energy_axis = true, converged=false);


    # Temperature axis
    plot3D(ts_no_err[NM], ts[NM], NM; energy_axis = false, include_prof = false, converged = false);
    plot3D(ts_no_err[NM], ts[NM], NM; energy_axis = false, include_prof = false, converged = true);
    plot2D_slices(ts_no_err[NM], ts[NM], NM; energy_axis = false, converged=true);
    plot2D_slices(ts_no_err[NM], ts[NM], NM; energy_axis = false, converged=false);
    end
    #! format: on

    convergence_percent = Dict()
    for NM in numerical_methods
        convergence_percent[NM] = count(ts[NM] .≠ nothing) / length(ts[NM])
    end
    println("Convergence percentages:")
    for (k, v) in convergence_percent
        println("$k = $v")
    end

    @warn "Note that the temperature axis for the non-converged
    plot is not necessarily accurate, since the temperatures are
    the result of a non-converged saturation adjustment. Additionally,
    non-converged cases (resulting in NaNs) do not appear in the
    non-converged plots"
end
