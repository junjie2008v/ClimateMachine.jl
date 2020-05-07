using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra

using CLIMA
using CLIMA.Atmos
using CLIMA.ConfigTypes
using CLIMA.Diagnostics
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics
using CLIMA.VariableTemplates

using CLIMAParameters
using CLIMAParameters.Planet: cp_d, MSLP, grav, LH_v0
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import CLIMA.DGmethods:
    vars_state_conservative,
    vars_state_auxiliary,
    vars_integrals,
    vars_reverse_integrals,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!

import CLIMA.DGmethods: boundary_state!
import CLIMA.Atmos: flux_second_order!

# -------------------- Radiation Model -------------------------- #
vars_state_conservative(::RadiationModel, FT) = @vars()
vars_state_auxiliary(::RadiationModel, FT) = @vars()
vars_integrals(::RadiationModel, FT) = @vars()
vars_reverse_integrals(::RadiationModel, FT) = @vars()

function atmos_nodal_update_auxiliary_state!(
    ::RadiationModel,
    ::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
) end

function integral_load_auxiliary_state!(
    ::RadiationModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) end
function integral_set_auxiliary_state!(::RadiationModel, aux::Vars, integ::Vars) end
function reverse_integral_load_auxiliary_state!(
    ::RadiationModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) end
function reverse_integral_set_auxiliary_state!(
    ::RadiationModel,
    aux::Vars,
    integ::Vars,
) end
function flux_radiation!(
    ::RadiationModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end

# ------------------------ Begin Radiation Model ---------------------- #
"""
  DYCOMSRadiation <: RadiationModel

Stevens et. al (2005) approximation of longwave radiative fluxes in DYCOMS.
Analytical description as a function of the liquid water path and inversion height zᵢ

* Stevens, B. et. al. (2005) "Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine Stratocumulus". Mon. Wea. Rev., 133, 1443–1462, https://doi.org/10.1175/MWR2930.1
"""
struct DYCOMSRadiation{FT} <: RadiationModel
    "mass absorption coefficient `[m^2/kg]`"
    κ::FT
    "Troposphere cooling parameter `[m^(-4/3)]`"
    α_z::FT
    "Inversion height `[m]`"
    z_i::FT
    "Density"
    ρ_i::FT
    "Large scale divergence `[s^(-1)]`"
    D_subsidence::FT
    "Radiative flux parameter `[W/m^2]`"
    F_0::FT
    "Radiative flux parameter `[W/m^2]`"
    F_1::FT
end

vars_state_auxiliary(m::DYCOMSRadiation, FT) = @vars(Rad_flux::FT)

vars_integrals(m::DYCOMSRadiation, FT) = @vars(attenuation_coeff::FT)
function integral_load_auxiliary_state!(
    m::DYCOMSRadiation,
    integrand::Vars,
    state::Vars,
    aux::Vars,
)
    FT = eltype(state)
    integrand.radiation.attenuation_coeff = state.ρ * m.κ * aux.moisture.q_liq
end
function integral_set_auxiliary_state!(
    m::DYCOMSRadiation,
    aux::Vars,
    integral::Vars,
)
    integral = integral.radiation.attenuation_coeff
    aux.∫dz.radiation.attenuation_coeff = integral
end

vars_reverse_integrals(m::DYCOMSRadiation, FT) = @vars(attenuation_coeff::FT)
function reverse_integral_load_auxiliary_state!(
    m::DYCOMSRadiation,
    integrand::Vars,
    state::Vars,
    aux::Vars,
)
    FT = eltype(state)
    #integrand.radiation.attenuation_coeff = state.ρ * m.κ * aux.moisture.q_liq
    integrand.radiation.attenuation_coeff = aux.∫dz.radiation.attenuation_coeff
end
function reverse_integral_set_auxiliary_state!(
    m::DYCOMSRadiation,
    aux::Vars,
    integral::Vars,
)
    aux.∫dnz.radiation.attenuation_coeff = integral.radiation.attenuation_coeff
end

function flux_radiation!(
    m::DYCOMSRadiation,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(flux)
    z = altitude(atmos, aux)
    Δz_i = max(z - m.z_i, -zero(FT))
    # Constants
    upward_flux_from_cloud = m.F_0 * exp(-aux.∫dnz.radiation.attenuation_coeff)
    upward_flux_from_sfc = m.F_1 * exp(-aux.∫dz.radiation.attenuation_coeff)
    free_troposphere_flux =
        m.ρ_i *
        FT(cp_d(atmos.param_set)) *
        m.D_subsidence *
        m.α_z *
        cbrt(Δz_i) *
        (Δz_i / 4 + m.z_i)
    F_rad =
        upward_flux_from_sfc + upward_flux_from_cloud + free_troposphere_flux
    ẑ = vertical_unit_vector(atmos, aux)
    flux.ρe += F_rad * ẑ
end
# -------------------------- End Radiation Model ------------------------ #

"""
  Initial Condition for DYCOMS_RF01 LES
@article{doi:10.1175/MWR2930.1,
author = {Stevens, Bjorn and Moeng, Chin-Hoh and Ackerman,
          Andrew S. and Bretherton, Christopher S. and Chlond,
          Andreas and de Roode, Stephan and Edwards, James and Golaz,
          Jean-Christophe and Jiang, Hongli and Khairoutdinov,
          Marat and Kirkpatrick, Michael P. and Lewellen, David C. and Lock, Adrian and
          Maeller, Frank and Stevens, David E. and Whelan, Eoin and Zhu, Ping},
title = {Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine Stratocumulus},
journal = {Monthly Weather Review},
volume = {133},
number = {6},
pages = {1443-1462},
year = {2005},
doi = {10.1175/MWR2930.1},
URL = {https://doi.org/10.1175/MWR2930.1},
eprint = {https://doi.org/10.1175/MWR2930.1}
}
"""
function init_dycoms!(bl, state, aux, (x, y, z), t)
    FT = eltype(state)

    z = altitude(bl, aux)

    # These constants are those used by Stevens et al. (2005)
    qref = FT(8.0e-3)
    q_pt_sfc = PhasePartition(qref)
    Rm_sfc = gas_constant_air(bl.param_set, q_pt_sfc)
    T_sfc = FT(290.4)
    _MSLP = FT(MSLP(bl.param_set))
    _grav = FT(grav(bl.param_set))

    # Specify moisture profiles
    q_liq = FT(0)
    q_ice = FT(0)
    zb = FT(600)         # initial cloud bottom
    zi = FT(840)         # initial cloud top

    if z <= zi
        θ_liq = FT(289.0)
        q_tot = qref
    else
        θ_liq = FT(297.0) + (z - zi)^(FT(1 / 3))
        #q_tot = FT(1.5e-3)
        q_tot = FT(1.0e-3)
    end

    ugeo = FT(7)
    vgeo = FT(-5.5)
    u, v, w = ugeo, vgeo, FT(0)

    # Perturb initial state to break symmetry and trigger turbulent convection
    r1 = FT(rand(Uniform(-0.001, 0.001)))
    if z <= 200.0
        θ_liq += r1 * θ_liq
    end

    # Pressure
    H = Rm_sfc * T_sfc / _grav
    p = _MSLP * exp(-z / H)

    # Density, Temperature
    # SM There is a difference in the order of arguments in LiquidIcePotTempSHumEquil_given_pressure
    ts = LiquidIcePotTempSHumEquil_given_pressure(bl.param_set, θ_liq, p, q_tot)
    ρ = air_density(ts)

    e_kin = FT(1 / 2) * FT((u^2 + v^2 + w^2))
    e_pot = gravitational_potential(bl.orientation, aux)
    E = ρ * total_energy(e_kin, e_pot, ts)

    state.ρ = ρ
    state.ρu = SVector(ρ * u, ρ * v, ρ * w)
    state.ρe = E
    state.moisture.ρq_tot = ρ * q_tot

    return nothing
end

function config_dycoms(FT, N, resolution, xmax, ymax, zmax)
    # Reference state
    T_min = FT(289)
    T_s = FT(290.4)
    _grav = FT(grav(param_set))
    _cp_d = FT(cp_d(param_set))
    Γ_lapse = _grav / _cp_d
    T = LinearTemperatureProfile(T_min, T_s, Γ_lapse)
    rel_hum = FT(0)
    ref_state = HydrostaticState(T, rel_hum)

    # Radiation model
    κ = FT(85)
    α_z = FT(1)
    z_i = FT(840)
    ρ_i = FT(1.13)

    D_subsidence = FT(3.75e-6)

    F_0 = FT(70)
    F_1 = FT(22)
    radiation = DYCOMSRadiation{FT}(κ, α_z, z_i, ρ_i, D_subsidence, F_0, F_1)

    # Sources
    f_coriolis = FT(1.03e-4)
    u_geostrophic = FT(7.0)
    v_geostrophic = FT(-5.5)
    w_ref = FT(0)
    u_relaxation = SVector(u_geostrophic, v_geostrophic, w_ref)
    # Sponge
    c_sponge = 1.0
    # Rayleigh damping
    zsponge = FT(980.0)
    rayleigh_sponge =
        RayleighSponge{FT}(zmax, zsponge, c_sponge, u_relaxation, 2)
    # Geostrophic forcing
    geostrophic_forcing =
        GeostrophicForcing{FT}(f_coriolis, u_geostrophic, v_geostrophic)

    # Boundary conditions
    # SGS Filter constants
    C_smag = FT(0.21) # 0.21 for stable testing, 0.18 in practice
    C_drag = FT(0.0011)
    LHF = FT(115)
    SHF = FT(15)
    ics = init_dycoms!
    moisture_flux = LHF / FT(LH_v0(param_set))

    source = (
        Gravity(),
        rayleigh_sponge,
        Subsidence{FT}(D_subsidence),
        geostrophic_forcing,
    )

    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        ref_state = ref_state,
        turbulence = Vreman{FT}(C_smag),
        moisture = EquilMoist{FT}(maxiter = 1, tolerance = FT(100)),
        radiation = radiation,
        source = source,
        boundarycondition = (
            AtmosBC(
                momentum = Impenetrable(DragLaw(
                    (state, aux, t, normPu) -> C_drag,
                )),
                energy = PrescribedEnergyFlux((state, aux, t) -> LHF + SHF),
                moisture = PrescribedMoistureFlux(
                    (state, aux, t) -> moisture_flux,
                ),
            ),
            AtmosBC(),
        ),
        init_state_conservative = ics,
    )

    ode_solver =
        CLIMA.ExplicitSolverType(solver_method = LSRK144NiegemannDiehlBusch)
    
    #= Choose multi-rate explicit solver
    ode_solver = CLIMA.MultirateSolverType(
        linear_model = AtmosAcousticGravityLinearModel,
        slow_method = LSRK144NiegemannDiehlBusch,
        fast_method = LSRK144NiegemannDiehlBusch,
        timestep_ratio = 14,
    )=#
    
    config = CLIMA.AtmosLESConfiguration(
        "DYCOMS",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_dycoms!,
        solver_type = ode_solver,
        model = model,
    )
    return config
end

function config_diagnostics(driver_config)
    interval = "10000steps"
    dgngrp = setup_atmos_default_diagnostics(interval, driver_config.name)
    return CLIMA.DiagnosticsConfiguration([dgngrp])
end

function main()
    CLIMA.init()

    FT = Float64

    # DG polynomial order
    N = 4

    # Domain resolution and size
    Δh = FT(40)
    Δv = FT(20)
    resolution = (Δh, Δh, Δv)

    xmax = FT(4000)
    ymax = FT(1000)
    zmax = FT(1500)

    t0 = FT(0)
    timeend = FT(200)

    driver_config = config_dycoms(FT, N, resolution, xmax, ymax, zmax)
    solver_config = CLIMA.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number=FT(1.0),
    )
    dgn_config = config_diagnostics(driver_config)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init = false)
        Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
        nothing
    end

    result = CLIMA.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbtmarfilter,),
        check_euclidean_distance = true,
    )
end

main()
