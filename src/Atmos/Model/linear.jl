using CLIMAParameters.Planet: R_d, cv_d, T_0, e_int_v0, e_int_i0

"""
    linearized_air_pressure(ρ, ρe_tot, ρe_pot, ρq_tot=0, ρq_liq=0, ρq_ice=0)

The air pressure, linearized around a dry rest state, from the equation of state
(ideal gas law) where:

 - `ρ` (moist-)air density
 - `ρe_tot` total energy density
 - `ρe_pot` potential energy density
and, optionally,
 - `ρq_tot` total water density
 - `ρq_liq` liquid water density
 - `ρq_ice` ice density
"""
function linearized_air_pressure(
    param_set::AbstractParameterSet,
    ρ::FT,
    ρe_tot::FT,
    ρe_pot::FT,
    ρq_tot::FT = FT(0),
    ρq_liq::FT = FT(0),
    ρq_ice::FT = FT(0),
) where {FT <: Real, PS}
    _R_d::FT = R_d(param_set)
    _cv_d::FT = cv_d(param_set)
    _T_0::FT = T_0(param_set)
    _e_int_v0::FT = e_int_v0(param_set)
    _e_int_i0::FT = e_int_i0(param_set)
    return ρ * _R_d * _T_0 +
           _R_d / _cv_d * (
        ρe_tot - ρe_pot - (ρq_tot - ρq_liq) * _e_int_v0 +
        ρq_ice * (_e_int_i0 + _e_int_v0)
    )
end

@inline function linearized_pressure(
    ::DryModel,
    param_set::AbstractParameterSet,
    orientation::Orientation,
    state::Vars,
    aux::Vars,
)
    ρe_pot = state.ρ * gravitational_potential(orientation, aux)
    return linearized_air_pressure(param_set, state.ρ, state.ρe, ρe_pot)
end
@inline function linearized_pressure(
    ::EquilMoist,
    param_set::AbstractParameterSet,
    orientation::Orientation,
    state::Vars,
    aux::Vars,
)
    ρe_pot = state.ρ * gravitational_potential(orientation, aux)
    linearized_air_pressure(
        param_set,
        state.ρ,
        state.ρe,
        ρe_pot,
        state.moisture.ρq_tot,
    )
end

abstract type AtmosLinearModel <: BalanceLaw end

function vars_state(lm::AtmosLinearModel, st::Prognostic, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        turbulence::vars_state(lm.atmos.turbulence, st, FT)
        hyperdiffusion::vars_state(lm.atmos.hyperdiffusion, st, FT)
        moisture::vars_state(lm.atmos.moisture, st, FT)
    end
end
vars_state(lm::AtmosLinearModel, ::AbstractStateType, FT) = @vars()
vars_state(lm::AtmosLinearModel, st::Auxiliary, FT) =
    vars_state(lm.atmos, st, FT)


function update_auxiliary_state!(
    dg::DGModel,
    lm::AtmosLinearModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    return false
end
function flux_second_order!(
    lm::AtmosLinearModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end
integral_load_auxiliary_state!(
    lm::AtmosLinearModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) = nothing
integral_set_auxiliary_state!(lm::AtmosLinearModel, aux::Vars, integ::Vars) =
    nothing
reverse_integral_load_auxiliary_state!(
    lm::AtmosLinearModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) = nothing
reverse_integral_set_auxiliary_state!(
    lm::AtmosLinearModel,
    aux::Vars,
    integ::Vars,
) = nothing
flux_second_order!(
    lm::AtmosLinearModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
) = nothing
function wavespeed(
    lm::AtmosLinearModel,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    ref = aux.ref_state
    return soundspeed_air(lm.atmos.param_set, ref.T)
end

function boundary_state!(
    nf::NumericalFluxFirstOrder,
    atmoslm::AtmosLinearModel,
    args...,
)
    atmos_boundary_state!(nf, AtmosBC(), atmoslm, args...)
end
function boundary_state!(
    nf::NumericalFluxSecondOrder,
    atmoslm::AtmosLinearModel,
    args...,
)
    nothing
end
init_state_auxiliary!(
    lm::AtmosLinearModel,
    aux::MPIStateArray,
    grid,
    direction,
) = nothing
init_state_prognostic!(
    lm::AtmosLinearModel,
    state::Vars,
    aux::Vars,
    coords,
    t,
) = nothing


struct AtmosAcousticLinearModel{M} <: AtmosLinearModel
    atmos::M
    function AtmosAcousticLinearModel(atmos::M) where {M}
        if atmos.ref_state === NoReferenceState()
            error("AtmosAcousticLinearModel needs a model with a reference state")
        end
        new{M}(atmos)
    end
end

function flux_first_order!(
    lm::AtmosAcousticLinearModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    FT = eltype(state)
    ref = aux.ref_state
    e_pot = gravitational_potential(lm.atmos.orientation, aux)

    flux.ρ = state.ρu
    pL = linearized_pressure(
        lm.atmos.moisture,
        lm.atmos.param_set,
        lm.atmos.orientation,
        state,
        aux,
    )
    flux.ρu += pL * I
    flux.ρe = ((ref.ρe + ref.p) / ref.ρ - e_pot) * state.ρu
    nothing
end
source!(::AtmosAcousticLinearModel, _...) = nothing

struct AtmosAcousticGravityLinearModel{M} <: AtmosLinearModel
    atmos::M
    function AtmosAcousticGravityLinearModel(atmos::M) where {M}
        if atmos.ref_state === NoReferenceState()
            error("AtmosAcousticGravityLinearModel needs a model with a reference state")
        end
        new{M}(atmos)
    end
end
function flux_first_order!(
    lm::AtmosAcousticGravityLinearModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    FT = eltype(state)
    ref = aux.ref_state
    e_pot = gravitational_potential(lm.atmos.orientation, aux)

    flux.ρ = state.ρu
    pL = linearized_pressure(
        lm.atmos.moisture,
        lm.atmos.param_set,
        lm.atmos.orientation,
        state,
        aux,
    )
    flux.ρu += pL * I
    flux.ρe = ((ref.ρe + ref.p) / ref.ρ) * state.ρu
    nothing
end
function source!(
    lm::AtmosAcousticGravityLinearModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    ::NTuple{1, Dir},
) where {Dir <: Direction}
    if Dir === VerticalDirection || Dir === EveryDirection
        ∇Φ = ∇gravitational_potential(lm.atmos.orientation, aux)
        source.ρu -= state.ρ * ∇Φ
    end
    nothing
end

function numerical_flux_first_order!(
    numerical_flux::RoeNumericalFlux,
    balance_law::AtmosLinearModel,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_conservative⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_conservative⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}
    @assert balance_law.atmos.moisture isa DryModel

    numerical_flux_first_order!(
        CentralNumericalFluxFirstOrder(),
        balance_law,
        fluxᵀn,
        normal_vector,
        state_conservative⁻,
        state_auxiliary⁻,
        state_conservative⁺,
        state_auxiliary⁺,
        t,
        direction,
    )

    atmos = balance_law.atmos
    param_set = atmos.param_set

    ρu⁻ = state_conservative⁻.ρu

    ref_ρ⁻ = state_auxiliary⁻.ref_state.ρ
    ref_ρe⁻ = state_auxiliary⁻.ref_state.ρe
    ref_T⁻ = state_auxiliary⁻.ref_state.T
    ref_p⁻ = state_auxiliary⁻.ref_state.p
    ref_h⁻ = (ref_ρe⁻ + ref_p⁻) / ref_ρ⁻
    ref_c⁻ = soundspeed_air(param_set, ref_T⁻)

    pL⁻ = linearized_pressure(
        atmos.moisture,
        param_set,
        atmos.orientation,
        state_conservative⁻,
        state_auxiliary⁻,
    )

    ρu⁺ = state_conservative⁺.ρu

    ref_ρ⁺ = state_auxiliary⁺.ref_state.ρ
    ref_ρe⁺ = state_auxiliary⁺.ref_state.ρe
    ref_T⁺ = state_auxiliary⁺.ref_state.T
    ref_p⁺ = state_auxiliary⁺.ref_state.p
    ref_h⁺ = (ref_ρe⁺ + ref_p⁺) / ref_ρ⁺
    ref_c⁺ = soundspeed_air(param_set, ref_T⁺)

    pL⁺ = linearized_pressure(
        atmos.moisture,
        param_set,
        atmos.orientation,
        state_conservative⁺,
        state_auxiliary⁺,
    )

    # not sure if arithmetic averages are a good idea here
    h̃ = (ref_h⁻ + ref_h⁺) / 2
    c̃ = (ref_c⁻ + ref_c⁺) / 2

    ΔpL = pL⁺ - pL⁻
    Δρuᵀn = (ρu⁺ - ρu⁻)' * normal_vector

    fluxᵀn.ρ -= ΔpL / 2c̃
    fluxᵀn.ρu -= c̃ * Δρuᵀn * normal_vector / 2
    fluxᵀn.ρe -= h̃ * ΔpL / 2c̃
end

function numerical_flux_first_order!(
    numerical_flux::LMARNumericalFlux,
    balance_law::AtmosLinearModel,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_conservative⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_conservative⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}
    @assert balance_law.atmos.moisture isa DryModel
    
    atmos = balance_law.atmos
    param_set = atmos.param_set
    
    numerical_flux_first_order!(
        CentralNumericalFluxFirstOrder(),
        balance_law,
        fluxᵀn,
        normal_vector,
        state_conservative⁻,
        state_auxiliary⁻,
        state_conservative⁺,
        state_auxiliary⁺,
        t,
        direction,
    )

    #=
    # Unpack reference parameters on left (minus) side
    ρu⁻ = state_conservative⁻.ρu
    ρ⁻ = state_conservative⁻.ρ
    u⁻ = ρu⁻/ρ⁻
    uᵀn⁻ = u⁻' * normal_vector
    ref_ρ⁻ = state_auxiliary⁻.ref_state.ρ
    ref_ρe⁻ = state_auxiliary⁻.ref_state.ρe
    ref_T⁻ = state_auxiliary⁻.ref_state.T
    ref_p⁻ = state_auxiliary⁻.ref_state.p
    ref_h⁻ = (ref_ρe⁻ + ref_p⁻) / ref_ρ⁻
    ref_c⁻ = soundspeed_air(param_set, ref_T⁻)

    pL⁻ = linearized_pressure(
        atmos.moisture,
        param_set,
        atmos.orientation,
        state_conservative⁻,
        state_auxiliary⁻,
    )

    # Unpack reference parameters on right (plus) side
    ρu⁺ = state_conservative⁺.ρu
    ρ⁺ = state_conservative⁺.ρ
    u⁺  = ρu⁺/ρ⁺
    uᵀn⁺ = u⁺' * normal_vector
    ref_ρ⁺ = state_auxiliary⁺.ref_state.ρ
    ref_ρe⁺ = state_auxiliary⁺.ref_state.ρe
    ref_T⁺ = state_auxiliary⁺.ref_state.T
    ref_p⁺ = state_auxiliary⁺.ref_state.p
    ref_h⁺ = (ref_ρe⁺ + ref_p⁺) / ref_ρ⁺
    ref_c⁺ = soundspeed_air(param_set, ref_T⁺)

    pL⁺ = linearized_pressure(
        atmos.moisture,
        param_set,
        atmos.orientation,
        state_conservative⁺,
        state_auxiliary⁺,
    )

    # Select +/- state
    ρ_b = u_half > FT(0) ? ρ⁻ : ρ⁺
    ρh_b = u_half > FT(0) ? ref_ρ⁻*ref_h⁻ : ref_ρ⁺*_ref_h⁺

    # Compute linearised pressure contribution
    u_half = 1/2 * (uᵀn⁺ + uᵀn⁻)
    p_half = 1/2 * (pL⁺ + pL⁻) 
    
    # Upwind fluxes for linear terms
    fluxᵀn.ρ = ρ_b * u_half
    fluxᵀn.ρu = p_half * normal_vector
    fluxᵀn.ρe =  ρh_b * u_half
    =# 
end
