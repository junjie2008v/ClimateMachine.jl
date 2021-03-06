##### Moisture tendencies

#####
##### First order fluxes
#####

function flux(::Advect{Tracers{N}}, atmos, args) where {N}
    @unpack state = args
    u = state.ρu / state.ρ
    return (state.tracers.ρχ .* u')'
end

#####
##### Second order fluxes
#####

function flux(::Diffusion{Tracers{N}}, atmos, args) where {N}
    @unpack state, aux, diffusive = args
    @unpack D_t = args.precomputed.turbulence
    d_χ = (-D_t) * aux.tracers.δ_χ' .* diffusive.tracers.∇χ
    return d_χ * state.ρ
end
