##### Array wrappers for numerical Fluxes

function numerical_flux_first_order_arr!(
    numerical_flux_first_order,
    balance_law,
    flux::AbstractArray,
    normal_vector::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
    face_direction,
)
    FT = eltype(flux)
    numerical_flux_first_order!(
        numerical_flux_first_order,
        balance_law,
        Vars{vars_state(balance_law, Prognostic(), FT)}(flux),
        SVector(normal_vector),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁻),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁻),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁺),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁺),
        t,
        face_direction,
    )
end

function numerical_flux_second_order_arr!(
    numerical_flux_second_order,
    balance_law,
    flux::AbstractArray,
    normal_vector::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_gradient_flux⁻::AbstractArray,
    state_hyperdiffusive⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_gradient_flux⁺::AbstractArray,
    state_hyperdiffusive⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
)
    FT = eltype(flux)
    numerical_flux_second_order!(
        numerical_flux_second_order,
        balance_law,
        Vars{vars_state(balance_law, Prognostic(), FT)}(flux),
        SVector(normal_vector),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁻),
        Vars{vars_state(balance_law, GradientFlux(), FT)}(state_gradient_flux⁻),
        Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
            state_hyperdiffusive⁻,
        ),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁻),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁺),
        Vars{vars_state(balance_law, GradientFlux(), FT)}(state_gradient_flux⁺),
        Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
            state_hyperdiffusive⁺,
        ),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁺),
        t,
    )
end

function numerical_flux_gradient_arr!(
    numerical_flux_gradient,
    balance_law,
    local_transform_gradient::AbstractArray,
    normal_vector::AbstractArray,
    local_transform⁻::AbstractArray,
    local_state_prognostic⁻::AbstractArray,
    local_state_auxiliary⁻::AbstractArray,
    local_transform⁺::AbstractArray,
    local_state_prognostic⁺::AbstractArray,
    local_state_auxiliary⁺::AbstractArray,
    t,
)
    FT = eltype(local_transform_gradient)
    numerical_flux_gradient!(
        numerical_flux_gradient,
        balance_law,
        local_transform_gradient,
        SVector(normal_vector),
        Vars{vars_state(balance_law, Gradient(), FT)}(local_transform⁻),
        Vars{vars_state(balance_law, Prognostic(), FT)}(
            local_state_prognostic⁻,
        ),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(local_state_auxiliary⁻),
        Vars{vars_state(balance_law, Gradient(), FT)}(local_transform⁺),
        Vars{vars_state(balance_law, Prognostic(), FT)}(
            local_state_prognostic⁺,
        ),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(local_state_auxiliary⁺),
        t,
    )
end

function transform_post_gradient_laplacian_arr!(
    balance_law,
    local_state_hyperdiffusion::AbstractArray,
    l_grad_lap::AbstractArray,
    local_state_prognostic::AbstractArray,
    local_state_auxiliary::AbstractArray,
    t,
)
    FT = eltype(local_state_hyperdiffusion)
    transform_post_gradient_laplacian!(
        balance_law,
        Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
            local_state_hyperdiffusion,
        ),
        Grad{vars_state(balance_law, GradientLaplacian(), FT)}(l_grad_lap),
        Vars{vars_state(balance_law, Prognostic(), FT)}(local_state_prognostic),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(local_state_auxiliary),
        t,
    )
end

function numerical_flux_higher_order_arr!(
    hyperviscnumflux,
    balance_law,
    local_state_hyperdiffusion::AbstractArray,
    normal_vector::AbstractArray,
    l_lap⁻::AbstractArray,
    local_state_prognostic⁻::AbstractArray,
    local_state_auxiliary⁻::AbstractArray,
    l_lap⁺::AbstractArray,
    local_state_prognostic⁺::AbstractArray,
    local_state_auxiliary⁺::AbstractArray,
    t,
)
    FT = eltype(local_state_hyperdiffusion)

    numerical_flux_higher_order!(
        hyperviscnumflux,
        balance_law,
        Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
            local_state_hyperdiffusion,
        ),
        normal_vector,
        Vars{vars_state(balance_law, GradientLaplacian(), FT)}(l_lap⁻),
        Vars{vars_state(balance_law, Prognostic(), FT)}(
            local_state_prognostic⁻,
        ),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(local_state_auxiliary⁻),
        Vars{vars_state(balance_law, GradientLaplacian(), FT)}(l_lap⁺),
        Vars{vars_state(balance_law, Prognostic(), FT)}(
            local_state_prognostic⁺,
        ),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(local_state_auxiliary⁺),
        t,
    )
end

#####
##### Boundary
#####

function numerical_boundary_flux_first_order_arr!(
    numerical_flux_first_order,
    bc,
    balance_law,
    flux::AbstractArray,
    normal_vector::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
    face_direction,
    state_prognostic_bottom1::AbstractArray,
    state_auxiliary_bottom1::AbstractArray,
)
    FT = eltype(flux)
    numerical_boundary_flux_first_order!(
        numerical_flux_first_order,
        bc,
        balance_law,
        Vars{vars_state(balance_law, Prognostic(), FT)}(flux),
        SVector(normal_vector),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁻),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁻),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁺),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁺),
        t,
        face_direction,
        Vars{vars_state(balance_law, Prognostic(), FT)}(
            state_prognostic_bottom1,
        ),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary_bottom1),
    )
end

function numerical_boundary_flux_second_order_arr!(
    numerical_flux_second_order,
    bc,
    balance_law,
    flux::AbstractArray,
    normal_vector::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_gradient_flux⁻::AbstractArray,
    state_hyperdiffusive⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_gradient_flux⁺::AbstractArray,
    state_hyperdiffusive⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
    state_prognostic_bottom1::AbstractArray,
    state_gradient_flux_bottom1::AbstractArray,
    state_auxiliary_bottom1::AbstractArray,
)
    FT = eltype(flux)

    numerical_boundary_flux_second_order!(
        numerical_flux_second_order,
        bc,
        balance_law,
        Vars{vars_state(balance_law, Prognostic(), FT)}(flux),
        SVector(normal_vector),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁻),
        Vars{vars_state(balance_law, GradientFlux(), FT)}(state_gradient_flux⁻),
        Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
            state_hyperdiffusive⁻,
        ),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁻),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁺),
        Vars{vars_state(balance_law, GradientFlux(), FT)}(state_gradient_flux⁺),
        Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
            state_hyperdiffusive⁺,
        ),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁺),
        t,
        Vars{vars_state(balance_law, Prognostic(), FT)}(
            state_prognostic_bottom1,
        ),
        Vars{vars_state(balance_law, GradientFlux(), FT)}(
            state_gradient_flux_bottom1,
        ),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary_bottom1),
    )
end

function numerical_boundary_flux_gradient_arr!(
    numerical_flux_gradient,
    bc,
    balance_law,
    flux,
    normal_vector::AbstractArray,
    state_gradient⁻::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_gradient⁺::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
    state_prognostic_bottom1::AbstractArray,
    state_auxiliary_bottom1::AbstractArray,
)

    FT = eltype(flux)
    numerical_boundary_flux_gradient!(
        numerical_flux_gradient,
        bc,
        balance_law,
        flux,
        SVector(normal_vector),
        Vars{vars_state(balance_law, Gradient(), FT)}(state_gradient⁻),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁻),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁻),
        Vars{vars_state(balance_law, Gradient(), FT)}(state_gradient⁺),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁺),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁺),
        t,
        Vars{vars_state(balance_law, Prognostic(), FT)}(
            state_prognostic_bottom1,
        ),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary_bottom1),
    )
end

function numerical_boundary_flux_divergence_arr!(
    divgradnumpenalty,
    bc,
    balance_law,
    l_div::AbstractArray,
    normal_vector::AbstractArray,
    l_grad⁻::AbstractArray,
    l_grad⁺::AbstractArray,
)
    FT = eltype(l_div)
    n̂ = SVector(normal_vector)
    numerical_boundary_flux_divergence!(
        divgradnumpenalty,
        bc,
        balance_law,
        Vars{vars_state(balance_law, GradientLaplacian(), FT)}(l_div),
        n̂,
        Grad{vars_state(balance_law, GradientLaplacian(), FT)}(l_grad⁻),
        Grad{vars_state(balance_law, GradientLaplacian(), FT)}(l_grad⁺),
    )
end

function numerical_boundary_flux_higher_order_arr!(
    hyperviscnumflux,
    bc,
    balance_law,
    local_state_hyperdiffusion::AbstractArray,
    normal_vector::AbstractArray,
    l_lap⁻::AbstractArray,
    local_state_prognostic⁻::AbstractArray,
    local_state_auxiliary⁻::AbstractArray,
    l_lap⁺::AbstractArray,
    local_state_prognostic⁺::AbstractArray,
    local_state_auxiliary⁺::AbstractArray,
    t,
)

    FT = eltype(l_lap⁻)

    numerical_boundary_flux_higher_order!(
        hyperviscnumflux,
        bc,
        balance_law,
        Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
            local_state_hyperdiffusion,
        ),
        normal_vector,
        Vars{vars_state(balance_law, GradientLaplacian(), FT)}(l_lap⁻),
        Vars{vars_state(balance_law, Prognostic(), FT)}(
            local_state_prognostic⁻,
        ),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(local_state_auxiliary⁻),
        Vars{vars_state(balance_law, GradientLaplacian(), FT)}(l_lap⁺),
        Vars{vars_state(balance_law, Prognostic(), FT)}(
            local_state_prognostic⁺,
        ),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(local_state_auxiliary⁺),
        t,
    )

end
