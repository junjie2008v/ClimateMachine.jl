# ClimateMachine supports multiple timesteppers, of different nature
# (explicit, semi-implicit, single-stage, multi-stage, single-step, multi-step,
# single-rate, multi-rate, etc). In this tutorial, we shall explore the use of
# explicit Runge-Kutta methods. For our model problem, we shall reuse the
# [rising thermal bubble tutorial](@ref Rising-termal-bubble)
# (see for details on the model and parameters)

include("tutorials/Numerics/TimeStepping/tutorial_config.jl")
FT = Float64

# After discretizing in space, the semi-discretization of
# the governing equations have the form:
# ``
#  \frac{∂ Q}{∂ t} + ∇ ⋅ (F_1(Q)) == S(Q) [eq:governing]
# ``
# [Intro something along these lines]

# A single step of an ``s``-stage Runge-Kutta (RK) method for
# solving the resulting ODE problem in [eq:governing] and can be
# expressed as the following:

# ``
# 	\boldsymbol{y}^{n+1} = \vec{y}^n + \Delta t \sum_{i=1}^{s} b_i \vec{\mathcal{T}}(\vec{Y}^i),
# ``

# where
# ``\vec{\mathcal{T}}(\vec{Y}^i)`` is the evaluation of the right-hand side
# tendency at the stage value $\vec{Y}^i$, defined at each stage
# of the RK method:

# ``
# 	\vec{Y}^i := \vec{y}^{n} + \Delta t \sum_{j=1}^{s} a_{i,j} \vec{\mathcal{T}}(\vec{Y}^j).
# ``

# ## [Low-storage Runge-Kutta methods](@id lsrk)

# Here, we use a 4-th order 5-stage explict LSRK method:
ode_solver = ClimateMachine.ExplicitSolverType(
    solver_method = LSRK54CarpenterKennedy,
)
CFL = FT(0.4)
timeend = FT(100)
run_simulation(ode_solver, CFL, timeend)

# [Show example of the simulation running using the 5-stage LSRK
# method.] Now let's try to increase the time-step size:
try
    CFL = FT(0.6)
    run_simulation(ode_solver, CFL, timeend)
catch err
    @assert isa(err, DomainError)
    println("The simulation blew up!")
end

# Uh oh, looks like the time integrator has gone unstable.
# This is a constant issue when using a fully explicit method.
# However, some explicit methods have larger stability regions
# than others. For example, let's try using a 14-stage method:
ode_solver = ClimateMachine.ExplicitSolverType(
    solver_method = LSRK144NiegemannDiehlBusch,
)
run_simulation(ode_solver, CFL, timeend)

# ## [Strong-stability-preserving Runge-Kutta methods](@id ssprk)
# ode_solver = ClimateMachine.ExplicitSolverType(
#     solver_method = SSPRK33ShuOsher,
# )
# CFL = FT(1.7)
# run_simulation(ode_solver, CFL, timeend)