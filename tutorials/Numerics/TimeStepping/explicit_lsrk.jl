
include("tutorial_config.jl")


function main()
    ## These are essentially arguments passed to the
    FT = Float64
    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )
    CFL = FT(1.7)
    timeend = FT(100)
    result = run_simulation(ode_solver, CFL, timeend)
end

main()
