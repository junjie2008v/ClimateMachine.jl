using Test, Pkg
@testset "Land" begin
    include("test_heat_parameterizations.jl")
    include("test_water_parameterizations.jl")
    include("prescribed_twice.jl")
    include("freeze_thaw_alone.jl")
    include("test_physical_bc.jl")
end
