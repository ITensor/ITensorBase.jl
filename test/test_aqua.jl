using ITensorBase: ITensorBase
using Aqua: Aqua
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(ITensorBase; ambiguities = false, persistent_tasks = false)
end
