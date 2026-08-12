using Test

include("../examples/two_springs_fem.jl")

@testset "Two-spring regression tests" begin
    atol = 1.0e-12
    rtol = 1.0e-12

    # Canonical two-spring benchmark
    k_1 = 1.0
    k_2 = 2.0
    applied_force = 10.0
    u_expected = [15.0, 0.0, 10.0]
    strain_energy_expected = 75.0

    u, strain_energy_global = solve_two_springs(k_1, k_2, applied_force)

    @test isapprox(u, u_expected; atol=atol, rtol=rtol)
    @test isapprox(
        strain_energy_global,
        strain_energy_expected;
        atol=atol,
        rtol=rtol,
    )

    # Noncanonical two-spring benchmark
    k_1 = 2.0
    k_2 = 3.0
    applied_force = 12.0
    u_expected = [10.0, 0.0, 6.0]
    strain_energy_expected = 60.0

    u, strain_energy_global = solve_two_springs(k_1, k_2, applied_force)

    @test isapprox(u, u_expected; atol=atol, rtol=rtol)
    @test isapprox(
        strain_energy_global,
        strain_energy_expected;
        atol=atol,
        rtol=rtol,
    )
end
