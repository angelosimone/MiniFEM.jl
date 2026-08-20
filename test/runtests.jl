using Test

include("../examples/fem_assembly.jl")
include("../examples/two_springs_fem.jl")
include("../examples/axial_bar_fem.jl")

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

@testset "Four-element axial-bar regression tests" begin
    atol = 1.0e-12
    rtol = 1.0e-12

    L = 1000.0
    E = 100000.0
    A = 100.0
    q = 10.0
    P = 10000.0
    node_coordinates = [0.0, L / 4, L / 2, 3 * L / 4, L]
    element_connectivity = [
        1 2;
        2 3;
        3 4;
        4 5
    ]
    element_E = [E, E, E, E]
    element_A = [A, A, A, A]
    element_q = [q, q, q, q]
    applied_forces = [(3, P)]
    constrained_dofs = [1]
    prescribed_displacements = [0.0]

    u_expected = [0.0, 0.46875, 0.875, 0.96875, 1.0]
    reaction_expected = -20000.0

    u, reactions = solve_axial_bar(
        node_coordinates,
        element_connectivity,
        element_E,
        element_A,
        element_q,
        applied_forces,
        constrained_dofs,
        prescribed_displacements,
    )

    @test isapprox(u, u_expected; atol=atol, rtol=rtol)
    @test isapprox(reactions[1], reaction_expected; atol=atol, rtol=rtol)
end
