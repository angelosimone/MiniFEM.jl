using Test

include("../examples/fem_assembly.jl")
include("../examples/two_springs_fem.jl")
include("../examples/axial_bar_fem.jl")
include("../examples/gauss_legendre_quadrature.jl")
include("../examples/axial_bar_gauss_legendre_fem.jl")

@testset "Two-spring regression tests" begin
    atol = 1.0e-12
    rtol = 1.0e-12

    # Reference two-spring benchmark
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

    # Additional two-spring case
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

@testset "Gauss--Legendre quadrature verification" begin
    atol = 1.0e-12
    rtol = 1.0e-12

    constant_one_point = 0.0
    linear_one_point = 0.0
    for (xi_g, w_g) in gauss_legendre_rule(1)
        constant_integrand = 5.0
        linear_integrand = 2.0 + 3.0 * xi_g
        constant_one_point += w_g * constant_integrand
        linear_one_point += w_g * linear_integrand
    end
    @test isapprox(constant_one_point, 10.0; atol=atol, rtol=rtol)
    @test isapprox(linear_one_point, 4.0; atol=atol, rtol=rtol)

    for degree in 0:3
        monomial_two_point = 0.0
        for (xi_g, w_g) in gauss_legendre_rule(2)
            monomial_two_point += w_g * xi_g^degree
        end

        monomial_exact = iseven(degree) ? 2.0 / (degree + 1) : 0.0
        @test isapprox(monomial_two_point, monomial_exact; atol=atol, rtol=rtol)
    end

    a = 0.0
    b = 3.0
    J = (b - a) / 2
    mapped_two_point = 0.0
    for (xi_g, w_g) in gauss_legendre_rule(2)
        x_g = (a + b) / 2 + J * xi_g
        mapped_two_point += w_g * (x_g^2 + 1.0) * J
    end
    @test isapprox(mapped_two_point, 12.0; atol=atol, rtol=rtol)

    x_e = [0.0, 3.0]
    E_e = 100000.0
    A_e = 100.0
    q_e = 10.0
    K_one_point, f_one_point = integrate_two_node_bar_gauss_legendre(
        x_e,
        E_e,
        A_e,
        q_e,
        q_e,
        1,
    )
    K_two_point, _ = integrate_two_node_bar_gauss_legendre(
        x_e,
        E_e,
        A_e,
        q_e,
        q_e,
        2,
    )
    K_closed_form = E_e * A_e / 3.0 * [1.0 -1.0; -1.0 1.0]
    f_closed_form = q_e * 3.0 / 2 * [1.0, 1.0]
    @test isapprox(K_one_point, K_closed_form; atol=atol, rtol=rtol)
    @test isapprox(K_two_point, K_one_point; atol=atol, rtol=rtol)
    @test isapprox(f_one_point, f_closed_form; atol=atol, rtol=rtol)

    _, f_linear_one_point = integrate_two_node_bar_gauss_legendre(
        x_e,
        E_e,
        A_e,
        2.0,
        8.0,
        1,
    )
    _, f_linear_two_point = integrate_two_node_bar_gauss_legendre(
        x_e,
        E_e,
        A_e,
        2.0,
        8.0,
        2,
    )
    @test isapprox(f_linear_one_point, [7.5, 7.5]; atol=atol, rtol=rtol)
    @test isapprox(f_linear_two_point, [6.0, 9.0]; atol=atol, rtol=rtol)
    @test isapprox(sum(f_linear_one_point), 15.0; atol=atol, rtol=rtol)
    @test isapprox(sum(f_linear_two_point), 15.0; atol=atol, rtol=rtol)
    @test !isapprox(f_linear_one_point, f_linear_two_point; atol=atol, rtol=rtol)
end

@testset "Four-element axial-bar Gauss--Legendre regression tests" begin
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
    num_gauss_legendre_points = 1

    u_expected = [0.0, 0.46875, 0.875, 0.96875, 1.0]
    reaction_expected = -20000.0
    u, reactions = solve_axial_bar_gauss_legendre(
        node_coordinates,
        element_connectivity,
        element_E,
        element_A,
        element_q,
        applied_forces,
        constrained_dofs,
        prescribed_displacements,
        num_gauss_legendre_points,
    )
    (
    gauss_point_reference_coordinates,
    gauss_point_coordinates,
    gauss_point_strain,
    gauss_point_stress,
    gauss_point_axial_force
) = recover_axial_bar_response_gauss_legendre(
        node_coordinates,
        element_connectivity,
        element_E,
        element_A,
        u,
        2,
    )

    reference_coordinate_expected = [
        -1 / sqrt(3) 1 / sqrt(3);
        -1 / sqrt(3) 1 / sqrt(3);
        -1 / sqrt(3) 1 / sqrt(3);
        -1 / sqrt(3) 1 / sqrt(3)
    ]
    physical_coordinate_expected = [
        125.0 - 125.0 / sqrt(3) 125.0 + 125.0 / sqrt(3);
        375.0 - 125.0 / sqrt(3) 375.0 + 125.0 / sqrt(3);
        625.0 - 125.0 / sqrt(3) 625.0 + 125.0 / sqrt(3);
        875.0 - 125.0 / sqrt(3) 875.0 + 125.0 / sqrt(3)
    ]
    strain_gauss_point_expected = [
        0.001875 0.001875;
        0.001625 0.001625;
        0.000375 0.000375;
        0.000125 0.000125
    ]
    stress_gauss_point_expected = [
        187.5 187.5;
        162.5 162.5;
        37.5 37.5;
        12.5 12.5
    ]
    axial_force_gauss_point_expected = [
        18750.0 18750.0;
        16250.0 16250.0;
        3750.0 3750.0;
        1250.0 1250.0
    ]
    axial_force_exact_at_gauss_points = zeros(4, 2)
    for e in 1:4
        for g in 1:2
            x_g = gauss_point_coordinates[e, g]

            if x_g < L / 2
                axial_force_exact_at_gauss_points[e, g] = q * (L - x_g) + P
            else
                axial_force_exact_at_gauss_points[e, g] = q * (L - x_g)
            end
        end
    end

    @test isapprox(u, u_expected; atol=atol, rtol=rtol)
    @test isapprox(reactions[1], reaction_expected; atol=atol, rtol=rtol)
    @test isapprox(
        gauss_point_reference_coordinates,
        reference_coordinate_expected;
        atol=atol,
        rtol=rtol,
    )
    @test isapprox(
        gauss_point_coordinates,
        physical_coordinate_expected;
        atol=atol,
        rtol=rtol,
    )
    @test isapprox(gauss_point_strain, strain_gauss_point_expected; atol=atol, rtol=rtol)
    @test isapprox(gauss_point_stress, stress_gauss_point_expected; atol=atol, rtol=rtol)
    @test isapprox(
        gauss_point_axial_force,
        axial_force_gauss_point_expected;
        atol=atol,
        rtol=rtol,
    )
    @test isapprox(gauss_point_strain[:, 1], gauss_point_strain[:, 2]; atol=atol, rtol=rtol)
    @test isapprox(gauss_point_stress[:, 1], gauss_point_stress[:, 2]; atol=atol, rtol=rtol)
    @test isapprox(
        gauss_point_axial_force[:, 1],
        gauss_point_axial_force[:, 2];
        atol=atol,
        rtol=rtol,
    )
    @test !isapprox(
        gauss_point_axial_force,
        axial_force_exact_at_gauss_points;
        atol=atol,
        rtol=rtol,
    )
end
