# Two-node axial-bar FEM operations evaluated with Gauss--Legendre quadrature.

# Integrate the stiffness matrix and consistent distributed-load vector of one two-node bar element.
function integrate_two_node_bar_gauss_legendre(
    x_e,
    E_e,
    A_e,
    q_1,
    q_2,
    num_gauss_legendre_points,
)
    K_e = zeros(2, 2)
    f_e = zeros(2)

    for (xi_g, w_g) in gauss_legendre_rule(num_gauss_legendre_points)
        J = (x_e[2] - x_e[1]) / 2
        x_g = (x_e[1] + x_e[2]) / 2 + J * xi_g
        N_g = [(1.0 - xi_g) / 2, (1.0 + xi_g) / 2]
        B_g = [-1.0 / (x_e[2] - x_e[1]), 1.0 / (x_e[2] - x_e[1])]
        E_g = E_e
        A_g = A_e
        q_g = N_g[1] * q_1 + N_g[2] * q_2

        # Stiffness contribution.
        for i in 1:2
            for j in 1:2
                K_e[i, j] += w_g * B_g[i] * E_g * A_g * B_g[j] * J
            end
        end

        # Consistent distributed-load contribution.
        for i in 1:2
            f_e[i] += w_g * N_g[i] * q_g * J
        end
    end

    return K_e, f_e
end

# Assemble axial-bar element matrices and distributed-load vectors using Gauss--Legendre quadrature.
function assemble_axial_bar_gauss_legendre!(
    K,
    f,
    node_coordinates,
    element_connectivity,
    element_E,
    element_A,
    element_q,
    num_gauss_legendre_points,
)
    num_elements = size(element_connectivity, 1)

    for e in 1:num_elements
        element_nodes = element_connectivity[e, :]
        x_e = node_coordinates[element_nodes]
        E_e = element_E[e]
        A_e = element_A[e]
        q_e = element_q[e]
        q_1 = q_e
        q_2 = q_e

        K_e, f_e = integrate_two_node_bar_gauss_legendre(
            x_e,
            E_e,
            A_e,
            q_1,
            q_2,
            num_gauss_legendre_points,
        )

        # For one displacement DOF per node, the connected node numbers
        # are also the corresponding global DOF numbers.
        dof_map = [element_connectivity[e, 1], element_connectivity[e, 2]]

        assemble_matrix!(K, K_e, dof_map)
        assemble_vector!(f, f_e, dof_map)
    end
end

# Solve a two-node axial-bar analysis using Gauss--Legendre element integration.
function solve_axial_bar_gauss_legendre(
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
    num_nodes = length(node_coordinates)
    dofs_per_node = 1
    total_dofs = num_nodes * dofs_per_node

    K = zeros(total_dofs, total_dofs)
    f = zeros(total_dofs)

    assemble_axial_bar_gauss_legendre!(
        K,
        f,
        node_coordinates,
        element_connectivity,
        element_E,
        element_A,
        element_q,
        num_gauss_legendre_points,
    )

    add_nodal_forces!(f, applied_forces)

    free_dofs = setdiff(1:total_dofs, constrained_dofs)
    K_ff = K[free_dofs, free_dofs]
    K_fc = K[free_dofs, constrained_dofs]
    f_f = f[free_dofs]
    u_c = prescribed_displacements

    # Solve K_ff * u_f = f_f - K_fc * u_c.
    u_f = K_ff \ (f_f - K_fc * u_c)

    # Reconstruct the complete global displacement vector.
    u = zeros(total_dofs)
    u[free_dofs] = u_f
    u[constrained_dofs] = u_c

    residual = K * u - f
    reactions = residual[constrained_dofs]

    return u, reactions
end

# Recover two-node axial-bar responses at Gauss--Legendre points.
function recover_axial_bar_response_gauss_legendre(
    node_coordinates,
    element_connectivity,
    element_E,
    element_A,
    u,
    num_gauss_legendre_points,
)
    num_elements = size(element_connectivity, 1)
    gauss_legendre_points = gauss_legendre_rule(num_gauss_legendre_points)
    num_points = length(gauss_legendre_points)

    gauss_point_reference_coordinates = zeros(num_elements, num_points)
    gauss_point_coordinates = zeros(num_elements, num_points)
    gauss_point_strain = zeros(num_elements, num_points)
    gauss_point_stress = zeros(num_elements, num_points)
    gauss_point_axial_force = zeros(num_elements, num_points)

    for e in 1:num_elements
        element_nodes = element_connectivity[e, :]
        x_e = node_coordinates[element_nodes]
        E_e = element_E[e]
        A_e = element_A[e]
        u_e = u[element_nodes]

        for g in 1:num_points
            xi_g, _ = gauss_legendre_points[g]
            N_g = [(1.0 - xi_g) / 2, (1.0 + xi_g) / 2]
            x_g = N_g[1] * x_e[1] + N_g[2] * x_e[2]
            B_g = [-1.0 / (x_e[2] - x_e[1]) 1.0 / (x_e[2] - x_e[1])]
            E_g = E_e
            A_g = A_e
            epsilon_g = (B_g * u_e)[1]
            sigma_g = E_g * epsilon_g
            N_g_axial = A_g * sigma_g

            gauss_point_reference_coordinates[e, g] = xi_g
            gauss_point_coordinates[e, g] = x_g
            gauss_point_strain[e, g] = epsilon_g
            gauss_point_stress[e, g] = sigma_g
            gauss_point_axial_force[e, g] = N_g_axial
        end
    end

    return (
        gauss_point_reference_coordinates,
        gauss_point_coordinates,
        gauss_point_strain,
        gauss_point_stress,
        gauss_point_axial_force,
    )
end
