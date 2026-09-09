# Three-node quadratic axial-bar FEM operations evaluated with Gauss--Legendre quadrature.

# Construct the quadratic shape functions and their reference-coordinate derivatives.
function quadratic_bar_shape_functions(xi)
    N = [
        0.5 * xi * (xi - 1.0),
        1.0 - xi^2,
        0.5 * xi * (xi + 1.0),
    ]
    dN_dxi = [
        xi - 0.5,
        -2.0 * xi,
        xi + 0.5,
    ]

    return N, dN_dxi
end

# Integrate the stiffness matrix and consistent distributed-load vector of one regular three-node bar element.
function integrate_three_node_bar_gauss_legendre(
    x_e,
    E_e,
    A_e,
    q_e,
    num_gauss_legendre_points,
)
    K_e = zeros(3, 3)
    f_e = zeros(3)

    for (xi_g, w_g) in gauss_legendre_rule(num_gauss_legendre_points)
        N_g, dN_dxi_g = quadratic_bar_shape_functions(xi_g)
        J_g = sum(dN_dxi_g[i] * x_e[i] for i in 1:3)
        B_g = dN_dxi_g / J_g
        E_g = E_e
        A_g = A_e
        q_g = q_e

        # Stiffness contribution.
        for i in 1:3
            for j in 1:3
                K_e[i, j] += w_g * B_g[i] * E_g * A_g * B_g[j] * J_g
            end
        end

        # Consistent distributed-load contribution.
        for i in 1:3
            f_e[i] += w_g * N_g[i] * q_g * J_g
        end
    end

    return K_e, f_e
end

# Assemble quadratic axial-bar element matrices and distributed-load vectors using Gauss--Legendre quadrature.
function assemble_quadratic_axial_bar_gauss_legendre!(
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

        K_e, f_e = integrate_three_node_bar_gauss_legendre(
            x_e,
            E_e,
            A_e,
            q_e,
            num_gauss_legendre_points,
        )

        # For one displacement DOF per node, the connected node numbers
        # are also the corresponding global DOF numbers.
        dof_map = [
            element_connectivity[e, 1],
            element_connectivity[e, 2],
            element_connectivity[e, 3],
        ]

        assemble_matrix!(K, K_e, dof_map)
        assemble_vector!(f, f_e, dof_map)
    end
end

# Solve a three-node quadratic axial-bar analysis using Gauss--Legendre element integration.
function solve_quadratic_axial_bar_gauss_legendre(
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

    assemble_quadratic_axial_bar_gauss_legendre!(
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

# Recover three-node quadratic axial-bar responses at selected reference coordinates.
function recover_quadratic_axial_bar_response(
    node_coordinates,
    element_connectivity,
    element_E,
    element_A,
    u,
    response_reference_coordinates,
)
    num_elements = size(element_connectivity, 1)
    num_response_points = length(response_reference_coordinates)

    response_coordinates = zeros(num_elements, num_response_points)
    response_strain = zeros(num_elements, num_response_points)
    response_stress = zeros(num_elements, num_response_points)
    response_axial_force = zeros(num_elements, num_response_points)

    for e in 1:num_elements
        element_nodes = element_connectivity[e, :]
        x_e = node_coordinates[element_nodes]
        E_e = element_E[e]
        A_e = element_A[e]
        u_e = u[element_nodes]

        for r in 1:num_response_points
            xi_r = response_reference_coordinates[r]
            N_r, dN_dxi_r = quadratic_bar_shape_functions(xi_r)
            x_r = sum(N_r[i] * x_e[i] for i in 1:3)
            J_r = sum(dN_dxi_r[i] * x_e[i] for i in 1:3)
            B_r = dN_dxi_r / J_r
            epsilon_r = sum(B_r[i] * u_e[i] for i in 1:3)
            sigma_r = E_e * epsilon_r
            N_r_axial = A_e * sigma_r

            response_coordinates[e, r] = x_r
            response_strain[e, r] = epsilon_r
            response_stress[e, r] = sigma_r
            response_axial_force[e, r] = N_r_axial
        end
    end

    return (
        response_coordinates,
        response_strain,
        response_stress,
        response_axial_force,
    )
end
