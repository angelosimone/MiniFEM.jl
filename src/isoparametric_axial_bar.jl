# Isoparametric axial-bar FEM operations evaluated with Gauss--Legendre quadrature.

# Evaluate the selected axial-bar shape functions and reference-coordinate derivatives.
function bar_shape_functions(xi, interpolation)
    if interpolation == "linear"
        N = [(1.0 - xi) / 2, (1.0 + xi) / 2]
        dN_dxi = [-1.0 / 2, 1.0 / 2]
    elseif interpolation == "quadratic"
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
    else
        error("Axial-bar interpolation must be \"linear\" or \"quadratic\".")
    end

    return N, dN_dxi
end

# Integrate the stiffness matrix and consistent distributed-load vector of one axial-bar element.
function integrate_axial_bar_gauss_legendre(
    x_e,
    E_e,
    A_e,
    q_e,
    num_gauss_legendre_points,
    interpolation,
)
    num_element_dofs = length(x_e)
    K_e = zeros(num_element_dofs, num_element_dofs)
    f_e = zeros(num_element_dofs)

    for (xi_g, w_g) in gauss_legendre_rule(num_gauss_legendre_points)
        N, dN_dxi = bar_shape_functions(xi_g, interpolation)
        J = sum(dN_dxi[i] * x_e[i] for i in 1:num_element_dofs)
        B = dN_dxi / J
        q_g = sum(N[i] * q_e[i] for i in 1:num_element_dofs)

        # Stiffness contribution.
        for i in 1:num_element_dofs
            for j in 1:num_element_dofs
                K_e[i, j] += B[i] * E_e * A_e * B[j] * J * w_g
            end
        end

        # Consistent distributed-load contribution.
        for i in 1:num_element_dofs
            f_e[i] += N[i] * q_g * J * w_g
        end
    end

    return K_e, f_e
end

# Assemble isoparametric axial-bar element matrices and distributed-load vectors.
function assemble_axial_bar_gauss_legendre!(
    K,
    f,
    node_coordinates,
    element_connectivity,
    element_E,
    element_A,
    element_q,
    num_gauss_legendre_points,
    interpolation,
)
    num_elements = size(element_connectivity, 1)

    for e in 1:num_elements
        element_nodes = element_connectivity[e, :]
        x_e = node_coordinates[element_nodes]
        E_e = element_E[e]
        A_e = element_A[e]
        num_element_dofs = length(element_nodes)
        q_e = fill(element_q[e], num_element_dofs)

        K_e, f_e = integrate_axial_bar_gauss_legendre(
            x_e,
            E_e,
            A_e,
            q_e,
            num_gauss_legendre_points,
            interpolation,
        )

        # For one displacement DOF per node, the connected node numbers
        # are also the corresponding global DOF numbers.
        dof_map = element_nodes

        assemble_matrix!(K, K_e, dof_map)
        assemble_vector!(f, f_e, dof_map)
    end
end

# Solve an isoparametric axial-bar analysis using Gauss--Legendre element integration.
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
    interpolation,
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
        interpolation,
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

# Recover isoparametric axial-bar responses at selected reference coordinates.
function recover_axial_bar_response(
    node_coordinates,
    element_connectivity,
    element_E,
    element_A,
    u,
    response_reference_coordinates,
    interpolation,
)
    num_elements = size(element_connectivity, 1)
    num_response_points = length(response_reference_coordinates)

    response_coordinates = zeros(num_elements, num_response_points)
    strain = zeros(num_elements, num_response_points)
    stress = zeros(num_elements, num_response_points)
    axial_force = zeros(num_elements, num_response_points)

    for e in 1:num_elements
        element_nodes = element_connectivity[e, :]
        x_e = node_coordinates[element_nodes]
        E_e = element_E[e]
        A_e = element_A[e]
        u_e = u[element_nodes]
        num_element_dofs = length(element_nodes)

        for r in 1:num_response_points
            xi_r = response_reference_coordinates[r]
            N, dN_dxi = bar_shape_functions(xi_r, interpolation)
            x = sum(N[i] * x_e[i] for i in 1:num_element_dofs)
            J = sum(dN_dxi[i] * x_e[i] for i in 1:num_element_dofs)
            B = dN_dxi / J
            strain_r = sum(B[i] * u_e[i] for i in 1:num_element_dofs)
            stress_r = E_e * strain_r
            axial_force_r = A_e * stress_r

            response_coordinates[e, r] = x
            strain[e, r] = strain_r
            stress[e, r] = stress_r
            axial_force[e, r] = axial_force_r
        end
    end

    return response_coordinates, strain, stress, axial_force
end
