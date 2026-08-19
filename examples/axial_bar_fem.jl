# Reusable finite element operations for two-node one-dimensional axial-bar analyses.

# Construct the stiffness matrix of a two-node axial-bar element.
function bar_stiffness(E_e, A_e, L_e)
    Ke = E_e * A_e / L_e * [1.0 -1.0; -1.0 1.0]
    return Ke
end

# Construct the consistent nodal load vector for a two-node bar element under a uniform distributed load.
function bar_distributed_load(q_e, L_e)
    fe = q_e * L_e / 2 * [1.0, 1.0]
    return fe
end

# Assemble axial-bar element stiffness matrices and distributed-load vectors.
function assemble_axial_bar!(
    K,
    f,
    node_coordinates,
    element_connectivity,
    element_E,
    element_A,
    element_q,
)
    num_elements = size(element_connectivity, 1)

    for e in 1:num_elements
        element_nodes = element_connectivity[e, :]
        x_e = node_coordinates[element_nodes]

        E_e = element_E[e]
        A_e = element_A[e]
        q_e = element_q[e]

        L_e = x_e[2] - x_e[1]
        K_e = bar_stiffness(E_e, A_e, L_e)
        f_e = bar_distributed_load(q_e, L_e)

        # For one displacement DOF per node, the connected node numbers
        # are also the corresponding global DOF numbers.
        dof_map = [element_connectivity[e, 1], element_connectivity[e, 2]]

        for i in 1:2
            I = dof_map[i]
            f[I] += f_e[i]

            for j in 1:2
                J = dof_map[j]
                K[I, J] += K_e[i, j]
            end
        end
    end
end

# Solve a two-node axial-bar analysis with supplied mesh, load, and constraint data.
function solve_axial_bar(
    node_coordinates,
    element_connectivity,
    element_E,
    element_A,
    element_q,
    applied_forces,
    constrained_dofs,
    prescribed_displacements,
)
    num_nodes = length(node_coordinates)
    dofs_per_node = 1
    total_dofs = num_nodes * dofs_per_node

    K = zeros(total_dofs, total_dofs)
    f = zeros(total_dofs)

    assemble_axial_bar!(
        K,
        f,
        node_coordinates,
        element_connectivity,
        element_E,
        element_A,
        element_q,
    )

    # Add concentrated forces that are already specified at global DOFs.
    for (dof, value) in applied_forces
        f[dof] += value
    end

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

# Recover element responses for two-node axial-bar elements.
function recover_axial_bar_response(
    node_coordinates,
    element_connectivity,
    element_E,
    element_A,
    u,
)
    num_elements = size(element_connectivity, 1)

    element_displacement_increments = zeros(num_elements)
    element_strain = zeros(num_elements)
    element_stress = zeros(num_elements)
    element_axial_force = zeros(num_elements)

    for e in 1:num_elements
        element_nodes = element_connectivity[e, :]
        x_e = node_coordinates[element_nodes]
        E_e = element_E[e]
        A_e = element_A[e]
        u_e = u[element_nodes]

        L_e = x_e[2] - x_e[1]
        displacement_increment = u_e[2] - u_e[1]
        strain = displacement_increment / L_e
        stress = E_e * strain
        axial_force = A_e * stress

        element_displacement_increments[e] = displacement_increment
        element_strain[e] = strain
        element_stress[e] = stress
        element_axial_force[e] = axial_force
    end

    return (
        element_displacement_increments,
        element_strain,
        element_stress,
        element_axial_force,
    )
end
