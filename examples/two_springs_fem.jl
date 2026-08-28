# Reusable numerical definitions for the two-spring finite element analysis.

# Construct the stiffness matrix of a two-node spring element.
function spring_stiffness(k_e)
    K_e = [
        +k_e -k_e;
        -k_e +k_e
    ]
    return K_e
end

# Assemble the stiffness matrices of all spring elements into the global matrix.
function assemble_springs!(K, element_connectivity, element_stiffness)
    num_elements = size(element_connectivity, 1)

    for e in 1:num_elements
        k_e = element_stiffness[e]
        K_e = spring_stiffness(k_e)

        # For one displacement DOF per node, the connected node numbers
        # are also the corresponding global DOF numbers.
        dof_map = [element_connectivity[e, 1], element_connectivity[e, 2]]

        assemble_matrix!(K, K_e, dof_map)
    end
end

# Impose homogeneous prescribed displacements by row-and-column modification.
function apply_homogeneous_constraints!(K, f, constrained_dofs)
    total_dofs = size(K, 1)

    for dof in constrained_dofs
        for i in 1:total_dofs
            K[i, dof] = 0.0
            K[dof, i] = 0.0
        end

        K[dof, dof] = 1.0
        f[dof] = 0.0
    end
end

# Compute the total internal strain energy.
function strain_energy(K, u)
    return 0.5 * transpose(u) * K * u
end

# Complete two-spring analysis.
function solve_two_springs(k_1, k_2, applied_force)
    # Fixed topology: physical nodes are labeled [2, 3, 1] from left to right.
    num_nodes = 3
    dofs_per_node = 1

    element_connectivity = [
        2 3;  # spring between physical nodes 2 and 3
        3 1   # spring between physical nodes 3 and 1
    ]

    constrained_dofs = [2]
    element_stiffness = [k_1, k_2]
    applied_forces = [(1, applied_force)]

    total_dofs = num_nodes * dofs_per_node
    K = zeros(total_dofs, total_dofs)
    f = zeros(total_dofs)

    assemble_springs!(K, element_connectivity, element_stiffness)

    # Keep a copy of the assembled stiffness matrix for the
    # strain-energy calculation.
    K_original = copy(K)

    add_nodal_forces!(f, applied_forces)
    apply_homogeneous_constraints!(K, f, constrained_dofs)

    u = K \ f
    strain_energy_global = strain_energy(K_original, u)

    return u, strain_energy_global
end
