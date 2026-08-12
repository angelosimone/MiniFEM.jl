# Solve a series combination of two 1D springs with a callable finite element
# analysis.

# Physical layout:
#
#   node 2                  node 3                  node 1
#   u_2 = 0                 internal node           applied force
#
#     /|
#     /|o------ spring 1 -------o------ spring 2 -------o  → F
#     /|
#
# Left-to-right node order: [2, 3, 1].

# -------------------------------
# Finite element functions
# -------------------------------

# Construct the stiffness matrix of a two-node spring element.
function spring_stiffness(k_e)
    K_e = [
        +k_e -k_e;
        -k_e +k_e
    ]
    return K_e
end

# Assemble all spring-element stiffness matrices into the global matrix.
function assemble_stiffness!(K, element_connectivity, element_stiffness)
    num_elements = size(element_connectivity, 1)

    for e in 1:num_elements
        k_e = element_stiffness[e]
        K_e = spring_stiffness(k_e)

        # For one displacement DOF per node, the connected node numbers
        # are also the corresponding global DOF numbers.
        dof_map = [element_connectivity[e, 1], element_connectivity[e, 2]]

        for i in 1:2
            I = dof_map[i]
            for j in 1:2
                J = dof_map[j]
                K[I, J] += K_e[i, j]
            end
        end
    end
end

# Assemble the applied nodal forces into the global force vector.
function assemble_forces!(f, applied_forces)
    for (dof, value) in applied_forces
        f[dof] += value
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

# -------------------------------
# Complete two-spring analysis
# -------------------------------

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

    assemble_stiffness!(K, element_connectivity, element_stiffness)

    # Preserve the physical stiffness matrix before Dirichlet modification.
    K_original = copy(K)

    assemble_forces!(f, applied_forces)
    apply_homogeneous_constraints!(K, f, constrained_dofs)

    u = K \ f
    strain_energy_global = strain_energy(K_original, u)

    return u, strain_energy_global
end

# -------------------------------
# Canonical two-spring benchmark
# -------------------------------

k_1 = 1.0
k_2 = 2.0
applied_force = 10.0

u, strain_energy_global = solve_two_springs(k_1, k_2, applied_force)

println("\n- Nodal displacements u [mm]:")
display(u)

# Display the analytical solution for comparison.
# Global-vector order is [u_1, u_2, u_3], not left-to-right geometric order.
u_analytical = [15.0, 0.0, 10.0]

println("\n- Analytical nodal displacements u [mm]:")
display(u_analytical)

# -------------------------------
# Analytical verification
# -------------------------------

println(
    "\n- Total internal strain energy stored in the springs (global computation) [N mm]: ",
    strain_energy_global,
)

# Compute the analytical strain energy.
# For two springs in series, the total compliance is 1/k_1 + 1/k_2.
strain_energy_analytical = 0.5 * applied_force^2 * (1 / k_1 + 1 / k_2)

println(
    "\n- Total internal strain energy stored in the springs (analytical solution) [N mm]: ",
    strain_energy_analytical,
)

# Compare numerical and analytical energy values.
println(
    "\n- Difference between numerical and analytical energy [N mm]: ",
    abs(strain_energy_global - strain_energy_analytical),
)
