# Solve a series combination of two 1D springs using basic Julia constructs.

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
# 1. Pre-processing: Input Data
# -------------------------------

# Number of nodes and displacement DOFs per node
num_nodes = 3  # physical nodes: labeled [2, 3, 1] from left to right
dofs_per_node = 1  # one displacement DOF per node

# Connectivity matrix defining which nodes belong to each element
element_connectivity = [
    2 3;  # spring between physical nodes 2 and 3
    3 1   # spring between physical nodes 3 and 1
]

# Spring constants for each element in N/mm (k_1 and k_2)
element_stiffness = [1.0, 2.0]

# Boundary conditions

# Prescribed displacement (Dirichlet boundary condition)
# - Constrain physical node 2 (DOF 2) to zero displacement:
constrained_dofs = [2]

# Applied force (Neumann boundary condition)
# - Apply a force of intensity 10 N at physical node 1 (DOF 1):
applied_force = 10.0
applied_forces = [(1, applied_force)]

# Derived quantities
# - Number of degrees of freedom
total_dofs = num_nodes * dofs_per_node
# - Number of elements
num_elements = size(element_connectivity, 1)

# Initialize global K matrix and force vector.
K = zeros(total_dofs, total_dofs)
f = zeros(total_dofs)

# -------------------------------
# 2. Processing
# 2.1 Assembly: Build Global K
# -------------------------------

println("\n- Assembling local stiffness matrices into global K")
for e in 1:num_elements
    # Element stiffness matrix: K_e = k_e * [1 -1; -1 1]
    k_e = element_stiffness[e]
    K_e = [
        +k_e -k_e;
        -k_e +k_e
    ]

    println("\n-- Element ", e, ", k_e = ", k_e)
    println("   Element stiffness matrix K_e:")
    display(K_e)

    # Map local DOFs to global DOFs for an element:
    # the local to global DOF mapping is built from element nodes
    # through the element connectivity matrix (for elements with one DOF
    # per node, the global DOF is the same as the node number).
    dof_map = [element_connectivity[e, 1], element_connectivity[e, 2]]

    # Assemble the local stiffness matrix into the global stiffness matrix.
    for i in 1:2
        I = dof_map[i]
        for j in 1:2
            J = dof_map[j]
            K[I, J] += K_e[i, j]
        end
    end
end

println("\n- Global stiffness matrix K:")
display(K)

# -------------------------------
# 2.2 Boundary Conditions
# -------------------------------

# Save the original K for the energy computation later.
K_original = copy(K)

# Apply Neumann BCs (forces).
for (dof, value) in applied_forces
    f[dof] += value
end

println("\n- Global force vector f:")
display(f)

# Apply homogeneous Dirichlet BCs (prescribed displacements).
for dof in constrained_dofs
    for i in 1:total_dofs
        K[i, dof] = 0.0
        K[dof, i] = 0.0
    end

    K[dof, dof] = 1.0
    f[dof] = 0.0
end

println("\n- Global stiffness matrix K after BCs:")
display(K)
println("\n- Global force vector f after BCs:")
display(f)

# -------------------------------
# 2.3 Solver
# -------------------------------

# Solve the linear system K * u = f.
# The backslash operator solves the system without explicitly computing inv(K).
u = K \ f

println("\n- Nodal displacements u [mm]:")
display(u)

# Display the analytical solution for comparison.
# Global-vector order is [u_1, u_2, u_3], not left-to-right geometric order.
u_analytical = [15.0, 0.0, 10.0]

println("\n- Analytical nodal displacements u [mm]:")
display(u_analytical)

# -------------------------------
# 3. Post-processing: Strain Energy
# -------------------------------

# Compute the total internal strain energy stored in the springs using
# the stiffness matrix before the Dirichlet modifications:
# 1/2 * u^T * K_original * u
strain_energy_global = 0.5 * transpose(u) * K_original * u

println(
    "\n- Total internal strain energy stored in the springs (global computation) [N mm]: ",
    strain_energy_global,
)

# Compute the analytical strain energy.
# For two springs in series, the total compliance is 1/k_1 + 1/k_2.
k_1 = element_stiffness[1]
k_2 = element_stiffness[2]
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
