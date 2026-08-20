# Executable benchmark for a uniform one-dimensional axial bar.

# Physical layout: four equal bar elements with geometric node numbering.
#
#   u_1 = 0                               node 3: P →                         free right end
#     |---- element 1 ----|---- element 2 ----|---- element 3 ----|---- element 4 ----|
#   node 1              node 2              node 3              node 4              node 5

include("fem_assembly.jl")
include("axial_bar_fem.jl")

# -------------------------------
# 1. Model data
# -------------------------------

L = 1000.0                 # Bar length [mm]
E = 100000.0               # Young's modulus [N/mm^2]
A = 100.0                  # Cross-sectional area [mm^2]
q = 10.0                   # Distributed axial load [N/mm]
P = 10000.0                # Concentrated axial force at x = L / 2 [N]

# Geometric node numbering follows the left-to-right coordinate order.
node_coordinates = [0.0, L / 4, L / 2, 3 * L / 4, L]

# Each row maps the two local element nodes to global nodes.
element_connectivity = [
    1 2;
    2 3;
    3 4;
    4 5
]

# Uniform material, cross-sectional, and distributed-load data by element.
element_E = [E, E, E, E]
element_A = [A, A, A, A]
element_q = [q, q, q, q]

# The concentrated force is already a nodal load at global DOF 3.
applied_forces = [(3, P)]

constrained_dofs = [1]
prescribed_displacements = [0.0]

# -------------------------------
# 2. Finite element solution
# -------------------------------

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

# -------------------------------
# 3. Element post-processing
# -------------------------------

(
    element_displacement_increments,
    element_strain,
    element_stress,
    element_axial_force,
) = recover_axial_bar_response(
    node_coordinates,
    element_connectivity,
    element_E,
    element_A,
    u,
)

# -------------------------------
# 4. Independent analytical verification
# -------------------------------

a = L / 2
EA = E * A
num_nodes = length(node_coordinates)
num_elements = size(element_connectivity, 1)
u_exact = zeros(num_nodes)
element_midpoints = zeros(num_elements)
axial_force_exact_at_midpoints = zeros(num_elements)

for node in 1:num_nodes
    x = node_coordinates[node]

    if x <= a
        u_exact[node] = (q * (L * x - x^2 / 2) + P * x) / EA
    else
        u_exact[node] = (q * (L * x - x^2 / 2) + P * a) / EA
    end
end

for e in 1:num_elements
    element_nodes = element_connectivity[e, :]
    x_e = node_coordinates[element_nodes]
    x = (x_e[1] + x_e[2]) / 2
    element_midpoints[e] = x

    if x < a
        axial_force_exact_at_midpoints[e] = q * (L - x) + P
    else
        axial_force_exact_at_midpoints[e] = q * (L - x)
    end
end

reaction_expected = -(q * L + P)
global_equilibrium_residual = reactions[1] + q * L + P
exact_force_jump = q * (L - a) - (q * (L - a) + P)

# -------------------------------
# 5. Results
# -------------------------------

println("\n- Nodal displacements u [mm]:")
display(u)

println("\n- Analytical nodal displacements [mm]:")
display(u_exact)

println("\n- Nodal displacement error [mm]:")
display(u - u_exact)

println("\n- Reactions at prescribed DOFs [N]:")
display(reactions)

println("\n- Expected reaction at x = 0 [N]: ", reaction_expected)
println("- Global equilibrium residual R + qL + P [N]: ", global_equilibrium_residual)

println("\n- Element displacement increments [mm]:")
display(element_displacement_increments)

println("\n- Element strains:")
display(element_strain)

println("\n- Element stresses [N/mm^2]:")
display(element_stress)

println("\n- Recovered element axial forces [N]:")
display(element_axial_force)

println("\n- Exact axial forces at element midpoints [N]:")
display(axial_force_exact_at_midpoints)

println("\n- Element axial-force error at midpoints [N]:")
display(element_axial_force - axial_force_exact_at_midpoints)

println("\n- Exact axial-force jump at x = L / 2 [N]: ", exact_force_jump)
