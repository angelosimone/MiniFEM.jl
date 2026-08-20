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

include("fem_assembly.jl")
include("two_springs_fem.jl")

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
