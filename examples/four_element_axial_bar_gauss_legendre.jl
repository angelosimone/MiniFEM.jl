# Executable Gauss--Legendre benchmark for a uniform one-dimensional axial bar.

# Physical layout: four equal bar elements with geometric node numbering.
#
#   u_1 = 0                               node 3: P →                         free right end
#     |---- element 1 ----|---- element 2 ----|---- element 3 ----|---- element 4 ----|
#   node 1              node 2              node 3              node 4              node 5

include("fem_assembly.jl")
include("gauss_legendre_quadrature.jl")
include("axial_bar_gauss_legendre_fem.jl")

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

# One point is exact here for constant E, A, and the uniform element load.
num_gauss_legendre_points = 1

# -------------------------------
# 2. Finite element solution
# -------------------------------

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

# -------------------------------
# 3. Element post-processing
# -------------------------------

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
    num_gauss_legendre_points,
)

# -------------------------------
# 4. Independent analytical verification
# -------------------------------

a = L / 2
EA = E * A
num_nodes = length(node_coordinates)
num_elements = size(element_connectivity, 1)
u_exact = zeros(num_nodes)
num_response_points = size(gauss_point_coordinates, 2)
axial_force_exact_at_gauss_points = zeros(num_elements, num_response_points)

for node in 1:num_nodes
    x = node_coordinates[node]

    if x <= a
        u_exact[node] = (q * (L * x - x^2 / 2) + P * x) / EA
    else
        u_exact[node] = (q * (L * x - x^2 / 2) + P * a) / EA
    end
end

for e in 1:num_elements
    for g in 1:num_response_points
        x_g = gauss_point_coordinates[e, g]

        if x_g < a
            axial_force_exact_at_gauss_points[e, g] = q * (L - x_g) + P
        else
            axial_force_exact_at_gauss_points[e, g] = q * (L - x_g)
        end
    end
end

reaction_expected = -(q * L + P)
global_equilibrium_residual = reactions[1] + q * L + P
exact_force_jump = q * (L - a) - (q * (L - a) + P)

# -------------------------------
# 5. Results
# -------------------------------

println("\n- Gauss--Legendre points per element: ", num_gauss_legendre_points)

println("- Nodal displacements u [mm]:")
display(u)

println("\n- Analytical nodal displacements [mm]:")
display(u_exact)

println("\n- Nodal displacement error [mm]:")
display(u - u_exact)

println("\n- Reactions at prescribed DOFs [N]:")
display(reactions)

println("\n- Expected reaction at x = 0 [N]: ", reaction_expected)
println("- Global equilibrium residual R + qL + P [N]: ", global_equilibrium_residual)

println("\n- Gauss-point reference coordinates xi:")
display(gauss_point_reference_coordinates)

println("\n- Gauss-point physical coordinates x [mm]:")
display(gauss_point_coordinates)

println("\n- Gauss-point strains:")
display(gauss_point_strain)

println("\n- Gauss-point stresses [N/mm^2]:")
display(gauss_point_stress)

println("\n- Recovered Gauss-point axial forces [N]:")
display(gauss_point_axial_force)

println("\n- Exact axial forces at Gauss points [N]:")
display(axial_force_exact_at_gauss_points)

println("\n- Gauss-point axial-force error [N]:")
display(gauss_point_axial_force - axial_force_exact_at_gauss_points)

println("\n- Exact axial-force jump at x = L / 2 [N]: ", exact_force_jump)
