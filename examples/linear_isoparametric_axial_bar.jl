# Executable reference benchmark using linear isoparametric axial-bar elements.

# Physical layout: four equal bar elements with geometric node numbering.
#
#   u_1 = 0                               node 3: P →                         free right end
#     |---- element 1 ----|---- element 2 ----|---- element 3 ----|---- element 4 ----|
#   node 1              node 2              node 3              node 4              node 5

include("../src/fem_assembly.jl")
include("../src/gauss_legendre_quadrature.jl")
include("../src/isoparametric_axial_bar.jl")

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
interpolation = "linear"

# Response locations are selected independently of the quadrature rule.
response_reference_coordinates = [0.0]

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
    interpolation,
)

# -------------------------------
# 3. Element post-processing
# -------------------------------

(
response_coordinates,
response_strain,
response_stress,
response_axial_force
) = recover_axial_bar_response(
    node_coordinates,
    element_connectivity,
    element_E,
    element_A,
    u,
    response_reference_coordinates,
    interpolation,
)

# -------------------------------
# 4. Independent analytical verification
# -------------------------------

a = L / 2
EA = E * A
num_nodes = length(node_coordinates)
num_elements = size(element_connectivity, 1)
u_exact = zeros(num_nodes)
num_response_points = length(response_reference_coordinates)
axial_force_exact_at_response_points = zeros(num_elements, num_response_points)

for node in 1:num_nodes
    x = node_coordinates[node]

    if x <= a
        u_exact[node] = (q * (L * x - x^2 / 2) + P * x) / EA
    else
        u_exact[node] = (q * (L * x - x^2 / 2) + P * a) / EA
    end
end

for e in 1:num_elements
    for r in 1:num_response_points
        x_r = response_coordinates[e, r]

        if x_r < a
            axial_force_exact_at_response_points[e, r] = q * (L - x_r) + P
        else
            axial_force_exact_at_response_points[e, r] = q * (L - x_r)
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

println("\n- Response reference coordinates xi:")
display(response_reference_coordinates)

println("\n- Response physical coordinates x [mm]:")
display(response_coordinates)

println("\n- Response strains:")
display(response_strain)

println("\n- Response stresses [N/mm^2]:")
display(response_stress)

println("\n- Recovered response axial forces [N]:")
display(response_axial_force)

println("\n- Exact axial forces at response points [N]:")
display(axial_force_exact_at_response_points)

println("\n- Response axial-force error [N]:")
display(response_axial_force - axial_force_exact_at_response_points)

println("\n- Exact axial-force jump at x = L / 2 [N]: ", exact_force_jump)
