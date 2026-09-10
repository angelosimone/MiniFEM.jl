# Executable reference cases using quadratic isoparametric axial-bar elements.

include("../src/fem_assembly.jl")
include("../src/gauss_legendre_quadrature.jl")
include("../src/isoparametric_axial_bar.jl")

# -------------------------------
# 1. Common model data
# -------------------------------

L = 1000.0                 # Bar length [mm]
E = 100000.0               # Young's modulus [N/mm^2]
A = 100.0                  # Cross-sectional area [mm^2]
q = 10.0                   # Distributed axial load [N/mm]
P = 10000.0                # Concentrated axial force at x = L / 2 [N]
EA = E * A

# Two points are exact for the regular quadratic-element stiffness and uniform load.
num_gauss_legendre_points = 2
interpolation = "quadratic"
response_reference_coordinates = [-1.0, 0.0, 1.0]

# -------------------------------
# 2. One element with an end force
# -------------------------------

F = 10000.0
node_coordinates_end_force = [0.0, L / 2, L]
element_connectivity_end_force = [1 2 3]
element_E_end_force = [E]
element_A_end_force = [A]
element_q_end_force = [0.0]
applied_forces_end_force = [(3, F)]
constrained_dofs_end_force = [1]
prescribed_displacements_end_force = [0.0]

u_end_force, reactions_end_force = solve_axial_bar_gauss_legendre(
    node_coordinates_end_force,
    element_connectivity_end_force,
    element_E_end_force,
    element_A_end_force,
    element_q_end_force,
    applied_forces_end_force,
    constrained_dofs_end_force,
    prescribed_displacements_end_force,
    num_gauss_legendre_points,
    interpolation,
)

(
response_coordinates_end_force,
response_strain_end_force,
response_stress_end_force,
response_axial_force_end_force
) = recover_axial_bar_response(
    node_coordinates_end_force,
    element_connectivity_end_force,
    element_E_end_force,
    element_A_end_force,
    u_end_force,
    response_reference_coordinates,
    interpolation,
)

u_end_force_exact = [0.0, F * L / (2 * EA), F * L / EA]
strain_end_force_exact = fill(F / EA, size(response_strain_end_force))
stress_end_force_exact = fill(F / A, size(response_stress_end_force))
axial_force_end_force_exact = fill(F, size(response_axial_force_end_force))
reaction_end_force_expected = -F

# -------------------------------
# 3. One element with a uniform distributed load
# -------------------------------

node_coordinates_uniform_load = [0.0, L / 2, L]
element_connectivity_uniform_load = [1 2 3]
element_E_uniform_load = [E]
element_A_uniform_load = [A]
element_q_uniform_load = [q]
applied_forces_uniform_load = Tuple{Int, Float64}[]
constrained_dofs_uniform_load = [1]
prescribed_displacements_uniform_load = [0.0]

u_uniform_load, reactions_uniform_load = solve_axial_bar_gauss_legendre(
    node_coordinates_uniform_load,
    element_connectivity_uniform_load,
    element_E_uniform_load,
    element_A_uniform_load,
    element_q_uniform_load,
    applied_forces_uniform_load,
    constrained_dofs_uniform_load,
    prescribed_displacements_uniform_load,
    num_gauss_legendre_points,
    interpolation,
)

(
response_coordinates_uniform_load,
response_strain_uniform_load,
response_stress_uniform_load,
response_axial_force_uniform_load
) = recover_axial_bar_response(
    node_coordinates_uniform_load,
    element_connectivity_uniform_load,
    element_E_uniform_load,
    element_A_uniform_load,
    u_uniform_load,
    response_reference_coordinates,
    interpolation,
)

u_uniform_load_exact = [0.0, 3 * q * L^2 / (8 * EA), q * L^2 / (2 * EA)]
strain_uniform_load_exact = q / EA .* (L .- response_coordinates_uniform_load)
stress_uniform_load_exact = E .* strain_uniform_load_exact
axial_force_uniform_load_exact = A .* stress_uniform_load_exact
reaction_uniform_load_expected = -q * L

# -------------------------------
# 4. Two-element reference benchmark
# -------------------------------

# The five global nodes have the same physical positions as the four-linear-element benchmark.
node_coordinates = [0.0, L / 4, L / 2, 3 * L / 4, L]

# Each row maps the left endpoint, midpoint, and right endpoint of one quadratic element.
element_connectivity = [
    1 2 3;
    3 4 5
]

element_E = [E, E]
element_A = [A, A]
element_q = [q, q]
applied_forces = [(3, P)]
constrained_dofs = [1]
prescribed_displacements = [0.0]

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
# 5. Independent analytical verification
# -------------------------------

a = L / 2
num_nodes = length(node_coordinates)
num_elements = size(element_connectivity, 1)
num_response_points = length(response_reference_coordinates)
u_exact = zeros(num_nodes)
strain_exact = zeros(num_elements, num_response_points)
stress_exact = zeros(num_elements, num_response_points)
axial_force_exact = zeros(num_elements, num_response_points)

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

        if e == 1
            axial_force_exact[e, r] = q * (L - x_r) + P
        else
            axial_force_exact[e, r] = q * (L - x_r)
        end

        strain_exact[e, r] = axial_force_exact[e, r] / EA
        stress_exact[e, r] = E * strain_exact[e, r]
    end
end

reaction_expected = -(q * L + P)
global_equilibrium_residual = reactions[1] + q * L + P
exact_force_jump = q * (L - a) - (q * (L - a) + P)

# -------------------------------
# 6. Results
# -------------------------------

println("\n- Gauss--Legendre points per element: ", num_gauss_legendre_points)

println("\n- One-element end-force displacements [mm]:")
display(u_end_force)
println("- Analytical displacements [mm]:")
display(u_end_force_exact)
println("- Recovered strains:")
display(response_strain_end_force)
println("- Exact strains:")
display(strain_end_force_exact)
println("- Strain error:")
display(response_strain_end_force - strain_end_force_exact)
println("- Recovered stresses [N/mm^2]:")
display(response_stress_end_force)
println("- Exact stresses [N/mm^2]:")
display(stress_end_force_exact)
println("- Stress error [N/mm^2]:")
display(response_stress_end_force - stress_end_force_exact)
println("- Recovered axial forces [N]:")
display(response_axial_force_end_force)
println("- Exact axial forces [N]:")
display(axial_force_end_force_exact)
println("- Axial-force error [N]:")
display(response_axial_force_end_force - axial_force_end_force_exact)
println("- Expected support reaction [N]: ", reaction_end_force_expected)
println("- Computed support reaction [N]: ", reactions_end_force[1])

println("\n- One-element uniform-load displacements [mm]:")
display(u_uniform_load)
println("- Analytical displacements [mm]:")
display(u_uniform_load_exact)
println("- Response coordinates x [mm]:")
display(response_coordinates_uniform_load)
println("- Recovered strains:")
display(response_strain_uniform_load)
println("- Exact strains:")
display(strain_uniform_load_exact)
println("- Strain error:")
display(response_strain_uniform_load - strain_uniform_load_exact)
println("- Recovered stresses [N/mm^2]:")
display(response_stress_uniform_load)
println("- Exact stresses [N/mm^2]:")
display(stress_uniform_load_exact)
println("- Stress error [N/mm^2]:")
display(response_stress_uniform_load - stress_uniform_load_exact)
println("- Recovered axial forces [N]:")
display(response_axial_force_uniform_load)
println("- Exact axial forces [N]:")
display(axial_force_uniform_load_exact)
println("- Axial-force error [N]:")
display(response_axial_force_uniform_load - axial_force_uniform_load_exact)
println("- Expected support reaction [N]: ", reaction_uniform_load_expected)
println("- Computed support reaction [N]: ", reactions_uniform_load[1])

println("\n- Two-element reference-benchmark nodal displacements u [mm]:")
display(u)
println("- Analytical nodal displacements [mm]:")
display(u_exact)
println("- Nodal displacement error [mm]:")
display(u - u_exact)
println("- Computed support reaction [N]: ", reactions[1])
println("- Expected support reaction [N]: ", reaction_expected)
println("- Global equilibrium residual R + qL + P [N]: ", global_equilibrium_residual)
println("- Response coordinates x [mm]:")
display(response_coordinates)
println("- Recovered strains:")
display(response_strain)
println("- Exact strains:")
display(strain_exact)
println("- Recovered stresses [N/mm^2]:")
display(response_stress)
println("- Exact stresses [N/mm^2]:")
display(stress_exact)
println("- Recovered axial forces [N]:")
display(response_axial_force)
println("- Exact branchwise axial forces [N]:")
display(axial_force_exact)
println("- Axial-force error [N]:")
display(response_axial_force - axial_force_exact)
println("- Exact axial-force jump at x = L / 2 [N]: ", exact_force_jump)
