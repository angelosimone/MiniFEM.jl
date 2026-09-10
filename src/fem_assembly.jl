# Shared finite element operations for assembling local contributions and adding global nodal loads.

# Scatter-add a local matrix into a global matrix using a DOF map.
function assemble_matrix!(K, K_e, dof_map)
    for i in axes(K_e, 1)
        I = dof_map[i]

        for j in axes(K_e, 2)
            J = dof_map[j]
            K[I, J] += K_e[i, j]
        end
    end
end

# Scatter-add a local vector into a global vector using a DOF map.
function assemble_vector!(f, f_e, dof_map)
    for i in axes(f_e, 1)
        I = dof_map[i]
        f[I] += f_e[i]
    end
end

# Add forces that are already specified at global DOFs.
function add_nodal_forces!(f, applied_forces)
    for (dof, value) in applied_forces
        f[dof] += value
    end
end
