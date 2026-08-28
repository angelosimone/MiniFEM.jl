# Gauss--Legendre point-and-weight data on the standard interval [-1, 1].

function gauss_legendre_rule(num_points)
    if num_points == 1
        return [
            (0.0, 2.0)
        ]
    elseif num_points == 2
        return [
            (-1 / sqrt(3), 1.0),
            (+1 / sqrt(3), 1.0),
        ]
    else
        error("Gauss--Legendre rules are available only for one or two points.")
    end
end
