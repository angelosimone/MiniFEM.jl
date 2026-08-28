# Explicit scalar examples of Gauss--Legendre quadrature.

include("gauss_legendre_quadrature.jl")

function evaluate_scalar_gauss_legendre_examples()
    # A one-point rule integrates a general linear polynomial exactly.
    linear_one_point = 0.0
    for (xi_g, w_g) in gauss_legendre_rule(1)
        integrand_value = 2.0 + 3.0 * xi_g
        weighted_contribution = w_g * integrand_value
        linear_one_point += weighted_contribution
    end
    linear_exact = 4.0

    # Compare one- and two-point rules for a general cubic polynomial.
    cubic_one_point = 0.0
    for (xi_g, w_g) in gauss_legendre_rule(1)
        integrand_value = 1.0 - 2.0 * xi_g + 3.0 * xi_g^2 + 4.0 * xi_g^3
        weighted_contribution = w_g * integrand_value
        cubic_one_point += weighted_contribution
    end

    cubic_two_point = 0.0
    for (xi_g, w_g) in gauss_legendre_rule(2)
        integrand_value = 1.0 - 2.0 * xi_g + 3.0 * xi_g^2 + 4.0 * xi_g^3
        weighted_contribution = w_g * integrand_value
        cubic_two_point += weighted_contribution
    end
    cubic_exact = 4.0

    # Map the standard interval to [0, 3] and integrate x^2 + 1.
    a = 0.0
    b = 3.0
    J = (b - a) / 2
    mapped_two_point = 0.0
    for (xi_g, w_g) in gauss_legendre_rule(2)
        x_g = (a + b) / 2 + J * xi_g
        integrand_value = x_g^2 + 1.0
        weighted_contribution = w_g * integrand_value * J
        mapped_two_point += weighted_contribution
    end
    mapped_exact = 12.0

    println("- Linear polynomial: one-point result = ", linear_one_point)
    println("  Exact result = ", linear_exact)
    println("- Cubic polynomial: one-point result = ", cubic_one_point)
    println("  Cubic polynomial: two-point result = ", cubic_two_point)
    println("  Exact result = ", cubic_exact)
    println("- Integral of x^2 + 1 on [0, 3]: two-point result = ", mapped_two_point)
    println("  Exact result = ", mapped_exact)

    return (
        linear_one_point,
        cubic_one_point,
        cubic_two_point,
        mapped_two_point,
    )
end

evaluate_scalar_gauss_legendre_examples()
