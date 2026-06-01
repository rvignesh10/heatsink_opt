function tools = eval_legendre()
% Returns a struct of function handles that can be used by other scripts.

    % Return a struct where each field is a handle to a local function
    tools.legendreP = @legendreP;
    tools.legendrePDeriv = @legendrePDeriv;
    tools.eval_legendre_with_derivs = @eval_legendre_with_derivs;
    tools.eval_legendre_surface_with_derivs = @eval_legendre_surface_with_derivs;
    tools.eval_legendre_surface_with_derivs_alt = @eval_legendre_surface_with_derivs_alt;
end


function L = legendreP(N, x)
    % Computes Legendre polynomials P_n(x) up to degree N.
    % Returns a matrix L of size [length(x), N+1], where L(:, n+1) = P_n(x).
    x = x(:); % Ensure x is a column vector
    L = zeros(length(x), N + 1);
    
    L(:, 1) = 1; % P_0(x) = 1
    if N > 0
        L(:, 2) = x; % P_1(x) = x
    end
    
    for n = 2:N
        % Recurrence relation: (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
        L(:, n + 1) = ((2*n + 1) .* x .* L(:, n) - n .* L(:, n-1)) / (n + 1);
    end
end

function L_deriv = legendrePDeriv(N, x)
    % Computes the derivatives of Legendre polynomials P'_n(x) up to degree N.
    % Uses the recurrence relation: P'_{n+1}(z) = (2n+1)P_n(z) + P'_{n-1}(z)
    x = x(:); % Ensure x is a column vector
    L_deriv = zeros(length(x), N + 1);
    
    % P'_0(x) = 0
    if N >= 0
        L_deriv(:, 1) = 0;
    end
    
    % P'_1(x) = 1
    if N >= 1
        L_deriv(:, 2) = 1;
    end
    
    % We need the polynomials themselves for the recurrence relation
    L = legendreP(N, x);
    
    for n = 2:N
        % Recurrence relation: P'_{n+1}(z) = (2n + 1)P_n(z) + P'_{n-1}(z)
        % L_deriv(:, n + 2) = (2*n + 1) .* (L(:, n + 1) + x.*L_deriv(:, n+1)) + L_deriv(:, n);
        L_deriv(:, n+1) = ( (2*n+1).*(L(:, n) + x.*L_deriv(:, n)) - n.*L_deriv(:, n-1) ) / (n+1);
    end
end

function [result, d_ds] = eval_legendre_with_derivs(s_new, coeffs, s_degree, x_min, x_range)
    % Transform and scale new input points
    x_new = log(s_new);
    x_new_scaled = 2 * (x_new - x_min) / x_range - 1;

    % Evaluate Legendre polynomials and their derivatives for the new points
    L_x_new = legendreP(s_degree, x_new_scaled(:));
    L_x_deriv_new = legendrePDeriv(s_degree, x_new_scaled(:));

    % Initialize outputs
    result = L_x_new * coeffs;
    dT_dx_s = L_x_deriv_new * coeffs;

    % --- Chain Rule Application ---
    dT_dx = dT_dx_s * (2 / x_range); % where x = log(s)
    d_ds = dT_dx .* (1./s_new);
end

function [result, d_dh, d_dp] = eval_legendre_surface_with_derivs(h_new, p_new, coeffs, coeff_indices, h_degree, p_degree, x_min, x_range, y_min, y_range)
    % Evaluates the surface and its derivatives with respect to h and p.
    
    % Transform and scale new input points
    x_new = log(h_new);
    y_new = log(p_new);
    x_new_scaled = 2 * (x_new - x_min) / x_range - 1;
    y_new_scaled = 2 * (y_new - y_min) / y_range - 1;
    
    % Evaluate Legendre polynomials and their derivatives for the new points
    L_x_new = legendreP(h_degree, x_new_scaled(:));
    L_y_new = legendreP(p_degree, y_new_scaled(:));
    L_x_deriv_new = legendrePDeriv(h_degree, x_new_scaled(:));
    L_y_deriv_new = legendrePDeriv(p_degree, y_new_scaled(:));
    
    % Initialize outputs
    result = zeros(size(h_new), 'like', h_new);
    dT_dx_s = zeros(size(h_new), 'like', h_new); % Derivative w.r.t scaled x
    dT_dy_s = zeros(size(h_new), 'like', h_new); % Derivative w.r.t scaled y
    
    % Sum the contributions from each basis function
    for k = 1:length(coeffs)
        i = coeff_indices(k, 1);
        j = coeff_indices(k, 2);
        
        % Value
        basis_eval = L_x_new(:, i+1) .* L_y_new(:, j+1);
        result = result + coeffs(k) * reshape(basis_eval, size(h_new));
        
        % Derivative w.r.t. scaled x (x_s)
        basis_deriv_x = L_x_deriv_new(:, i+1) .* L_y_new(:, j+1);
        dT_dx_s = dT_dx_s + coeffs(k) * reshape(basis_deriv_x, size(h_new));

        % Derivative w.r.t. scaled y (y_s)
        basis_deriv_y = L_x_new(:, i+1) .* L_y_deriv_new(:, j+1);
        dT_dy_s = dT_dy_s + coeffs(k) * reshape(basis_deriv_y, size(h_new));
    end

    % --- Chain Rule Application ---
    % 1. Convert from scaled derivatives to unscaled derivatives (dT/d(log(h)), dT/d(log(p)))
    dT_dx = dT_dx_s * (2 / x_range); % where x = log(h)
    dT_dy = dT_dy_s * (2 / y_range); % where y = log(p)

    % 2. Convert from (log(h), log(p)) derivatives to (h, p) derivatives
    % d/dh = d/d(log(h)) * d(log(h))/dh = d/dx * (1./h)
    % d/dp = d/d(log(p)) * d(log(p))/dp = d/dy * (1./p)
    d_dh = dT_dx .* (1./h_new);
    d_dp = dT_dy .* (1./p_new);
end

function [result, d_dh, d_dp] = eval_legendre_surface_with_derivs_alt(h_new, p_new, coeffs, coeff_indices, h_degree, p_degree, x_min, x_range, y_min, y_range)
    % Evaluates the surface and its derivatives with respect to h and p.

    % Transform and scale new input points
    x_new = h_new;
    y_new = log(p_new);
    x_new_scaled = 2 * (x_new - x_min) / x_range - 1;
    y_new_scaled = 2 * (y_new - y_min) / y_range - 1;
    
    % Evaluate Legendre polynomials and their derivatives for the new points
    L_x_new = legendreP(h_degree, x_new_scaled(:));
    L_y_new = legendreP(p_degree, y_new_scaled(:));
    L_x_deriv_new = legendrePDeriv(h_degree, x_new_scaled(:));
    L_y_deriv_new = legendrePDeriv(p_degree, y_new_scaled(:));
    
    % Initialize outputs
    result = zeros(size(h_new), 'like', h_new);
    dT_dx_s = zeros(size(h_new), 'like', h_new); % Derivative w.r.t scaled x
    dT_dy_s = zeros(size(h_new), 'like', h_new); % Derivative w.r.t scaled y
    
    % Sum the contributions from each basis function
    for k = 1:length(coeffs)
        i = coeff_indices(k, 1);
        j = coeff_indices(k, 2);
        
        % Value
        basis_eval = L_x_new(:, i+1) .* L_y_new(:, j+1);
        result = result + coeffs(k) * reshape(basis_eval, size(h_new));
        
        % Derivative w.r.t. scaled x (x_s)
        basis_deriv_x = L_x_deriv_new(:, i+1) .* L_y_new(:, j+1);
        dT_dx_s = dT_dx_s + coeffs(k) * reshape(basis_deriv_x, size(h_new));

        % Derivative w.r.t. scaled y (y_s)
        basis_deriv_y = L_x_new(:, i+1) .* L_y_deriv_new(:, j+1);
        dT_dy_s = dT_dy_s + coeffs(k) * reshape(basis_deriv_y, size(h_new));
    end

    % --- Chain Rule Application ---
    % 1. Convert from scaled derivatives to unscaled derivatives (dT/d(log(h)), dT/d(log(p)))
    dT_dx = dT_dx_s * (2 / x_range); % where x = h
    dT_dy = dT_dy_s * (2 / y_range); % where y = log(p)

    % 2. Convert from (log(h), log(p)) derivatives to (h, p) derivatives
    % d/dh = d/d(h) * d(h)/dh = d/dx * (1)
    % d/dp = d/d(log(p)) * d(log(p))/dp = d/dy * (1./p)
    d_dh = dT_dx .* (1);
    d_dp = dT_dy .* (1./p_new);
end