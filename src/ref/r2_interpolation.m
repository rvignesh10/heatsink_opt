clc
clear
%%
load('re347-r2-data.mat');
h = h_logP_T_v_m_q(:, 1);
p = exp(h_logP_T_v_m_q(:, 2));
T = h_logP_T_v_m_q(:, 3);
vol = h_logP_T_v_m_q(:, 4);
mu = h_logP_T_v_m_q(:, 5);
quality = h_logP_T_v_m_q(:, 6);

%% --- B. Create the Interpolating Functions ---
H_DEGREE_T = 10; % Polynomial degree for enthalpy (h)
P_DEGREE_T = 10; % Polynomial degree for pressure (p)

H_DEGREE_V = 10; % Polynomial degree for enthalpy (h)
P_DEGREE_V = 10; % Polynomial degree for pressure (p)

H_DEGREE_M = 10; % Polynomial degree for enthalpy (h)
P_DEGREE_M = 10; % Polynomial degree for pressure (p)

H_DEGREE_Q = 10;
P_DEGREE_Q = 10;

fprintf('Creating interpolators with Legendre polynomials of h and p...\n');

% Create function handles for the interpolators
[T_func, Tcoeffs, Tcoeff_indices, Tx_min, Ty_min, Tx_range, Ty_range] = create_2d_legendre_interpolator(h, p, T, H_DEGREE_T, P_DEGREE_T);
[vol_func, Vcoeffs, Vcoeff_indices, Vx_min, Vy_min, Vx_range, Vy_range] = create_2d_legendre_interpolator(h, p, vol, H_DEGREE_V, P_DEGREE_V);
[mu_func, Mcoeffs, Mcoeff_indices, Mx_min, My_min, Mx_range, My_range] = create_2d_legendre_interpolator(h, p, mu, H_DEGREE_M, P_DEGREE_M);
[quality_func, Qcoeffs, Qcoeff_indices, Qx_min, Qy_min, Qx_range, Qy_range] = create_2d_legendre_interpolator(h, p, quality, H_DEGREE_Q, P_DEGREE_Q);

fprintf('Interpolating functions T(h,p), vol(h,p), and mu(h,p) created successfully.\n');
Tfsave = 1;
Vfsave = 0;
Mfsave = 0;
Qfsave = 0;
%% --- B.1 Save Interpolation variables ---
if Tfsave == 1
    Tfunc_data.h_degree = H_DEGREE_T;
    Tfunc_data.p_degree = P_DEGREE_T;
    Tfunc_data.coeffs = Tcoeffs;
    Tfunc_data.coeff_indices = Tcoeff_indices;
    Tfunc_data.x_min = Tx_min;
    Tfunc_data.x_range = Tx_range;
    Tfunc_data.y_min = Ty_min;
    Tfunc_data.y_range = Ty_range;
    save('re347-r2-Tfunc.mat', "Tfunc_data");
end

if Vfsave == 1
    Vfunc_data.h_degree = H_DEGREE_V;
    Vfunc_data.p_degree = P_DEGREE_V;
    Vfunc_data.coeffs = Vcoeffs;
    Vfunc_data.coeff_indices = Vcoeff_indices;
    Vfunc_data.x_min = Vx_min;
    Vfunc_data.x_range = Vx_range;
    Vfunc_data.y_min = Vy_min;
    Vfunc_data.y_range = Vy_range;
    save('re347-r2-Vfunc.mat', "Vfunc_data");
end

if Mfsave == 1
    Mfunc_data.h_degree = H_DEGREE_M;
    Mfunc_data.p_degree = P_DEGREE_M;
    Mfunc_data.coeffs = Mcoeffs;
    Mfunc_data.coeff_indices = Mcoeff_indices;
    Mfunc_data.x_min = Mx_min;
    Mfunc_data.x_range = Mx_range;
    Mfunc_data.y_min = My_min;
    Mfunc_data.y_range = My_range;
    save('re347-r2-Mfunc.mat', "Mfunc_data");
end

if Qfsave == 1
    Qfunc_data.h_degree = H_DEGREE_Q;
    Qfunc_data.p_degree = P_DEGREE_Q;
    Qfunc_data.coeffs = Qcoeffs;
    Qfunc_data.coeff_indices = Qcoeff_indices;
    Qfunc_data.x_min = Qx_min;
    Qfunc_data.x_range = Qx_range;
    Qfunc_data.y_min = Qy_min;
    Qfunc_data.y_range = Qy_range;
    save('re347-r2-Qfunc.mat', "Qfunc_data");
end

%% --- C. Evaluate Interpolation Error ---
% Predict values at the original data points to assess the fit quality.
% We only need the first output (the predicted value), so we use ~ to ignore the derivatives.
[T_pred_orig, ~, ~] = T_func(h, p);
[vol_pred_orig, ~, ~] = vol_func(h, p);
[mu_pred_orig, ~, ~] = mu_func(h, p);
[q_pred_orig, ~, ~] = quality_func(h, p);

% Calculate Root Mean Square Error (RMSE)
T_rmse = sqrt(mean((T_pred_orig - T).^2));
vol_rmse = sqrt(mean((vol_pred_orig - vol).^2));
mu_rmse = sqrt(mean((mu_pred_orig - mu).^2));
q_rmse  = sqrt(mean((q_pred_orig - quality).^2));

% Calculate Mean Absolute Percentage Error (MAPE)
T_mape = mean(abs((T_pred_orig - T) ./ T)) * 100;
vol_mape = mean(abs((vol_pred_orig - vol) ./ vol)) * 100;
mu_mape = mean(abs((mu_pred_orig - mu) ./ mu)) * 100;
q_mape = mean(abs((q_pred_orig-quality)./quality)) * 100;

fprintf('\n--- Goodness of Fit ---\n');
fprintf('Temperature Fit:\n  - RMSE: %.4f K\n  - MAPE: %.2f%%\n', T_rmse, T_mape);
fprintf('Density Fit:\n  - RMSE: %.4f kg/m^3\n  - MAPE: %.2f%%\n', vol_rmse, vol_mape);
fprintf('Viscosity Fit:\n  - RMSE: %.2e Pa*s\n  - MAPE: %.2f%%\n', mu_rmse, mu_mape);
fprintf('Quality Fit:\n  - RMSE: %.2e \n  - MAPE: %.2f%%\n', q_rmse, q_mape);
  
%% --- E. Visualize the Interpolated Surfaces ---

% --- Temperature Visualization ---
figure('Name', 'Temperature Interpolation', 'Position', [100 100 900 650]);
ax_T = axes('Parent', gcf);

% Create a meshgrid for the surface plots (can be reused for all plots)
h_grid = linspace(min(h), max(h), 30);
p_grid = linspace(min(p), max(p), 30);
[H_mesh, P_mesh] = meshgrid(h_grid, p_grid);

% Predict T over the entire grid
T_surf = T_func(H_mesh, P_mesh);

% Plot the original data points, colored by their T value
scatter3(ax_T, h, p, T, 60, T, 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'Original Data Points');
hold(ax_T, 'on');

% Plot the interpolated surface for Temperature
surf(ax_T, H_mesh, P_mesh, T_surf, 'FaceAlpha', 0.8, 'EdgeColor', 'none', 'DisplayName', 'Interpolated Surface');

% Set color limits and z-axis limits to the range of the original data
clim(ax_T, [min(T), max(T)]);
zlim(ax_T, [min(T), max(T)]);

title(ax_T, 'Interpolated Temperature Surface vs. Original Data');
xlabel(ax_T, 'Enthalpy h (J/kg)');
ylabel(ax_T, 'Pressure p (Pa)');
zlabel(ax_T, 'Temperature T (K)');
grid(ax_T, 'on');
view(ax_T, -45, 20);
legend(ax_T, 'show');
colormap(ax_T, 'parula');
colorbar(ax_T);


% --- Density (rho) Visualization ---
figure('Name', 'Volume Interpolation', 'Position', [150 150 900 650]);
ax_vol = axes('Parent', gcf);

% Predict rho over the entire grid
vol_surf = vol_func(H_mesh, P_mesh);

% Plot the original data points, colored by their rho value
scatter3(ax_vol, h, p, vol, 60, vol, 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'Original Data Points');
hold(ax_vol, 'on');

% Plot the interpolated surface for Density
surf(ax_vol, H_mesh, P_mesh, vol_surf, 'FaceAlpha', 0.8, 'EdgeColor', 'none', 'DisplayName', 'Interpolated Surface');

% Set color limits and z-axis limits to the range of the original data
clim(ax_vol, [min(vol), max(vol)]);
zlim(ax_vol, [min(vol), max(vol)]);

title(ax_vol, 'Interpolated Volume Surface vs. Original Data');
xlabel(ax_vol, 'Enthalpy h (J/kg)');
ylabel(ax_vol, 'Pressure p (Pa)');
zlabel(ax_vol, 'Volume (m^3/kg)');
grid(ax_vol, 'on');
view(ax_vol, -45, 20);
legend(ax_vol, 'show');
colormap(ax_vol, 'parula');
colorbar(ax_vol);


% --- Viscosity (mu) Visualization ---
figure('Name', 'Viscosity Interpolation', 'Position', [200 200 900 650]);
ax_mu = axes('Parent', gcf);

% Predict mu over the entire grid
mu_surf = mu_func(H_mesh, P_mesh);

% Plot the original data points, colored by their mu value
scatter3(ax_mu, h, p, mu, 60, mu, 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'Original Data Points');
hold(ax_mu, 'on');

% Plot the interpolated surface for Viscosity
surf(ax_mu, H_mesh, P_mesh, mu_surf, 'FaceAlpha', 0.8, 'EdgeColor', 'none', 'DisplayName', 'Interpolated Surface');

% Set color limits and z-axis limits to the range of the original data
clim(ax_mu, [min(mu), max(mu)]);
zlim(ax_mu, [min(mu), max(mu)]);

title(ax_mu, 'Interpolated Viscosity Surface vs. Original Data');
xlabel(ax_mu, 'Enthalpy h (J/kg)');
ylabel(ax_mu, 'Pressure p (Pa)');
zlabel(ax_mu, 'Viscosity mu (Pa*s)');
grid(ax_mu, 'on');
view(ax_mu, -45, 20);
legend(ax_mu, 'show');
colormap(ax_mu, 'parula');
colorbar(ax_mu);


figure('Name', 'Quality Interpolation (Log-Log)', 'Position', [200 200 900 650]);
ax_qu = axes('Parent', gcf);
qu_surf = quality_func(H_mesh, P_mesh);
scatter3(ax_qu, h, p, quality, 60, quality, 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'Original Data');
hold(ax_qu, 'on');
surf(ax_qu, H_mesh, P_mesh, qu_surf, 'FaceAlpha', 0.8, 'EdgeColor', 'none', 'DisplayName', 'Interpolated Surface');
clim(ax_qu, [min(quality), max(quality)]);
zlim(ax_qu, [min(quality), max(quality)]);
title(ax_qu, 'Interpolated Quality Surface (Log-Log Model)');
xlabel(ax_qu, 'Enthalpy h (J/kg)'); ylabel(ax_qu, 'Pressure p (Pa)'); zlabel(ax_qu, 'Quality');
grid(ax_qu, 'on'); view(ax_qu, -45, 20); legend(ax_qu, 'show'); colormap(ax_qu, 'parula'); colorbar(ax_qu);

%% --- Function Definitions ---

function [interpolator, coeffs, coeff_indices, x_min, y_min, x_range, y_range] = create_2d_legendre_interpolator(h_data, p_data, target_data, h_degree, p_degree)
    % Creates a 2D interpolating function using Legendre polynomials.
    % The function will be of the form f(log(h), log(p)).

    % --- 1. Prepare the independent variables (log(h), log(p)) ---
    x_data = h_data;
    y_data = log(p_data);

    % --- 2. Scale the data to the Legendre domain [-1, 1] ---
    x_min = min(x_data); x_max = max(x_data);
    y_min = min(y_data); y_max = max(y_data);

    x_range = x_max - x_min; if x_range == 0, x_range = 1; end
    y_range = y_max - y_min; if y_range == 0, y_range = 1; end

    x_scaled = 2 * (x_data - x_min) / x_range - 1;
    y_scaled = 2 * (y_data - y_min) / y_range - 1;

    % --- 3. Construct the design matrix using Legendre polynomials ---
    L_x = legendreP(h_degree, x_scaled);
    L_y = legendreP(p_degree, y_scaled);

    design_matrix_cols = {};
    coeff_indices = [];
    for i = 0:h_degree
        for j = 0:p_degree
            % This column corresponds to the basis function L_i(x) * L_j(y)
            design_matrix_cols{end+1} = L_x(:, i+1) .* L_y(:, j+1);
            coeff_indices = [coeff_indices; i, j];
        end
    end
    design_matrix = horzcat(design_matrix_cols{:});
    
    % --- 4. Solve for the polynomial coefficients using least squares ---
    % In MATLAB, the backslash operator is the standard for least squares
    coeffs = design_matrix \ target_data;

    % --- 5. Return a closure (the interpolating function) ---
    % This anonymous function captures the coefficients and scaling parameters.
    interpolator = @(h_new, p_new) eval_legendre_surface_with_derivs(h_new, p_new, ...
        coeffs, coeff_indices, h_degree, p_degree, x_min, x_range, y_min, y_range);
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


function [result, d_dh, d_dp] = eval_legendre_surface_with_derivs(h_new, p_new, coeffs, coeff_indices, h_degree, p_degree, x_min, x_range, y_min, y_range)
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