% function tools = levelset()
% % LEVELSET - A collection of functions for level-set operations.
% % Returns a struct of function handles that can be used by other scripts.
% 
%     % Return a struct where each field is a handle to a local function
%     tools.calculate_sdf_single_point = @calculate_sdf_single_point;
%     tools.normalize_sdf = @normalize_sdf;
%     tools.diff_sdf_single_point = @diff_sdf_single_point;
%     tools.diff_normalize_sdf = @diff_normalize_sdf;
% end
% 
% function normalized_value = normalize_sdf(sdf, k)
%     % Normalizes an SDF value to the range [0, 1] using a sigmoid function.
%     % Maps negative SDF (inside) to 1 and positive SDF (outside) to 0.
%     %
%     % INPUTS:
%     %   sdf - The signed distance value(s). Can be a scalar or a matrix.
%     %   k   - The sharpness parameter. A larger k means a steeper transition.
%     %
%     % OUTPUT:
%     %   normalized_value - The smoothly normalized value(s) between 0 and 1.
% 
%     normalized_value = 1 ./ (1 + exp(-k * sdf));
% end
% 
% function diff_normalized_value = diff_normalize_sdf(sdf, k)
%     Nsdf = normalize_sdf(sdf, k);
%     diff_normalized_value = k * exp(-k * sdf) ./ Nsdf.^2;
% end
% 
% function sdf = calculate_sdf_single_point(query_point, shape_data, k)
%     % Calculates the Signed Distance for a single query point from a pre-defined shape.
%     %
%     % INPUTS:
%     %   query_point - A 1x2 vector [x, y] of the point to test.
%     %   shape_data  - A struct containing the pre-defined curve and normal data.
%     %
%     % OUTPUT:
%     %   sdf         - A single scalar value: the signed distance.
%     %                 (Negative inside, positive outside, zero on the boundary).
% 
%     % 1. Find the index and distance to the closest point on the curve
%     %    'knnsearch' is highly optimized for this task.
%     [closest_idx, distance] = knnsearch(shape_data.points, query_point);
% 
%     % 2. Get the normal vector at that closest point
%     nx = shape_data.normal_nx(closest_idx);
%     ny = shape_data.normal_ny(closest_idx);
% 
%     % 3. Get the coordinates of the closest point on the curve
%     cx = shape_data.curve_x(closest_idx);
%     cy = shape_data.curve_y(closest_idx);
% 
%     % 4. Create the vector from the curve to our query point
%     vec_x = query_point(1) - cx;
%     vec_y = query_point(2) - cy;
% 
%     % 4. Calculate the dot product
%     dot_product = vec_x * nx + vec_y * ny;
% 
%     % 5. Use tanh for a smooth, differentiable "sign"
%     smooth_sign = tanh(k * dot_product);
% 
%     % 6. The differentiable SDF is the distance multiplied by the smooth sign
%     sdf = distance * smooth_sign;
% end
% %--------------------------------------------------------------------------
% 
% function grad = diff_sdf_single_point(query_point, shape_data, k)
%     % Calculates the gradient of the differentiable SDF with respect to the
%     % query point's x and y coordinates.
%     [closest_idx, distance] = knnsearch(shape_data.points, query_point);
%     nx = shape_data.normal_nx(closest_idx);
%     ny = shape_data.normal_ny(closest_idx);
%     cx = shape_data.curve_x(closest_idx);
%     cy = shape_data.curve_y(closest_idx);
%     vec_x = query_point(1) - cx;
%     vec_y = query_point(2) - cy;
% 
%     % At the boundary, the gradient is theoretically undefined due to knnsearch,
%     % but we can define it as zero for practical purposes.
%     if distance < 1e-9
%         grad = [0, 0];
%         return;
%     end
% 
%     dot_product = vec_x * nx + vec_y * ny;
%     tanh_val = tanh(k * dot_product);
%     sech_sq_val = 1 - tanh_val^2; % sech^2(x) = 1 - tanh^2(x)
%     unit_vec_x = vec_x / distance;
%     unit_vec_y = vec_y / distance;
% 
%     % Term 1 from differentiating tanh() part
%     term1_dx = distance * k * sech_sq_val * nx;
%     term1_dy = distance * k * sech_sq_val * ny;
%     % Term 2 from differentiating distance part
%     term2_dx = tanh_val * unit_vec_x;
%     term2_dy = tanh_val * unit_vec_y;
% 
%     grad = [term1_dx + term2_dx, term1_dy + term2_dy];
% end

function tools = levelset()
% LEVELSET_ANALYTIC - A collection of COMPLEX-SAFE (ANALYTIC) functions
% for level-set operations, suitable for complex-step differentiation.
%
% This version replaces the non-analytic 'knnsearch' with a
% continuous weighted-average algorithm.
%
% Returns a struct of function handles:
%   .calculate_sdf:   Analytic "SDF" calculation.
%   .normalize_sdf:   Analytic Heaviside/sigmoid function.

    tools.calculate_sdf = @calculate_sdf;
    tools.normalize_sdf = @normalize_sdf;
    % The 'diff' functions are removed. Use the complex-step method on the
    % functions above to get their derivatives.
end

%--------------------------------------------------------------------------

function normalized_value = normalize_sdf(sdf, k)
    % Normalizes an SDF value to the range [0, 1] using a sigmoid function.
    % This function IS already complex-safe.
    %
    % INPUTS:
    %   sdf - The signed distance value(s). Can be scalar/matrix (real or complex).
    %   k   - The sharpness parameter (real scalar).
    %
    % OUTPUT:
    %   normalized_value - The smoothly normalized value(s).

    % This is fully analytic.
    normalized_value = 1 ./ (1 + exp(-k .* sdf));
end

%--------------------------------------------------------------------------

function sdf = calculate_sdf(query_point, shape_data, k, delta)
    % This is the analytic version of your "distance * sign" function.

    vecs = query_point - shape_data.points;

    % --- Analytic "Distance" ---
    % dist = ||x - c||
    % We must use the analytic .dot(.) version
    dist_sq_analytic = vecs(:,1).^2 + vecs(:,2).^2;
    distances = sqrt(dist_sq_analytic + delta); % Nx1 vector of distances

    % --- Analytic "Sign" ---
    % sign = tanh(k * dot_product)
    dot_products = sum(vecs .* shape_data.normals, 2); % Nx1 vector
    smooth_signs = tanh(k .* dot_products); % Nx1 vector

    % --- Local SDF value for each point ---
    local_sdfs = distances .* smooth_signs; % This is (dist_i * sign_i)

    % --- Weighted Average ---
    % We still need weights to combine them
    weights = 1 ./ (dist_sq_analytic + delta);

    numerator = sum(local_sdfs .* weights);
    denominator = sum(weights);

    sdf = numerator ./ (denominator + delta);
end
% % This is the correct, analytic, and mathematically sound function.
% % It matches your C++ class.
% function sdf = calculate_sdf(query_point, shape_data, k, delta)
%     vecs = query_point - shape_data.points; 
% 
%     % 1. Calculate local (linear) SDF for each curve point.
%     %    This IS the correct linear approximation.
%     local_sdfs = sum(vecs .* shape_data.normals, 2);
% 
%     % 2. Calculate analytic-safe weights.
%     dist_sq_analytic = vecs(:,1).^2 + vecs(:,2).^2;
%     weights = 1 ./ (dist_sq_analytic + delta);
% 
%     % 3. Calculate the weighted average.
%     numerator = sum(local_sdfs .* weights);
%     denominator = sum(weights);
% 
%     sdf = numerator ./ (denominator + delta);
% end