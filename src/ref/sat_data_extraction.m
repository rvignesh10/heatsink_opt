clc
clear 

%% extract saturation data 

Header= readtable("RE347mcc_saturation_T_full.txt","VariableNamingRule","preserve");
file = readmatrix('RE347mcc_saturation_T_full.txt');

%% save saturation curve data
file1 = file(1:end, :);
skip = 0;

% go over file 1 sat L 
for i=1:length(file1(:,1))
    T = file1(i, 1);
    p = file1(i, 2)*1e+06;
    hL = file1(i, 5)*1e+03;
    if i ==1
        logH_logP_T = [log(hL), log(p), T];
    else
        logH_logP_T = [logH_logP_T; log(hL), log(p), T];
    end
end


% go over file1 sat V
for i=length(file1(:,1)):-1:1
    T = file1(i, 1);
    p = file1(i, 2)*1e+06;
    hV = file1(i, 6)*1e+03;
    logH_logP_T = [logH_logP_T; log(hV), log(p), T];
end


%% store the xcenters and normal vectors
x = logH_logP_T(:, 1)';
y = logH_logP_T(:, 2)';

% % close it?
% x1 = x(1); y1 = y(1); xN = x(end); yN = y(end);
% m = (yN-y1)/(xN-x1);
% eps = 1.0e-3;
% xadd = linspace(xN-eps, x1+eps, 10);
% yadd = m * (xadd-1) + y1;
% x = [x xadd];
% y = [y yadd];

xcenter = [x;y];
tangent = zeros(2,length(x));
normal  = zeros(size(xcenter));

theta = pi/2;
R = [cos(theta) -sin(theta); sin(theta) cos(theta)];

figure
for i=1:length(x)
    if i==1
        tangent(1, i) = ( x(i+1) - x(i) );
        tangent(2, i) = ( y(i+1) - y(i) );
    elseif i==length(x)
        tangent(1, i) = ( x(i) - x(i-1) );
        tangent(2, i) = ( y(i) - y(i-1) );
    else
        tangent(1, i) = ( x(i+1) - x(i-1) );
        tangent(2, i) = ( y(i+1) - y(i-1) );
    end
    normal(:, i) = R * tangent(:, i);
    normal(:, i) = normal(:, i)/norm( normal(:, i), 2 );
    plot([x(i) x(i)+normal(1,i)], [y(i) y(i)+normal(2,i)], 'k');
    hold on;
end

plot(x, y, 'r', 'LineWidth', 2);
curve_x = xcenter(1, :);
curve_y = xcenter(2, :);
normal_nx = normal(1, :);
normal_ny = normal(2, :);

%%
tools = levelset();
% --- Create the pre-defined shape structure ---
fprintf('Pre-defining the level-set shape...\n');
shape_data = struct();
shape_data.curve_x = curve_x(:); % Store as column vector
shape_data.curve_y = curve_y(:); % Store as column vector
shape_data.normal_nx = normal_nx(:); % Store as column vector
shape_data.normal_ny = normal_ny(:); % Store as column vector


% Pre-combine points for faster searching in the SDF function
shape_data.points = [shape_data.curve_x, shape_data.curve_y];
shape_data.normals= [shape_data.normal_nx, shape_data.normal_ny];
frame = zeros(2, length(shape_data.curve_x));
for i=1:length(shape_data.curve_x) 
    % Normalization is done in real-space (double), so .normalized() is safe.
    frame(:,i) = shape_data.normals(i, :)/norm(shape_data.normals(i, :));
end
shape_data.normals = frame.';
disp(shape_data);

[grid_x, grid_y] = meshgrid(linspace(min(x),max(x), 10), linspace(min(y), max(y), 10));
query_points = [grid_x(:), grid_y(:)];

fprintf('\n--- Calculating Normalized SDF for individual points ---\n');
sdf_value = zeros(size(query_points, 1), 1);
normalized_sdf_value = zeros(size(query_points, 1), 1);
k = 100000;
for i = 1:size(query_points, 1)
    pt = query_points(i, :);
    
    % Call the single-point SDF function
    sdf_value(i) = tools.calculate_sdf(pt, shape_data, k, 1e-10);
    normalized_sdf_value(i) = tools.normalize_sdf(sdf_value(i), k);
    
    fprintf('Point %d at [%.1f, %.1f]: N * SDF = %.4f\n', i, pt(1), pt(2), normalized_sdf_value(i));
end

% --- 4. Visualize the results ---
figure;
hold on; grid on; 
plot((shape_data.curve_x), shape_data.curve_y, 'k-', 'LineWidth', 2, 'DisplayName', 'Level Set Curve');

% Plot the query points
for i = 1:size(query_points, 1)
    pt = query_points(i, :);
    
    if normalized_sdf_value(i) > 1.0e-15
        plot((pt(1)), pt(2), 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 8, 'DisplayName', sprintf('Point %d (Outside)', i));
    else
        plot((pt(1)), pt(2), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8, 'DisplayName', sprintf('Point %d (Inside)', i));
    end
end
title('Testing Single Points against a Pre-defined Level Set');
% ylim([1, 25])
% legend('show', 'Location', 'northeastoutside');
% set(gca, 'YDir', 'reverse'); % Match image coordinates

%% save data for level set formulation

save('re347-sat-data', 'logH_logP_T', 'xcenter', 'tangent','normal');
