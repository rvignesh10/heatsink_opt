clc
clear
%% get required files

fileList = {'RE347mcc_isothermal_full.txt'};

%% gather required information {p, h, T} and form {log(p*), h*, T*}

for i=1:length(fileList)
    tab = readmatrix(fileList{i});
    if i==1
        T = (tab(1:end, 1));
        v = 1./(tab(1:end, 3));
        m = tab(1:end, 10)*1e-06;
        p = tab(1:end, 2) * 1e+06;
        h = tab(1:end, 4) * 1e+03;
    else
        T = [T; (tab(1:end, 1))];
        v = [v; 1./tab(1:end, 3)];
        m = [m; tab(1:end, 10)*1e-06];
        p = [p; tab(1:end, 2) * 1e+06];
        h = [h; tab(1:end, 4) * 1e+03];
    end
end

h_logP_T_v_m = [h, log(p), T, v, m];
clear T p h v m

%% save data as .mat file 

save('re347-r1-data', "h_logP_T_v_m" );

%% plot and visualize data

figure
scatter3(h_logP_T_v_m(:, 1), h_logP_T_v_m(:, 2), h_logP_T_v_m(:, 3));

figure
scatter3(h_logP_T_v_m(:, 1), h_logP_T_v_m(:, 2), h_logP_T_v_m(:, 4));

figure
scatter3(h_logP_T_v_m(:, 1), h_logP_T_v_m(:, 2), h_logP_T_v_m(:, 5));