clc
clear

%% extract saturation data 

Header= readtable("RE347mcc_saturation_T_full.txt","VariableNamingRule","preserve");
file = readmatrix('RE347mcc_saturation_T_full.txt');

%% sample saturation data

N = 50;
k = 1;
for i=1:length(file(:,1))
    T = (file(i, 1));
    p = file(i, 2)*1e+06;
    hL = file(i, 5)*1e+03;
    hV = file(i, 6)*1e+03;
    vL = 1./file(i, 3);
    vV = 1./file(i, 4);
    mL = file(i, 17)*1e-6;
    mV = file(i, 18)*1e-6;
    h  = linspace(hL, hV, N)';
    m  = linspace(mL, mV, N)';
    quality = linspace(0, 1, N)';
    v = linspace(vL, vV, N)';
    if k==1
        T_d = T * ones(N,1);
        p_d = p * ones(N,1);
        h_d = h;
        v_d = v;
        m_d = m;
        q_d = quality;
    else
        T_d = [T_d; T * ones(N,1)];
        p_d = [p_d; p * ones(N,1)];
        h_d = [h_d; h];
        v_d = [v_d; v];
        m_d = [m_d; m];
        q_d = [q_d; quality];
    end
    k = k + 1;
end

h_logP_T_v_m_q = [h_d, log(p_d), T_d, v_d, m_d, q_d];
clear p_d T_d h_d v_d m_d q_d

%% save data

save('re347-r2-data', 'h_logP_T_v_m_q');

%% plot and visualize data

figure
scatter3(h_logP_T_v_m_q(:, 1), h_logP_T_v_m_q(:, 2), h_logP_T_v_m_q(:, 3));

figure
scatter3(h_logP_T_v_m_q(:, 1), h_logP_T_v_m_q(:, 2), h_logP_T_v_m_q(:, 4));

figure
scatter3(h_logP_T_v_m_q(:, 1), h_logP_T_v_m_q(:, 2), h_logP_T_v_m_q(:, 5));

figure
scatter3(h_logP_T_v_m_q(:, 1), h_logP_T_v_m_q(:, 2), h_logP_T_v_m_q(:, 6));