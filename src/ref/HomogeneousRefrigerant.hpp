#ifndef HOMOGENEOUS_REFRIGERANT_HPP
#define HOMOGENEOUS_REFRIGERANT_HPP

#include <Eigen/Dense>
#include <vector>
#include <complex>
#include <string>
#include "../utils/utils.hpp"
// #include "FlowPartials.hpp"

// --- Structs returned by helper functions ---
template <typename T>
struct SaturationProperties {
    T Tsat;
    T rho_satL;
    T rho_satV;
    T mu_satL;
    T mu_satV;
    T h_satL;
    T h_satV;
    T kappa_satL;
    T kappa_satV;
    T Cp_satL;
    T Cp_satV;
    T Pr_satL;
    T Pr_satV;

    SaturationProperties() {}

    template <typename U>
    SaturationProperties(const SaturationProperties<U>&other) : Tsat(T(other.Tsat)), 
                        rho_satL(T(other.rho_satL)), rho_satV(T(other.rho_satV)), 
                        mu_satL(T(other.mu_satL)), mu_satV(T(other.mu_satV)),
                        h_satL(T(other.h_satL)), h_satV(T(other.h_satV)), 
                        kappa_satL(T(other.kappa_satL)), kappa_satV(T(other.kappa_satV)),
                        Cp_satL(T(other.Cp_satL)), Cp_satV(T(other.Cp_satV)),
                        Pr_satL(T(other.Pr_satL)), Pr_satV(T(other.Pr_satV)) {}
};

template <typename T>
struct CharacteristicProperties {
    T p_c;
    T T_c;
    T L_c;
    T rho_c;
    T mu_c;
    T h_c;
    T u_c;
    T g_c;
    T htc_c;
    T mdot_c;
    T flux_c;
    T Phi_c;
    T kappa_c;
    T Cp_c;

    CharacteristicProperties() {}

    template <typename U>
    CharacteristicProperties(const CharacteristicProperties<U>& other) : p_c(T(other.p_c)), T_c(T(other.T_c)),
                            L_c(T(other.L_c)), rho_c(T(other.rho_c)), mu_c(T(other.mu_c)), h_c(T(other.h_c)), 
                            u_c(T(other.u_c)), g_c(T(other.g_c)), htc_c(T(other.htc_c)), mdot_c(T(other.mdot_c)),
                            flux_c(T(other.flux_c)), Phi_c(T(other.Phi_c)), kappa_c(T(other.kappa_c)), Cp_c(T(other.Cp_c)) {}
};

// Helper to convert double vector to T vector
template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> loadCoeffs(const std::vector<double>& coeffs_double) {
    Eigen::VectorXd v = Eigen::Map<const Eigen::VectorXd>(coeffs_double.data(), coeffs_double.size());
    return v.template cast<T>();
}

/**
 * @class HomogeneousRefrigerant
 * @brief New Abstract Base Class based on Research_Notes-7.pdf
 */
template <typename T>
class HomogeneousRefrigerant {
public:
    // --- Type Aliases ---
    using TemplateVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    virtual ~HomogeneousRefrigerant() = default;

    // --- Public Member Functions from PDF ---
    virtual T calc_levelset(T h, T p) = 0;
    virtual TemplateVector diff_levelset(T h, T p) = 0; // [d/dh, d/dp]

    virtual T calc_satL_density(T Temp) = 0;
    virtual T calc_satV_density(T p) = 0;
    // "returns drho_dx"
    virtual T diff_homogeneous_density(T h, T p, T chi, T densityL, T densityV) = 0;
    // virtual FlowStatePartials calc_homogeneous_drho_partials(T h, T p, T chi, T densityL, T densityV) = 0;

    virtual T calc_satL_viscosity(T Temp) = 0;
    virtual T calc_satV_viscosity(T p) = 0;
    // "returns dmu_dx"
    virtual T diff_homogeneous_viscosity(T h, T p, T chi,T viscosityL, T viscosityV) = 0;
    // virtual FlowStatePartials calc_homogenous_dmu_partials(T h, T p, T chi, T viscosityL, T viscosityV) = 0;

    virtual T calc_sat_pressure(T Temp) = 0;
    virtual T calc_sat_temperature(T p) = 0;
    virtual T diff_sat_temperature(T p) = 0;

    virtual T calc_homogeneous_temperature(T h, T p) = 0;
    // "return dT_dh, dT_dp"
    virtual TemplateVector diff_homogeneous_temperature(T h, T p) = 0;

    virtual T calc_satL_enthalpy(T Temp) = 0;
    virtual T diff_satL_enthalpy(T Temp) = 0;
    virtual T calc_satV_enthalpy(T Temp) = 0;
    // "returns dx_dh"
    virtual T diff_homogeneous_quality(T h, T p, T enthalpyL, T enthalpyV) = 0;

    virtual T calc_satL_kappa(T Temp) = 0;
    virtual T calc_satV_kappa(T p) = 0;

    virtual T calc_satL_Cp(T p) = 0;
    virtual T calc_satV_Cp(T p) = 0;

    virtual T calc_friction_factor(T Re) = 0;
    virtual T diff_friction_factor(T Re) = 0;

    T calc_friction_factor_sp(T Re) {
        T f_lam = T(16.0) / Re;
        // T f_tur = std::pow( T(1.58) * std::log(Re) - T(3.28), -2.0 );
        T f_tur = T(0.079) * std::pow(Re, -0.25);
        T b1 = blend(Re, T(3000.0), T(10.0));
        return (T(1.0) - b1) * f_lam + b1 * f_tur;
    }

    T diff_friction_factor_sp(T Re) {
        T f_lam = T(64.0) / Re;
        T dfdRe_lam = T(-64.0) / (Re * Re);

        // T f_tur = std::pow( T(1.58) * std::log(Re) - T(3.28), -2.0 );
        // T dfdRe_tur = T(-2.0) * std::pow( T(1.58) * std::log(Re) - T(3.28), -3.0 ) * T(1.58) / Re;
        T f_tur = T(0.079) * std::pow(Re, -0.25);
        T dfdRe_tur = T(-0.079 * 0.25) * std::pow(Re, -0.25-1.0);

        T b1 = blend(Re, T(3000.0), T(10.0)); T db1 = Dblend(Re, T(3000.0), T(10.0));
        return -db1 * f_lam + (T(1.0) - b1) * dfdRe_lam + db1 * f_tur + b1 * dfdRe_tur;
    }

    // --- Helpers (templated return) ---
    SaturationProperties<T> calc_saturation_properties(T p_out) {
        T Tsat = this->calc_sat_temperature(p_out);
        
        // saturatred liquid and vapor enthalpy at exit pressure 
        T hsatL = this->calc_satL_enthalpy(Tsat);
        T hsatV = this->calc_satV_enthalpy(Tsat);

        // saturated liquid and vapor density at exit pressure
        T rsatL = this->calc_satL_density(Tsat);
        T rsatV = this->calc_satV_density(p_out);

        // saturated liquid and vapor viscosity at exit pressure
        T msatL = this->calc_satL_viscosity(Tsat);
        T msatV = this->calc_satV_viscosity(p_out);

        // saturated liquid and vapor thermal conductivity at exit pressure
        T kappaL = this->calc_satL_kappa(Tsat);
        T kappaV = this->calc_satV_kappa(p_out);

        // saturated liquid and vapor Cp at exit pressure
        T CpL = this->calc_satL_Cp(p_out);
        T CpV = this->calc_satV_Cp(p_out);

        // saturated liquid and vapor Prandtl number at exit pressure
        T PrL = msatL * CpL / kappaL;
        T PrV = msatV * CpV / kappaV;

        SaturationProperties<T> sat_props;
        sat_props.Tsat = Tsat;
        sat_props.rho_satL = rsatL;
        sat_props.rho_satV = rsatV;
        sat_props.mu_satL  = msatL;
        sat_props.mu_satV  = msatV;
        sat_props.h_satL   = hsatL;
        sat_props.h_satV   = hsatV;
        sat_props.kappa_satL = kappaL;
        sat_props.kappa_satV = kappaV;
        sat_props.Cp_satL  = CpL;
        sat_props.Cp_satV  = CpV;
        sat_props.Pr_satL  = PrL;
        sat_props.Pr_satV  = PrV;

        return sat_props;
    }


    CharacteristicProperties<T> calc_characteristic_properties(T T_in, T p_out, T L_c) {
        SaturationProperties<T> sat = calc_saturation_properties(p_out);
        CharacteristicProperties<T> scales;
        scales.p_c = p_out;
        scales.T_c = T_in;
        scales.L_c = L_c;
        scales.rho_c = sat.rho_satL;
        scales.mu_c = sat.mu_satL;
        scales.h_c = sat.h_satV - sat.h_satL;
        scales.u_c = std::sqrt(scales.p_c / scales.rho_c);
        scales.g_c = scales.p_c / (scales.rho_c * scales.L_c);
        scales.Phi_c = (scales.rho_c * scales.u_c * scales.h_c) / scales.L_c;
        scales.kappa_c = scales.Phi_c * scales.L_c * scales.L_c/ scales.T_c;
        scales.htc_c = scales.Phi_c * scales.L_c / scales.T_c;
        scales.flux_c = scales.Phi_c * scales.L_c;
        scales.mdot_c = scales.rho_c * scales.u_c * scales.L_c * scales.L_c;
        scales.Cp_c = scales.h_c / scales.T_c;
        return scales;
    }

};


#endif // HOMOGENEOUS_REFRIGERANT_HPP

