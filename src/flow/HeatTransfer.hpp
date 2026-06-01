#ifndef HEAT_TRANSFER_HPP
#define HEAT_TRANSFER_HPP

#include <string>
#include <vector>
#include "../utils/utils.hpp"
#include "../ref/HomogeneousRefrigerant.hpp"
#include "../ref/R11.hpp" 
#include "../ref/H2O.hpp" 
#include "../ref/HFE7000.hpp"


template <typename T>
struct PipeParameters {
    // --- Type Aliases ---
    using TemplateVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using TemplateMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    T D; 
    PipeParameters() : D(T(0.001)) {}
    template<typename U>
    PipeParameters(const PipeParameters<U>& other) : D(T(other.D)) {}
};


template <typename T>
struct FlowLoadParameters {
    using TemplateVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using TemplateMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    bool f_heat_flux;
    int nxi;
    TemplateVector xi_star;
    TemplateVector g_star;
    TemplateVector load_star;
    FlowLoadParameters() : f_heat_flux(false), nxi(0) {}
    FlowLoadParameters(bool f_heat_flux_, int nxi_, 
                        TemplateVector xi_star_, 
                        TemplateVector g_star_, 
                        TemplateVector load_star_) :
                        f_heat_flux(f_heat_flux_), nxi(nxi_), 
                        xi_star(xi_star_), g_star(g_star_), 
                        load_star(load_star_) {
        if (xi_star.size() != nxi || g_star.size()!= nxi || load_star.size()!= nxi) {
            throw std::runtime_error("FlowLoadParameters: Input vectors (xi, g, load) "
                                     "must all have the same length.");
        }
    }

    template<typename U>
    FlowLoadParameters(const FlowLoadParameters<U>& other) : 
        f_heat_flux(other.f_heat_flux), nxi(other.nxi),
        xi_star(other.xi_star.template cast<T>()), 
        g_star(other.g_star.template cast<T>()),
        load_star(other.load_star.template cast<T>()) {
        if (xi_star.size() != nxi || g_star.size()!= nxi || load_star.size()!= nxi) {
            throw std::runtime_error("FlowLoadParameters: Input vectors (xi, g, load) "
                                     "must all have the same length.");
        }
    }
    const int getNumChannelNodes() const {return nxi;}
};


// --- Factory function to create the correct refrigerant type ---
template <typename T>
inline std::unique_ptr<HomogeneousRefrigerant<T>> createRefrigerant(const std::string& ref_name) {
    if (ref_name == "R11") {
        return std::make_unique<R11_Refrigerant<T>>();
    }
    if (ref_name == "H2O" || ref_name == "water") {
        return std::make_unique<H2O_Refrigerant<T>>();
    }
    if (ref_name == "HFE7000") {
        return std::make_unique<HFE7000_Refrigerant<T>>(); 
    }
    
    std::cerr << "Warning: Refrigerant '" << ref_name << "' not recognized. Defaulting to R11." << std::endl;
    return std::make_unique<R11_Refrigerant<T>>();
}

template<typename T>
inline T non_dimensionalize(const T& v, const T v_c) { return v / v_c; }

template<typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> non_dimensionalize(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v, const T v_c) { 
    return v / v_c; 
}

template<typename T>
inline T dimensionalize(const T& v_star, const T v_c) { return v_c * v_star; }

template<typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> dimensionalize(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v_star, const T v_c) {
    return v_c * v_star;
}


// --- Boundary Conditions Struct ---
template<typename T>
struct FlowBoundaryConditions {
    T pin_star;  
    T pout_star; 
    T Tin_star;  
    T hin_star;  
    T mdot_star;  
    std::string refrigerant_name;

    FlowBoundaryConditions() = default;
    template <typename U>
    FlowBoundaryConditions(const FlowBoundaryConditions<U>& other) : 
                        pin_star(T(other.pin_star)), pout_star(T(other.pout_star)), 
                        Tin_star(T(other.Tin_star)), hin_star(T(other.hin_star)), 
                        mdot_star(T(other.mdot_star)), refrigerant_name(other.refrigerant_name) {}
};

template<typename T>
struct FlowStatePartials {
    T d_du = T(0.0);   // partial w.r.t to velocity
    T d_dp = T(0.0);   // partial w.r.t to pressure
    T d_dh = T(0.0);   // partial w.r.t to enthalpy
    T d_dchi = T(0.0); // partial w.r.t to vapor quality
    T d_drho = T(0.0); // partial w.r.t to density
    T d_dmu = T(0.0);  // partial w.r.t to viscosity 
    T d_dT = T(0.0);   // partial w.r.t to temperature
};

template<typename T>
struct FlowInputPartials {
    T d_dxi = T(0.0);  // partial w.r.t to xi - length along channel
    T d_dTw = T(0.0);  // partial w.r.t to wall temperature along channel if f_heat_flux = false
    T d_dPhi = T(0.0); // partial w.r.t to heat/unit_vol if f_heat_flux = true
};


template <typename T>
class ConvectiveHeatTransfer{
private:
    static constexpr T xi_fd = T(0.05);
    static constexpr T k_fd  = T(20.0);

    static constexpr T Re_lam= T(3000.0);
    static constexpr T k_lam = T(500.0);

    static constexpr T Re_trans= T(1.0e+04);
    static constexpr T k_trans = T(10.0);
public:

    ConvectiveHeatTransfer<T>(){}
    ~ConvectiveHeatTransfer<T>(){}

    T calc_Nusselt_number_thermal_developing(T xi, T D, T Re, T Pr) {
        T delta = T(1.0e-03);
        T L_d = ( (xi / D) / (Re * Pr) ) + delta;
        return T(4.36) + (T(0.00668)/L_d) / (T(1.0) + T(0.04) * std::pow(L_d, 2./3.));
    }

    /**
     * @brief Calculates Nu = (1-b1)*Nu_lam + b1*((1-b2)*Nu_gnie + b2*Nu_db).
     * @param Re single phase Reynolds number of flow.
     * @param Pr single phase Prandtl number of flow
     * @return single phase Nu.
     */
    T calc_Nusselt_number_fully_developed(T Re, T Pr) {
        T Nu_lam = T(4.36);

        // Gnielinski
        T f = std::pow(T(1.58)*std::log(Re)-T(3.28), -2.);
        T Nr = T(0.5)*f * (Re - T(1000.)) * Pr;
        T Dr = T(1.0) + T(12.7) * std::sqrt(T(0.5)*f) * (std::pow(Pr, 2./3.) - T(1.0));
        T Nu_gnie = Nr / Dr ;

        // Dittus-Boelter
        T Nu_db = T(0.023) * std::pow(Re, 0.8) * std::pow(Pr, 0.4);

        // blending
        T b1 = blend(Re, Re_lam, k_lam);
        T b2 = blend(Re, Re_trans, k_trans);

        return (T(1.0) - b1) * Nu_lam + b1*((T(1.0) - b2)*Nu_gnie + b2*Nu_db);
    }


    /**
     * @brief calculate single phase heat transfer coefficient Nu * kappa / D
     * @param D diameter of channel
     * @param u mean two phase velocity of flow
     * @param rho single phase density of fluid
     * @param mu single phase viscosity of fluid
     * @param Pr single phase Prandtl number of flow
     * @param kappa single phase thermal conductivity of fluid
     * @return htc_{*} - single phase heat transfer coefficient
     */
    T calc_single_phase_htc(T xi, T D, T u, T rho, T mu, T Pr, T kappa) {
        T Re = (rho * u * D) / mu;
        T Nu_dev = calc_Nusselt_number_thermal_developing(xi, D, Re, Pr);
        T Nu_fd = calc_Nusselt_number_fully_developed(Re, Pr);
        T b1 = blend(xi, xi_fd, k_fd);
        T Nu = (T(1.0) - b1) * Nu_dev + b1 * Nu_fd;
        // T Nu = calc_Nusselt_number_fully_developed(Re, Pr);
        return Nu * kappa / D;
    }


    /**
     * @brief calculate two-phase heat transfer coefficient htc = 1.2*HTC_l - (0.3 - quality)**2 * (2.222*HTC_l + quality*(0.227*HTC_l-2.041*HTC_v))
     * @param D diameter of channel
     * @param u mean two-phase velocity of flow
     * @param chi two-phase vapor quality of fluid
     * @param sat_props_exit saturation properties calculated based on the exit pressure
     * @return Htp - two-phase flow heat transfer coefficient
     */
    T calculate_cvHTC(T xi, T D, T u, T chi, const SaturationProperties<T>& sat_props_exit) {
        T htc_l = calc_single_phase_htc(xi, D, u, sat_props_exit.rho_satL, sat_props_exit.mu_satL, 
                                        sat_props_exit.Pr_satL, sat_props_exit.kappa_satL);
        
        T htc_v = calc_single_phase_htc(xi, D, u, sat_props_exit.rho_satV, sat_props_exit.mu_satV, 
                                        sat_props_exit.Pr_satV, sat_props_exit.kappa_satV);
        
        return T(1.2)*htc_l - std::pow(T(0.3) - chi, 2.0) * (T(2.222)*htc_l + chi*(T(0.227)*htc_l - T(2.041)*htc_v));
    }


    /**
     * @brief calculates Heat/unitVol, Phi = (4/D) * Htp * (Twall - Tref)
     * @param D diameter of the channel
     * @param u mean two-phase velocity of flow
     * @param chi vapor quality of fluid
     * @param Tref Temperature of the fluid
     * @param Twall Temperature of the wall 
     * @param sat_props_exit Saturation properties calculated at the exit
     * @return Phi - heat/unitVol based on convective heat-transfer from the wall
     */
    T calculate_heat_load(T xi, T D, T u, T chi, T Tref, T Twall, const SaturationProperties<T>& sat_props_exit) {
        T cvHTC = calculate_cvHTC(xi, D, u, chi, sat_props_exit);
        return (T(4.0)/D) * cvHTC * (Twall - Tref);
    }


};

#endif