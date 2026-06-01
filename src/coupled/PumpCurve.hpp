#ifndef PUMP_CURVE_HPP
#define PUMP_CURVE_HPP

#include <iostream>
#include <Eigen/Dense>
#include "../ref/HomogeneousRefrigerant.hpp"
#include "../ref/R11.hpp"
#include "../ref/H2O.hpp"

class PumpConstraints {
private:
    CharacteristicProperties<double> char_props;
    static constexpr double KSlip = (2.904 / 76.0) * (1.0 / 60000.0) * (1.0 / 6894.76); // (m^3/s) / Pa
    static constexpr double Disp  = 4.6e-06;                                            // m^3/rev
    static constexpr double Nmax  = 3450.0 / 60.0;                                      // rev / s
    static constexpr double Nmin  = 500.0 / 60.0;                                       // rev / s
    static constexpr double Dp_max= 125.0 * 6894.76;                                    // Pa

public:

    PumpConstraints(const CharacteristicProperties<double> char_props_) : char_props(char_props_) 
    {  }

    double calcNDPumpSpeed(const double& pressure_drop_star, 
                           const double& vol_flow_rate_star) 
    {
        double pressure_drop_Pa = pressure_drop_star * this->char_props.p_c;
        double qdot_m3_p_s      = vol_flow_rate_star * this->char_props.mdot_c / this->char_props.rho_c;
        double N_rev_p_s        = ( qdot_m3_p_s + this->KSlip * pressure_drop_Pa ) / this->Disp;
        return (N_rev_p_s / this->Nmax);
    }

    Eigen::VectorXd calcNDPumpSpeedGradient(const double& pressure_drop_star, 
                                   const double& vol_flow_rate_star)
    {
        Eigen::VectorXd d_dip(2); // there are two inputs 
        d_dip(0) = (this->KSlip/this->Disp) * ( this->char_props.p_c / this->Nmax);
        d_dip(1) = (1.0 / this->Disp) * ( (this->char_props.mdot_c/ this->char_props.rho_c) / this->Nmax);
        return d_dip;
    }

    double calcNDMaxHeadAvailable(const double& vol_flow_rate_star)
    {   
        double qdot_m3_p_s      = vol_flow_rate_star * this->char_props.mdot_c / this->char_props.rho_c;
        double max_pressure_head = ( this->Disp * this->Nmax - qdot_m3_p_s ) / this->KSlip;

        return max_pressure_head / this->char_props.p_c;
    }

    double calcNDMaxHeadAvailableGradient(const double& vol_flow_rate_star)
    {
        double qdot_char = this->char_props.mdot_c / this->char_props.rho_c;
        return (-1.0 / this->KSlip) * (qdot_char / this->char_props.p_c);
    }

    double getNDPumpSpeedMin() {return (this->Nmin / this->Nmax);}
    double getNDPumpSpeedMax() {return 1.0;}
    double getNDPressureDropMax() {return (this->Dp_max/this->char_props.p_c);}
};

class ModifiedPumpConstraints {
private:
    CharacteristicProperties<double> char_props;

    // Pump Hardware Limits (Fixed geometric/speed limits)
    static constexpr double Nmax_rpm = 3450.0;
    static constexpr double Nmin_rpm = 500.0;
    
    // Reference pump parameters (The "Water" baseline from the datasheet)
    static constexpr double Ref_Disp  = 4.6e-6; // m^3/rev (Original GLHH21)
    static constexpr double Ref_KSlip = (2.904 / 76.0) * (1.0 / 60000.0) * (1.0 / 6894.76); 
    static constexpr double Ref_Mu    = 1.0e-3; // Reference viscosity (1 cP approx for water curve)

    // Member variables for the "Virtual" Pump
    double Disp;
    double KSlip;
    double Nmax;
    double Nmin;
    double Dp_max;

public:

    ModifiedPumpConstraints(const CharacteristicProperties<double> char_props_, double pump_mdot_target_ = 0.010) : char_props(char_props_) 
    {
        // 1. Convert Speed to rev/s
        this->Nmax = Nmax_rpm / 60.0;
        this->Nmin = Nmin_rpm / 60.0;

        // 2. Determine Required Displacement for 10 g/s Limit
        //    Q_max_vol = mdot_target / rho_fluid
        double mdot_target = pump_mdot_target_; // 10 g/s
        double rho_fluid   = char_props.rho_c; // Density of the fluid being simulated
        double Q_max_vol   = mdot_target / rho_fluid;

        //    Disp = Q_max_vol / N_max
        this->Disp = Q_max_vol / this->Nmax;

        // 3. Scale Slip for Viscosity
        //    Thinner fluid = More Slip = Higher KSlip
        double mu_fluid = char_props.mu_c; // Viscosity of fluid being simulated
        
        //    First, scale KSlip geometrically (smaller pump = less leak area)
        double geom_scale = this->Disp / Ref_Disp;
        
        //    Second, scale for viscosity (lower viscosity = more leak flow)
        double visc_scale = Ref_Mu / mu_fluid; 

        this->KSlip = Ref_KSlip * geom_scale * visc_scale;

        // 4. Set Pressure Limit (safe limit for micro-gears)
        this->Dp_max = 5.0 * 100000.0; // 5 Bar
    }

    double calcNDPumpSpeed(const double& pressure_drop_star, 
                           const double& vol_flow_rate_star) 
    {
        double pressure_drop_Pa = pressure_drop_star * this->char_props.p_c;
        double qdot_m3_p_s      = vol_flow_rate_star * this->char_props.mdot_c / this->char_props.rho_c;
        
        // Use the member variables calculated in constructor
        double N_rev_p_s        = ( qdot_m3_p_s + this->KSlip * pressure_drop_Pa ) / this->Disp;
        
        return (N_rev_p_s / this->Nmax);
    }

    Eigen::VectorXd calcNDPumpSpeedGradient(const double& pressure_drop_star, 
                                            const double& vol_flow_rate_star)
    {
        Eigen::VectorXd d_dip(2); 
        d_dip(0) = (this->KSlip / this->Disp) * ( this->char_props.p_c / this->Nmax);
        d_dip(1) = (1.0 / this->Disp) * ( (this->char_props.mdot_c/ this->char_props.rho_c) / this->Nmax);
        return d_dip;
    }

    double calcNDMaxHeadAvailable(const double& vol_flow_rate_star)
    {   
        double qdot_m3_p_s      = vol_flow_rate_star * this->char_props.mdot_c / this->char_props.rho_c;
        double max_pressure_head = ( this->Disp * this->Nmax - qdot_m3_p_s ) / this->KSlip;

        return max_pressure_head / this->char_props.p_c;
    }

    double calcNDMaxHeadAvailableGradient(const double& vol_flow_rate_star)
    {
        double qdot_char = this->char_props.mdot_c / this->char_props.rho_c;
        return (-1.0 / this->KSlip) * (qdot_char / this->char_props.p_c);
    }

    double getNDPumpSpeedMin() {return (this->Nmin / this->Nmax);}
    double getNDPumpSpeedMax() {return 1.0;}
    double getNDPressureDropMax() {return (this->Dp_max / this->char_props.p_c);}
};

#endif // PUMP_CURVE_HPP