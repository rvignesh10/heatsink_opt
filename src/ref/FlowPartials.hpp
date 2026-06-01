#ifndef FLOW_PARTIALS_HPP
#define FLOW_PARTIALS_HPP

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

#endif 