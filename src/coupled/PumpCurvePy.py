import numpy as np

class PumpConstraints:
    # Static constants (mirroring static constexpr)
    # KSlip units: (m^3/s) / Pa
    KSlip = (2.904 / 76.0) * (1.0 / 60000.0) * (1.0 / 6894.76) 
    Disp  = 4.6e-06          # m^3/rev
    Nmax  = 3450.0 / 60.0    # rev / s
    Nmin  = 500.0 / 60.0     # rev / s
    Dp_max = 125.0 * 6894.76 # Pa

    def __init__(self, char_props):
        """
        Initialize with characteristic properties object.
        Expected attributes in char_props: p_c, mdot_c, rho_c
        """
        self.char_props = char_props

    def calcNDPumpSpeed(self, pressure_drop_star, vol_flow_rate_star):
        """
        Calculates the Non-Dimensional Pump Speed (N*).
        """
        pressure_drop_Pa = pressure_drop_star * self.char_props["p_c"]
        qdot_m3_p_s      = vol_flow_rate_star * self.char_props["mdot_c"] / self.char_props["rho_c"]
        
        # Pump equation: N = (Q + K_slip * dP) / Disp
        N_rev_p_s        = ( qdot_m3_p_s + self.KSlip * pressure_drop_Pa ) / self.Disp
        
        return (N_rev_p_s / self.Nmax)

    def calcNDPumpSpeedGradient(self, pressure_drop_star, vol_flow_rate_star):
        """
        Calculates the gradient of N* w.r.t [pressure_drop_star, vol_flow_rate_star].
        Returns a numpy array of size 2.
        """
        d_dip = np.zeros(2)
        
        # Partial derivative w.r.t pressure_drop_star
        d_dip[0] = (self.KSlip / self.Disp) * (self.char_props["p_c"] / self.Nmax)
        
        # Partial derivative w.r.t vol_flow_rate_star
        d_dip[1] = (1.0 / self.Disp) * ((self.char_props["mdot_c"] / self.char_props["rho_c"]) / self.Nmax)
        
        return d_dip

    def getNDPumpSpeedMin(self):
        return (self.Nmin / self.Nmax)

    def getNDPumpSpeedMax(self):
        return 1.0

    def getNDPressureDropMax(self):
        return (self.Dp_max / self.char_props["p_c"])


class ModifiedPumpConstraints:
    # Pump Hardware Limits (Fixed geometric/speed limits)
    NMAX_RPM = 3450.0
    NMIN_RPM = 500.0
    
    # Reference pump parameters (The "Water" baseline from the datasheet)
    REF_DISP  = 4.6e-6 # m^3/rev (Original GLHH21)
    REF_KSLIP = (2.904 / 76.0) * (1.0 / 60000.0) * (1.0 / 6894.76)
    REF_MU    = 1.0e-3 # Reference viscosity (1 cP approx for water curve)

    def __init__(self, char_props, pump_mdot_target: float=0.010):
        """
        Initialize with characteristic properties.
        Expected char_props attributes: rho_c, mu_c, p_c, mdot_c
        """
        self.char_props = char_props

        # 1. Convert Speed to rev/s
        self.n_max = self.NMAX_RPM / 60.0
        self.n_min = self.NMIN_RPM / 60.0

        # 2. Determine Required Displacement for 10 g/s Limit
        #    Q_max_vol = mdot_target / rho_fluid
        mdot_target = pump_mdot_target # 10 g/s
        rho_fluid   = char_props["rho_c"] # Density of the fluid being simulated
        q_max_vol   = mdot_target / rho_fluid

        #    Disp = Q_max_vol / N_max
        self.disp = q_max_vol / self.n_max

        # 3. Scale Slip for Viscosity
        #    Thinner fluid = More Slip = Higher KSlip
        mu_fluid = char_props["mu_c"] # Viscosity of fluid being simulated
        
        #    First, scale KSlip geometrically (smaller pump = less leak area)
        geom_scale = self.disp / self.REF_DISP
        
        #    Second, scale for viscosity (lower viscosity = more leak flow)
        visc_scale = self.REF_MU / mu_fluid 

        self.k_slip = self.REF_KSLIP * geom_scale * visc_scale

        # 4. Set Pressure Limit (safe limit for micro-gears)
        self.dp_max = 5.0 * 100000.0 # 5 Bar

    def calc_nd_pump_speed(self, pressure_drop_star: float, vol_flow_rate_star: float) -> float:
        """
        Calculates non-dimensional pump speed.
        """
        pressure_drop_pa = pressure_drop_star * self.char_props["p_c"]
        qdot_m3_p_s      = vol_flow_rate_star * self.char_props["mdot_c"] / self.char_props["rho_c"]
        
        # Use the member variables calculated in constructor
        n_rev_p_s = (qdot_m3_p_s + self.k_slip * pressure_drop_pa) / self.disp
        
        return n_rev_p_s / self.n_max

    def calc_nd_pump_speed_gradient(self, pressure_drop_star: float, vol_flow_rate_star: float) -> np.ndarray:
        """
        Returns the gradient as a numpy array [d/d_pressure, d/d_flow].
        """
        d_dip = np.zeros(2)
        
        # Derivative w.r.t Pressure Drop
        d_dip[0] = (self.k_slip / self.disp) * (self.char_props["p_c"] / self.n_max)
        
        # Derivative w.r.t Volume Flow Rate
        d_dip[1] = (1.0 / self.disp) * ((self.char_props["mdot_c"] / self.char_props["rho_c"]) / self.n_max)
        
        return d_dip

    def calc_nd_vol_flow_rate(self, pressure_drop_star: float, nd_speed: float):
        speed = nd_speed * self.n_max
        pressure_drop_pa = pressure_drop_star * self.char_props["p_c"]
        qdot_m3_p_s = self.disp * speed - self.k_slip * pressure_drop_pa
        return qdot_m3_p_s * self.char_props["rho_c"] / self.char_props["mdot_c"]
    
    def calc_max_head_available(self, vol_flow_rate_star: float, nd_speed: float) -> float:
        speed = nd_speed * self.n_max
        qdot_m3_p_s      = vol_flow_rate_star * self.char_props["mdot_c"] / self.char_props["rho_c"]
        
        head_avail =  (self.disp * speed - qdot_m3_p_s) / self.k_slip
        return head_avail / self.char_props["p_c"]
    
    def getNDPumpSpeedMin(self) -> float:
        return self.n_min / self.n_max

    def getNDPumpSpeedMax(self) -> float:
        return 1.0

    def getNDPressureDropMax(self) -> float:
        return self.dp_max / self.char_props["p_c"]