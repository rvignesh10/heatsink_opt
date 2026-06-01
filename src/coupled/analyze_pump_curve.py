import numpy as np
import sys
sys.path.append("/Users/vigneshramakrishnan/Desktop/heatsink_opt/c++/")
from PumpCurvePy import ModifiedPumpConstraints
import heatsink_solver as hss
import matplotlib.pyplot as plt

if __name__ == "__main__":
    L_c   = 0.0254
    p_out = 0.7e+05
    T_in  = 273.15 + 20.0
    ref = "R11"
    
    char_props = hss.getCharProps(ref, T_in, p_out, L_c)
    
    print(char_props)
    pump_mdot_target = 0.050
    
    pump_cons = ModifiedPumpConstraints(char_props, pump_mdot_target)
    nmax_star = pump_cons.getNDPumpSpeedMax()
    
    mdot_star = np.arange(0.001, 0.050, 0.005) / char_props["mdot_c"]
    rho_star  = 1.0
    vol_star  = mdot_star/ rho_star 
    
    head_avail_star = np.zeros_like(vol_star)
    
    for idx, vols in enumerate(vol_star):
        head_avail_star[idx] = pump_cons.calc_max_head_available(vols, nmax_star)
    
    vol_rate = vol_star * char_props["mdot_c"] * 60000 / char_props["rho_c"] # in L/min
    head_avail = head_avail_star * char_props["p_c"] / 1.0e+05 # in bar
    
    print("Qdot_star", vol_star)
    print("Dp_star", head_avail_star)
    
    plt.plot(mdot_star*char_props["mdot_c"]*1000, head_avail)
    plt.show()
    
    
    
    
    