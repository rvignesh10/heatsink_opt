import numpy as np
import heatsink_solver as hss

hss.init_mpi()
# general design params

ne = 1
nxi = 1000
kappa = 15.0
material = "stainless-steel"

L = 0.037
W = 0.031
H = 0.0095

D = 0.001

# heat load
Q = 15.0
A = L * W

# flow boundary conditions
p_out = 1.0e+05
T_in  = 273.15 + 20.0
ref = "water"

# thermal boundary conditions
Tsrc  = 273.15 + 80.0
Tsrc_bdr_id = 1

f_in_bcs = False
Tin = T_in
Tin_bdr_id = 5

flux_bdr_id = 1
qflux = Q / A

robin_bdr_id = 5
h = 0.2
Tinf = T_in

config_solver = {
    "design_params" : {
        "ne": ne,
        "nxi": nxi,
        "L_c": L,
        "kappa": kappa,
        "material": material
    },
    "mesh_params" : {
        "num_x": 51,
        "num_y": 51,
        "num_z": 51,
        "elem_type": "HEXAHEDRON",
        "x_domain": L,
        "y_domain": W,
        "z_domain": H
    },
    "fbcs" : {
        "p_out": p_out,
        "T_in": T_in,
        "refrigerant_name": ref
    },
    "tbcs" : {
        "Tsrc": Tsrc,
        "Tsrc_bdr_id": Tsrc_bdr_id,
        "f_in_bcs": f_in_bcs,
        "Tin": Tin,
        "Tin_bdr_id": Tin_bdr_id,
        "flux_bdr_id": flux_bdr_id,
        "qflux": qflux,
        "robin_bdr_id": robin_bdr_id,
        "h": h,
        "Tinf": Tinf
    },
    "pipe_params" : {
        "D" : D
    },
    "solve_mdot": False,
    "two_phase": False,
    "dirichlet": False
}


char_props = hss.getCharProps(ref, T_in, p_out, L)
solver = hss.HeatSinkSolver(config_solver)
num_states = solver.getTotalNumStates()

# pin_star = 1.2 * p_out / char_props["p_c"]
mdot_star= 1.0e-04 / char_props["mdot_c"]


xstar = np.zeros([nxi, ne])
ystar = np.zeros_like(xstar)
zstar = np.zeros_like(ystar) 

flow_loads = []

for e in range(ne):
    xstar[:, e] = np.linspace(0.0, L/char_props["L_c"], nxi)
    ystar[:, e] = (e+1) * ( (W/char_props["L_c"]) /(ne+1)) * np.ones(nxi)
    zstar[:, e] = (e+1) * ( (H/char_props["L_c"]) /(ne+1)) * np.ones(nxi)
    
    flow_ip = {
        "nxi": nxi,
        "f_heat_flux" : False,
        "xi_star" : xstar[:, e],
        "g_star"  : np.zeros(nxi),
        "load_star": np.zeros(nxi)
    }
    flow_loads.append(flow_ip)


vxyz_star = np.hstack( [xstar.flatten('F'), ystar.flatten('F'), zstar.flatten('F')] )

thermal_load = {
    "vxyz_star" : vxyz_star
}

# solver.set_flow_bcs(pin_star)
solver.set_flow_bcs(mdot_star)
solver.setInputs(flow_loads, thermal_load)
state = np.zeros(num_states)

solver.solveForState(state, False, True)


thermal_flux_star, norm_thermal_flux_star = solver.calcHeatSinkOutputs(state)
print(f"Thermal Flux: {-thermal_flux_star*char_props['flux_c']*1e-3:.4f} kW/m**2, Norm. Thermal Flux: {norm_thermal_flux_star:.5e}")

solver.SaveToParaview(state, "thermal", "flow")

hss.finalize_mpi()