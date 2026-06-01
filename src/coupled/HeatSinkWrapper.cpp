#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "HeatSinkSolver.hpp"

namespace py = pybind11;

// -------------------------------------------------------------------------
// Helper Functions: Unpack Python Dictionary -> C++ Structs
// -------------------------------------------------------------------------

DesignParameters genDesignParams(const py::dict& des_params) {
    DesignParameters params;
    if (des_params.contains("ne")) params.ne = des_params["ne"].cast<int>();
    if (des_params.contains("nxi"))params.nxi= des_params["nxi"].cast<int>();
    if (des_params.contains("kappa")) params.kappa = des_params["kappa"].cast<double>();
    if (des_params.contains("material")) params.material = des_params["material"].cast<std::string>();
    return params;
}

MeshParameters genMeshParams(const py::dict& d) {
    MeshParameters params;
    params.elem_type = (d.contains("elem_type")) ? d["elem_type"].cast<std::string>() : "HEXAHEDRON";
    params.num_x     = (d.contains("num_x")) ? d["num_x"].cast<int>() : (int)(30);
    params.num_y     = (d.contains("num_y")) ? d["num_y"].cast<int>() : (int)(30);
    params.num_z     = (d.contains("num_z")) ? d["num_z"].cast<int>() : (int)(30);
    params.x_domain  = (d.contains("x_domain")) ? d["x_domain"].cast<double>() : (double)(1.0);
    params.y_domain  = (d.contains("y_domain")) ? d["y_domain"].cast<double>() : (double)(1.0);
    params.z_domain  = (d.contains("z_domain")) ? d["z_domain"].cast<double>() : (double)(1.0);
    return params;
}

FlowBoundaryConditions<double> 
genFlowBCs(const py::dict& d, const CharacteristicProperties<double>& char_props,
          HomogeneousRefrigerant<double>* ref)  
{
    FlowBoundaryConditions<double> fbcs;
    fbcs.refrigerant_name = d["refrigerant_name"].cast<std::string>();

    fbcs.pout_star = d["p_out"].cast<double>() / char_props.p_c;
    fbcs.Tin_star  = d["T_in"].cast<double>() / char_props.T_c;
    fbcs.hin_star  = ref->calc_satL_enthalpy(d["T_in"].cast<double>()) / char_props.h_c;

    fbcs.mdot_star = (d.contains("mdot")) ? d["mdot"].cast<double>() / char_props.mdot_c : 0.010 / char_props.mdot_c;
    fbcs.pin_star  = (d.contains("p_in")) ? d["p_in"].cast<double>() / char_props.p_c : 1.5 * fbcs.pout_star;
    return fbcs;
}

ThermalBoundaryConditions 
genThermalBCs(const py::dict& d, const CharacteristicProperties<double>& char_props)  
{
    ThermalBoundaryConditions tbcs;
    tbcs.Tsrc_star = d["Tsrc"].cast<double>() / char_props.T_c;
    tbcs.Tsrc_bdr_id = d["Tsrc_bdr_id"].cast<int>();
    
    tbcs.f_in_bcs = d["f_in_bcs"].cast<bool>();
    tbcs.Tin_star = d["Tin"].cast<double>() / char_props.T_c;
    tbcs.Tin_bdr_id = d["Tin_bdr_id"].cast<int>();

    tbcs.flux_bdr_id = d["flux_bdr_id"].cast<int>();
    tbcs.qflux_star  = d["qflux"].cast<double>() / char_props.flux_c;

    tbcs.robin_bdr_id = d["robin_bdr_id"].cast<int>();
    tbcs.h_star = d["h"].cast<double>() / char_props.h_c;
    tbcs.Tinf_star = d["Tinf"].cast<double>() / char_props.T_c;
    return tbcs;
}

PipeParameters<double> genPipeParams(const py::dict& d) {
    PipeParameters<double> params;
    params.D = d["D"].cast<double>();
    return params;
}


// -------------------------------------------------------------------------
// Main Wrapper
// -------------------------------------------------------------------------

PYBIND11_MODULE(heatsink_solver, m) {
    m.doc() = "Pybind11 wrapper for HeatSinkSolver with Dictionary Initialization";
    // --- ADD THIS FUNCTION ---
    m.def("init_mpi", []() {
        int flag;
        MPI_Initialized(&flag);
        if (!flag) {
            MPI_Init(NULL, NULL);
        }
    }, "Initialize the MPI environment (required for MFEM)");

    // --- ADD THIS FUNCTION (Optional but good practice) ---
    m.def("finalize_mpi", []() {
        int flag;
        MPI_Finalized(&flag);
        if (!flag) {
            MPI_Finalize();
        }
    }, "Finalize the MPI environment");

    m.def("getCharProps",
        [](std::string ref_name, double T_in, double p_out, double L_c){
            auto ref = createRefrigerant<double>(ref_name);
            auto cp  = ref->calc_characteristic_properties(T_in, p_out, L_c);
            py::dict d;
            d["p_c"]    = cp.p_c;
            d["T_c"]    = cp.T_c;
            d["L_c"]    = cp.L_c;
            d["rho_c"]  = cp.rho_c;
            d["mu_c"]   = cp.mu_c;
            d["h_c"]    = cp.h_c;
            d["u_c"]    = cp.u_c;
            d["g_c"]    = cp.g_c;
            d["Phi_c"]  = cp.Phi_c;
            d["kappa_c"]= cp.kappa_c;
            d["htc_c"]  = cp.htc_c;
            d["flux_c"] = cp.flux_c;
            d["mdot_c"] = cp.mdot_c;
            d["Cp_c"]   = cp.Cp_c;
            return d;
        },
        py::arg("ref_name"),
        py::arg("T_in"),
        py::arg("p_out"),
        py::arg("L_c"),
        "returns characteristic properties of refrigerant"
    );

    py::class_<HeatSinkSolver>(m, "HeatSinkSolver")
        
        // ---------------------------------------------------------------------
        // Dictionary Factory Constructor
        // ---------------------------------------------------------------------
        .def(py::init([](py::dict config) {
                     
            // 1. Extract Key Values for Characteristic Scales
            // ---------------------------------------------
            // We need to carefully unpack the sub-dictionaries
            py::dict fbcs_dict = config["fbcs"].cast<py::dict>();
            py::dict des_dict  = config["design_params"].cast<py::dict>();
            
            // 2. Setup Refrigerant for Property Calculation
            // ---------------------------------------------   
            std::string ref_name = (fbcs_dict.contains("refrigerant_name")) ? fbcs_dict["refrigerant_name"].cast<std::string>() : "R11";
            
            // Assuming createRefrigerant returns a unique_ptr
            auto ref = createRefrigerant<double>(ref_name); 

            double p_out = fbcs_dict["p_out"].cast<double>();
            double T_in  = fbcs_dict["T_in"].cast<double>();
            
            // Assuming L_c exists
            double L_c   = des_dict.contains("L_c") ? des_dict["L_c"].cast<double>() : 1.0; 

            // 3. Calculate Properties
            // ---------------------------------------------
            auto sat_props = ref->calc_saturation_properties(p_out);
            auto char_props = ref->calc_characteristic_properties(T_in, p_out, L_c);

            // 4. Generate Structs using Helpers
            // ---------------------------------------------
            DesignParameters des_params = genDesignParams(des_dict);
            
            // Check if keys exist before casting sub-dicts
            py::dict mesh_dict = config.contains("mesh_params") ? config["mesh_params"].cast<py::dict>() : py::dict();
            MeshParameters mesh_params  = genMeshParams(mesh_dict);
            
            py::dict tbc_dict = config.contains("tbcs") ? config["tbcs"].cast<py::dict>() : py::dict();
            ThermalBoundaryConditions tbcs = genThermalBCs(tbc_dict, char_props);
            
            FlowBoundaryConditions<double> fbcs = genFlowBCs(fbcs_dict, char_props, ref.get());

            py::dict pipe_dict = config.contains("pipe_params") ? config["pipe_params"].cast<py::dict>() : py::dict();
            PipeParameters<double> pipe_params = genPipeParams(pipe_dict);

            // 5. Extract Solver Flags
            // ---------------------------------------------
            bool solve_mdot = config.contains("solve_mdot") ? config["solve_mdot"].cast<bool>() : false;
            bool two_phase  = config.contains("two_phase")  ? config["two_phase"].cast<bool>()  : false;
            bool dirichlet  = config.contains("dirichlet")  ? config["dirichlet"].cast<bool>()  : false;
            double pump_mdot_target = config.contains("pump_mdot_target") ? config["pump_mdot_target"].cast<double>() : 0.010; 

            // 6. Instantiate HeatSinkSolver
            // ---------------------------------------------
            return std::make_unique<HeatSinkSolver>(
                des_params,
                mesh_params,
                tbcs,
                fbcs,
                sat_props,
                char_props,
                pipe_params,
                solve_mdot,
                two_phase,
                dirichlet,
                pump_mdot_target
            );
        }), py::arg("config"), "Initialize HeatSinkSolver using a nested Python dictionary")

        .def("getTotalNumStates", 
            &HeatSinkSolver::getTotalNumStates,
            "returns the total number of state variables"
        )
        .def("getCharProps", 
            [](const HeatSinkSolver& self) {
                // 1. Get the C++ struct
                const CharacteristicProperties<double> cp = self.getCharProps();
                
                // 2. Create and populate Python dictionary
                py::dict d;
                d["p_c"]    = cp.p_c;
                d["T_c"]    = cp.T_c;
                d["L_c"]    = cp.L_c;
                d["rho_c"]  = cp.rho_c;
                d["mu_c"]   = cp.mu_c;
                d["h_c"]    = cp.h_c;
                d["u_c"]    = cp.u_c;
                d["g_c"]    = cp.g_c;
                d["Phi_c"]  = cp.Phi_c;
                d["kappa_c"]= cp.kappa_c;
                d["htc_c"]  = cp.htc_c;
                d["flux_c"] = cp.flux_c;
                d["mdot_c"] = cp.mdot_c;
                d["Cp_c"]   = cp.Cp_c;
                return d;
            }, 
            "Returns the characteristic properties as a dictionary"
        )
        // ---------------------------------------------------------------------
        // input setters 
        // ---------------------------------------------------------------------
        .def("set_thermal_bcs",
            &HeatSinkSolver::set_thermal_bcs,
            py::arg("source_bcs_star"),
            "sets the thermal source boundary condition - either dirichlet temp or heat flux value"
        )
        .def("set_flow_bcs", 
            &HeatSinkSolver::set_flow_bcs, 
            py::arg("flow_bcs_star"),
            "sets the flow solver boundary condition - either inlet pressure des var or mass-flow/channel des var"
        )
        .def("setInputs", 
            [](HeatSinkSolver& self, py::list flow_list_py, py::dict thermal_dict_py) 
            {
            // 1. Convert Python List[Dict] -> C++ std::vector<FlowLoadParameters>
            std::vector<FlowLoadParameters<double>> flow_load_list;
            flow_load_list.reserve(flow_list_py.size());

            for (auto handle : flow_list_py) {
                py::dict d = handle.cast<py::dict>();
                FlowLoadParameters<double> params;
                
                // Extract Eigen vectors from dict (automatically handles numpy arrays)
                params.f_heat_flux = d["f_heat_flux"].cast<bool>();
                params.nxi         = d["nxi"].cast<int>();
                if (d.contains("xi_star"))   params.xi_star   = d["xi_star"].cast<Eigen::VectorXd>();
                if (d.contains("g_star")) params.g_star       = d["g_star"].cast<Eigen::VectorXd>();
                if (d.contains("load_star")) params.load_star = d["load_star"].cast<Eigen::VectorXd>();
                
                flow_load_list.push_back(params);
            }

            // 2. Convert Python Dict -> C++ LoadParameters (Thermal)
            LoadParameters thermal_load;
            if (thermal_dict_py.contains("vxyz_star")) 
                thermal_load.vxyz_star = thermal_dict_py["vxyz_star"].cast<Eigen::VectorXd>();
            
            if (thermal_dict_py.contains("qblf_star")) 
                thermal_load.qblf_star = thermal_dict_py["qblf_star"].cast<Eigen::VectorXd>();
                
            if (thermal_dict_py.contains("qlf_star"))  
                thermal_load.qlf_star  = thermal_dict_py["qlf_star"].cast<Eigen::VectorXd>();

            // 3. Call the actual C++ method
            self.setInputs(flow_load_list, thermal_load);

            }, 
            py::arg("flow_loads"), py::arg("thermal_load"), 
            "Set inputs using a list of flow dictionaries and one thermal dictionary"
        )
        // ---------------------------------------------------------------------
        // solver functions
        // ---------------------------------------------------------------------
        .def("solveForState", 
            &HeatSinkSolver::solveForState, 
            py::arg("state"),
            py::arg("use_eigen"),
            py::arg("prec_solve"),
            "solves the forward problem R(u)=0"
        )
        .def("solveForAdjoint",
            &HeatSinkSolver::solveForAdjoint,
            py::arg("state"),
            py::arg("dJdstate"),
            py::arg("adjoint"),
            py::arg("use_eigen_solver"),
            "Solves the adjoint equation (dRdu)^T * lambda = -dJdu"
        )
        .def("calcHeatSinkTotalGradient",
            &HeatSinkSolver::calcHeatSinkTotalGradient,
            py::arg("state"),
            py::arg("adjoint"),
            py::arg("flow_bcs_star_bar"),
            py::arg("vxyz_star_bar"),
            py::arg("xi_star_bar"),
            py::arg("g_star_bar"),
            py::arg("J_bar"),
            "returns the total gradient of the Lagrangian L(u, lambda, alpha), dL_dalpha = J_bar + lambda^T * R_bar"
        )
        .def("calcHeatSinkTotalGradientMulti",
            &HeatSinkSolver::calcHeatSinkTotalGradientMulti,
            py::arg("state"),
            py::arg("adj_matrix"),        // Changed: Now expects a Matrix [n_states x n_outputs]
            py::arg("flow_bcs_star_bar"),
            py::arg("vxyz_star_bar"),
            py::arg("xi_star_bar"),
            py::arg("g_star_bar"),
            py::arg("J_bar_vec"),         // Changed: Now expects a Vector [n_outputs]
            py::arg("total_gradients"),   // Added: Output Vector [n_outputs] (Modified in-place)
            "Computes total gradients for multiple outputs simultaneously.\n"
            "Arguments:\n"
            "  state: System state vector\n"
            "  adj_matrix: Matrix of adjoint vectors (cols=outputs)\n"
            "  flow_bcs_star_bar: Scalar sensitivity\n"
            "  vxyz_star_bar: Geometry sensitivity vector\n"
            "  xi_star_bar: Arc-length sensitivity vector\n"
            "  g_star_bar: Gravity sensitivity vector\n"
            "  J_bar_vec: Vector of direct partial derivatives for each output\n"
            "  total_gradients: Pre-allocated vector to store results (written in-place)"
        )
        .def("SaveToParaview",
            &HeatSinkSolver::SaveToParaview,
            py::arg("state"),
            py::arg("tcollection"),
            py::arg("fcollection"),
            "saves the solution state as a .pvd file that can be viewed through ParaView"
        )
        //---------------------------------------------------------------------------
        //-------------------- Heat Sink Outputs ------------------------------------
        //---------------------------------------------------------------------------
        .def("calcHeatSinkOutputs",
            &HeatSinkSolver::calcHeatSinkOutputs,
            py::arg("state"),
            "returns non-dimensional thermal-flux and normalized thermal-flux as a tuple"
        )
        .def("calcHeatSinkOutputsJacobian",
            &HeatSinkSolver::calcHeatSinkOutputsJacobian,
            py::arg("state"),
            py::arg("dflux_star_dstate").noconvert(),
            py::arg("dnormFlux_star_dstate").noconvert(),
            "sets the jacobian of the outputs w.r.t heat sink state variables"
        )
        .def("diffHeatSinkOutputs",
            &HeatSinkSolver::diffHeatSinkOutputs,
            py::arg("state"),
            py::arg("flow_bcs_star_bar"),
            "returns the partials of outputs w.r.t design variables as a tuple"
        )
        //---------------------------------------------------------------------------
        //--------------- constraint 1 - Tagg ---------------------------------------
        //---------------------------------------------------------------------------
        .def("calcAggregatedSourceTemperature",
            &HeatSinkSolver::calcAggregatedSourceTemperature,
            py::arg("state"),
            py::arg("boundary_att_id"),
            "returns aggregated temperature at a given boundary attribute"
        )
        .def("calcAggregateSourceTemperatureJacobian",
            &HeatSinkSolver::calcAggregateSourceTemperatureJacobian,
            py::arg("state"),
            py::arg("dTagg_star_dstate").noconvert(),
            py::arg("boundary_att_id"),
            "calculates the jacobian of Tagg w.r.t state variables"
        )
        .def("diffAggregateSourceTemperature",
            &HeatSinkSolver::diffAggregateSourceTemperature,
            "returns 0.0 as Tagg is not dependent on design variables"
        )
        //---------------------------------------------------------------------------
        //--------------- constraint 2 - xagg ---------------------------------------
        //---------------------------------------------------------------------------
        .def("calcAggregateOutletVaporQuality",
            &HeatSinkSolver::calcAggregateOutletVaporQuality,
            py::arg("state"),
            "returns aggregated vapor quality at the outlet of the heat sink"
        )
        .def("calcAggregateOutletVaporQualityJacobian",
            &HeatSinkSolver::calcAggregateOutletVaporQualityJacobian,
            py::arg("state"),
            py::arg("dxagg_dstate").noconvert(),
            "calculates the jacobian of xagg w.r.t heat sink states"
        )
        .def("diffAggregateOutletVaporQuality",
            &HeatSinkSolver::diffAggregateOutletVaporQuality,
            "returns 0.0 as aggregated quality is not a function of design variables"
        )
        //---------------------------------------------------------------------------
        //--------------- constraint 3 - mod_Rth ------------------------------------
        //---------------------------------------------------------------------------
        .def("calcModifiedThermalResistance",
            &HeatSinkSolver::calcModifiedThermalResistance,
            py::arg("state"),
            py::arg("Tsrc_star"),
            py::arg("alpha"),
            py::arg("beta"),
            py::arg("boundary_att_id"),
            "returns (Tagg - Tin) * ( Qdot * delta_p )^gamma : as the modified thermal resistance"
        )
        .def("calcModifiedThermalResistanceJacobian",
            &HeatSinkSolver::calcModifiedThermalResistanceJacobian,
            py::arg("state"),
            py::arg("dmtr_star_dstate").noconvert(),
            py::arg("Tsrc_star"),
            py::arg("alpha"),
            py::arg("beta"),
            py::arg("boundary_att_id"),
            "calculates the jacobian of the modified thermal resistance w.r.t state variables"
        )
        .def("diffModifiedThermalResistance",
            &HeatSinkSolver::diffModifiedThermalResistance,
            py::arg("state"),
            py::arg("Tsrc_star"),
            py::arg("flow_bcs_star_bar"),
            py::arg("alpha"),
            py::arg("beta"),
            py::arg("boundary_att_id"),
            "calculates the total derivative of mod_Rth w.r.t alpha_p"
        )
        //---------------------------------------------------------------------------
        //--------------- safe source temp calc -------------------------------------
        //---------------------------------------------------------------------------
        .def("calcSafeSourceTemperature",
            &HeatSinkSolver::calcSafeSourceTemperature,
            py::arg("state"),
            py::arg("Tsrc_star"),
            py::arg("alpha"),
            py::arg("boundary_att_id"),
            "calculates a combination of temp uniformity and temp target"
        )
        .def("calcSafeSourceTemperatureJacobian",
            &HeatSinkSolver::calcSafeSourceTemperatureJacobian,
            py::arg("state"),
            py::arg("dJ_dstate").noconvert(),
            py::arg("Tsrc_star"),
            py::arg("alpha"),
            py::arg("boundary_att_id"),
            "calculates the jacobian of safe source temperature obj w.r.t state variables"
        )
        .def("diffSafeSourceTemperature",
            &HeatSinkSolver::diffSafeSourceTemperature,
            py::arg("state"),
            py::arg("Tsrc_star"),
            py::arg("flow_bcs_star_bar"),
            py::arg("alpha"),
            py::arg("boundary_att_id"),
            "returns 0.0 as its not a function of design variables at all"
        )
        //---------------------------------------------------------------------------
        //--------------- constraint 4 - Nstar --------------------------------------
        //---------------------------------------------------------------------------
        .def("calcNDPumpSpeed",
            &HeatSinkSolver::calcNDPumpSpeed,
            py::arg("state"),
            "returns the non-dimensional rpm of the pump for a specified pressure-drop and vol-flow rate through the heat sink"
        )
        .def("calcNDPumpSpeedJacobian",
            &HeatSinkSolver::calcNDPumpSpeedJacobian,
            py::arg("state"),
            py::arg("dNstar_dstate").noconvert(),
            "calculates the jacobian of non-dimensional pump speed w.r.t heat sink states"
        )
        .def("diffNDPumpSpeed",
            &HeatSinkSolver::diffNDPumpSpeed,
            py::arg("state"),
            py::arg("flow_bcs_star_bar"),
            "returns the partial gradient of non-dimensional pump speed w.r.t des variables"
        )
        //---------------------------------------------------------------------------
        //------------------------------ PumpingPower -------------------------------
        //---------------------------------------------------------------------------
        .def("calcNDPumpingPower",
            &HeatSinkSolver::calcNDPumpingPower,
            py::arg("state"),
            "returns the non-dimensional pumping power for a specified pressure-drop and vol-flow rate through the heat sink"
        )
        .def("calcNDPumpingPowerJacobian",
            &HeatSinkSolver::calcNDPumpingPowerJacobian,
            py::arg("state"),
            py::arg("dpower_dstate").noconvert(),
            "calculates the jacobian of non-dimensional pump power w.r.t heat sink states"
        )
        .def("diffNDPumpingPower",
            &HeatSinkSolver::diffNDPumpingPower,
            py::arg("state"),
            py::arg("flow_bcs_star_bar"),
            "returns partial gradient of non-dimensional pumping power w.r.t des variables"
        )
        //---------------------------------------------------------------------------
        //--------------- constraint 5 - PressureDrop -------------------------------
        //---------------------------------------------------------------------------
        .def("calcPressureDropConstraint",
            &HeatSinkSolver::calcPressureDropConstraint,
            py::arg("state"),
            "return the pressure-drop-heatsink - pressure-drop-available"
        )
        .def("calcPressureDropConstraintJacobian",
            &HeatSinkSolver::calcPressureDropConstraintJacobian,
            py::arg("state"),
            py::arg("dpdcons_dstate").noconvert(),
            "calculates the jacobian of non-dimensional pressure-drop-constraint w.r.t heat sink states"
        )
        .def("diffPressureDropConstraint",
            &HeatSinkSolver::diffPressureDropConstraint,
            py::arg("state"),
            py::arg("flow_bcs_star_bar"),
            "returns the gradient of pressure-drop-constraint w.r.t des variables"
        );
}