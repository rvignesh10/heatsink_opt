#ifndef HEATSINK_SOLVER_HPP
#define HEATSINK_SOLVER_HPP

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "mfem.hpp"
#include "../flow/FlowSolver.hpp"
#include "../thermal/ThermalSolver.hpp"
#include "../utils/NewtonSolver.hpp"
#include "../utils/LinearAlgebra.hpp"
#include "../utils/InexactNewton.hpp"
#include "../utils/utils.hpp"
#include "StateView.hpp"
#include "JacobianView.hpp"
#include "PumpCurve.hpp"
#include <span>


using Triplet = Eigen::Triplet<double>;
using TripletList = std::vector<Triplet>;
using ListOf_TripletList = std::vector<TripletList>;

class HeatSinkSolver {
private:

    Eigen::VectorXd vxyz_star;
    std::vector<FlowLoadParameters<double>> flow_load_list;
    LoadParameters thermal_load;

    std::vector<std::unique_ptr<FlowSolverBase<double>>> FlowSolverList;
    std::unique_ptr<BaseThermalSolver> ThermalSolver;
    // std::unique_ptr<PumpConstraints> PumpCons;
    std::unique_ptr<ModifiedPumpConstraints> PumpCons;
    DesignParameters design_params_;
    CharacteristicProperties<double> char_props_;

    bool solve_for_mdot;
    bool two_phase_;
    bool dirichlet_;

    static constexpr bool flow_reorder = true;
    double flow_bcs_star_;

    static constexpr double rho_agg = 10.0;

    std::unique_ptr<mfem::HypreBoomerAMG> real_amg;
    std::unique_ptr<mfem::FrozenPreconditioner> frozen_amg;
    std::unique_ptr<mfem::GMRESSolver> gmres;
    std::unique_ptr<InexactNewton> newton;
    // std::unique_ptr<mfem::NewtonSolver> newton;

    std::unique_ptr<mfem::HypreBoomerAMG> real_amg_adj;
    std::unique_ptr<mfem::FrozenPreconditioner> frozen_amg_adj;
    std::unique_ptr<mfem::GMRESSolver> gmres_adj;

    std::unique_ptr<mfem::HypreParMatrix> adj_matrix;
    std::unique_ptr<mfem::HypreParMatrix> prec_matrix;

public:
    HeatSinkSolver(
                  const DesignParameters& design_params, 
                  const MeshParameters& mesh_params, 
                  const ThermalBoundaryConditions& tbc, 
                  const FlowBoundaryConditions<double>& fbc, 
                  const SaturationProperties<double>& sat_props_exit,
                  const CharacteristicProperties<double>& char_props,
                  const PipeParameters<double>& pipe_params,
                  bool solve_mdot = false, 
                  bool two_phase = false,
                  bool dirichlet = false,
                  double pump_mdot_target = 0.010
                ) : design_params_(design_params), 
                char_props_(char_props),
                solve_for_mdot(solve_mdot),
                two_phase_(two_phase),
                dirichlet_(dirichlet)
    {
        if (this->dirichlet_) {
            ThermalSolver = std::make_unique<DirichletThermalSolver>(design_params_, mesh_params, tbc, char_props);
        }
        else {
            ThermalSolver = std::make_unique<RobinFluxThermalSolver>(design_params_, mesh_params, tbc, char_props);
        }
        
        flow_load_list.reserve(design_params_.ne);
        FlowSolverList.resize(design_params_.ne);
        for (int i=0; i<design_params_.ne; ++i) {
            if (this->two_phase_){
                FlowSolverList[i] = std::make_unique<FlowSolverForwardEulerTP<double>>(fbc, pipe_params, char_props, sat_props_exit, solve_for_mdot);
            }
            else {
                FlowSolverList[i] = std::make_unique<FlowSolverForwardEulerSP<double>>(fbc, pipe_params, char_props, sat_props_exit, solve_for_mdot);
            }
            
        }
        // PumpCons = std::make_unique<PumpConstraints>(char_props_);
        PumpCons = std::make_unique<ModifiedPumpConstraints>(char_props_, pump_mdot_target);
        this->initSolvers();
    }

    void initSolvers() {
        this->gmres = std::make_unique<mfem::GMRESSolver>(hypre_MPI_COMM_WORLD);
        this->gmres->SetKDim(50);
        this->gmres->SetMaxIter(100);
        this->gmres->SetRelTol(1e-08);

        this->newton = std::make_unique<InexactNewton>();
        // this->newton = std::make_unique<mfem::NewtonSolver>(hypre_MPI_COMM_WORLD);
        this->newton->SetSolver(*gmres.get());
        this->newton->SetPrintLevel(1);
        this->newton->SetRelTol(1.0e-06);
        this->newton->SetAbsTol(1.0e-10);
        this->newton->SetMaxIter(10);

        this->gmres_adj = std::make_unique<mfem::GMRESSolver>(hypre_MPI_COMM_WORLD);
        this->gmres_adj->SetKDim(50);
        this->gmres_adj->SetMaxIter(100);
        this->gmres_adj->SetAbsTol(1e-12);
    }

    ~HeatSinkSolver() {}

    const int getTotalNumStates() {
        const int ne = design_params_.ne;
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();
        return (ne * (nf * nxi) + nhs);
    }

    const CharacteristicProperties<double> getCharProps() const {return char_props_;}

    void set_thermal_bcs(const double source_bcs_star) {
        this->ThermalSolver->updateThermalBCs(source_bcs_star);
    }

    void set_flow_bcs(double flow_bcs_star) {
        flow_bcs_star_ = flow_bcs_star;
    }

    void setInputs(const std::vector<FlowLoadParameters<double>>& flow_load_list_, 
                  const LoadParameters& thermal_load_) {
        assert(flow_load_list_.size() == design_params_.ne);
        for (int e=0; e<design_params_.ne; ++e) assert(flow_load_list_[e].xi_star.size() == design_params_.nxi);
        assert(thermal_load_.vxyz_star.size() == 3 * design_params_.nxi * design_params_.ne);
        flow_load_list = flow_load_list_;
        thermal_load = thermal_load_;
    }

    double calcHeatSinkResidual(const Eigen::Ref<const Eigen::VectorXd>& state, Eigen::Ref<Eigen::VectorXd> res) {
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();
        const int block_size = nf * nxi;
        const int input_size = ne*nxi;

        auto sview = make_state_view(state, ne, nf, nxi, nhs);
        auto rset  = make_state_view(res, ne, nf, nxi, nhs);


        auto [thermal_flux_star, Twall_star] = ThermalSolver->calcThermalOutputs(sview.Ths());
        assert(Twall_star.size() == input_size);

        // 1. calculating and assembling Flow residuals
        Eigen::VectorXd qblf_star(input_size);
        Eigen::VectorXd qlf_star(input_size);
        for (int e=0; e<ne; ++e) {

            flow_load_list[e].load_star = Twall_star.segment(e*nxi, nxi);
            FlowSolverList[e]->set_bcs(this->flow_bcs_star_);
            FlowSolverList[e]->setInputs(flow_load_list[e]);
            FlowSolverList[e]->calcFlowResidual(sview.fstate_mat(e), res.segment(e*block_size, block_size), flow_reorder);

            FlowSolverList[e]->calcFlowOutputs(sview.fstate_mat(e), 
                                               qblf_star.segment(e*nxi, nxi),
                                               qlf_star.segment(e*nxi, nxi));

        }

        // 2. calculating and assembling Thermal residuals
        thermal_load.qblf_star = qblf_star;
        thermal_load.qlf_star  = qlf_star;
        ThermalSolver->setInputs(thermal_load);

        ThermalSolver->calcThermalResidual(sview.Ths(), rset.Ths());
        return res.norm();
    }

    void calcHeatSinkJacobian(const Eigen::Ref<const Eigen::VectorXd>& state, Eigen::SparseMatrix<double>& jac) {
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();
        const int total_size = ne * (nf * nxi) + nhs;

        assert(jac.rows() == total_size && jac.cols() == jac.rows());

        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        TripletList jactrips;
        JacobianView jacview(jactrips, ne, nf, nxi, nhs);

        TripletList dqblf_trips, dqlf_trips; 
        InputView qbf_view(dqblf_trips, ne, nf, nxi, nhs), qlf_view(dqlf_trips, ne, nf, nxi, nhs);


        jacview.add_block_Ths_Ths(ThermalSolver->calcThermalStateJacobian(sview.Ths()));
        ListOf_TripletList dTwall_trips = ThermalSolver->calcThermalOutputsJacobian(sview.Ths()).second;
        assert(dTwall_trips.size() == ne * nxi);

        for (int e=0; e<ne; ++e) {
            std::span<TripletList> dTwall_star_dThs_star_e( dTwall_trips.data()+e*nxi, dTwall_trips.data()+e*nxi+nxi );

            jacview.add_block_elem(e, FlowSolverList[e]->calcFlowStateJacobian(sview.fstate_mat(e), flow_reorder));
            jacview.add_block_elem_Ths(e, FlowSolverList[e]->calcFlowInputJacobian(sview.fstate_mat(e), dTwall_star_dThs_star_e, flow_reorder));
            
            auto[dqblf_star_l, dqlf_star_l] = FlowSolverList[e]->calcFlowOutputsJacobian(sview.fstate_mat(e), true);
            qbf_view.set_input_jac(e, dqblf_star_l); qlf_view.set_input_jac(e, dqlf_star_l);

        }

        Eigen::SparseMatrix<double> dqblf_duf, dqlf_duf;
        dqblf_duf.resize(ne*nxi, ne*(nf*nxi));
        dqlf_duf.resize(ne*nxi, ne*(nf*nxi));
        dqblf_duf.setFromTriplets(dqblf_trips.begin(), dqblf_trips.end()); dqblf_duf.makeCompressed();
        dqlf_duf.setFromTriplets(dqlf_trips.begin(), dqlf_trips.end()); dqlf_duf.makeCompressed();
        
        jacview.add_Ths_elems(ThermalSolver->calcThermalInputJacobian(sview.Ths(), dqblf_duf, dqlf_duf));
        jac.setFromTriplets(jactrips.begin(), jactrips.end());
        jac.makeCompressed();
    }

    double calcHeatSinkTotalGradient(const Eigen::Ref<const Eigen::VectorXd> state,
                                    const Eigen::Ref<const Eigen::VectorXd> adj,
                                    const double& flow_bcs_star_bar,
                                    const Eigen::Ref<const Eigen::VectorXd> vxyz_star_bar,
                                    const Eigen::Ref<const Eigen::VectorXd> xi_star_bar,
                                    const Eigen::Ref<const Eigen::VectorXd> g_star_bar,
                                    const double& J_bar)
    {   
        assert(state.size() == adj.size() && "state.size() != adj.size(), they should have the same size!");
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();
        const int input_size = ne * nxi;
        const int block_size = nf * nxi;

        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        Eigen::VectorXd Twall_star_bar(input_size); Twall_star_bar.setZero();
        ThermalSolver->diffThermalOutputs(sview.Ths(), vxyz_star_bar, Twall_star_bar);

        Eigen::VectorXd qblf_star_bar(input_size), qlf_star_bar(input_size);
        qblf_star_bar.setZero(); qlf_star_bar.setZero();

        double total_gradient = 0.0;
        for (int e=0; e<ne; ++e) {
            total_gradient += FlowSolverList[e]->calcFlowTotalGradient(sview.fstate_mat(e), 
                                                     adj.segment(e*block_size, block_size),
                                                     flow_bcs_star_bar,
                                                     g_star_bar.segment(e*nxi, nxi), 
                                                     Twall_star_bar.segment(e*nxi, nxi),
                                                     xi_star_bar.segment(e*nxi, nxi), 
                                                     this->flow_reorder);
            
            FlowSolverList[e]->diffFlowOutputs(sview.fstate_mat(e),
                                                xi_star_bar.segment(e*nxi, nxi),
                                                qblf_star_bar.segment(e*nxi, nxi),
                                                qlf_star_bar.segment(e*nxi, nxi));
        }

        total_gradient += ThermalSolver->calcThermalTotalGradient(sview.Ths(), 
                                                                  adj.segment(ne*block_size, nhs), 
                                                                  vxyz_star_bar, 
                                                                  qblf_star_bar,
                                                                  qlf_star_bar);

        total_gradient += J_bar;        
        return total_gradient;
    }

    void calcHeatSinkTotalGradientMulti(const Eigen::Ref<const Eigen::VectorXd> state,
                               const Eigen::Ref<const Eigen::MatrixXd> adj_matrix, // [n_states x n_outputs]
                               const double& flow_bcs_star_bar,
                               const Eigen::Ref<const Eigen::VectorXd> vxyz_star_bar,
                               const Eigen::Ref<const Eigen::VectorXd> xi_star_bar,
                               const Eigen::Ref<const Eigen::VectorXd> g_star_bar,
                               const Eigen::Ref<const Eigen::VectorXd> J_bar_vec, // [n_outputs]
                               Eigen::Ref<Eigen::VectorXd> total_gradients)       // Output: [n_outputs]
    {   
        // Guards against shape mismatches
        assert(state.size() == adj_matrix.rows() && "Adjoint matrix rows must equal state size");
        assert(adj_matrix.cols() == total_gradients.size() && "Adjoint matrix cols must equal output size");
        assert(J_bar_vec.size() == total_gradients.size() && "J_bar_vec size must match output size");

        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();
        const int input_size = ne * nxi;
        const int block_size = nf * nxi;
        const int n_outputs = adj_matrix.cols();

        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        // --- OPTIMIZATION: Compute geometric sensitivities ONCE ---
        Eigen::VectorXd Twall_star_bar(input_size); 
        Twall_star_bar.setZero();
        ThermalSolver->diffThermalOutputs(sview.Ths(), vxyz_star_bar, Twall_star_bar);

        Eigen::VectorXd qblf_star_bar(input_size), qlf_star_bar(input_size);
        qblf_star_bar.setZero(); 
        qlf_star_bar.setZero();

        for (int e=0; e<ne; ++e) {
            FlowSolverList[e]->diffFlowOutputs(sview.fstate_mat(e),
                                                xi_star_bar.segment(e*nxi, nxi),
                                                qblf_star_bar.segment(e*nxi, nxi),
                                                qlf_star_bar.segment(e*nxi, nxi));
        }

        // --- Loop over multiple adjoint vectors (outputs) ---
        for (int k = 0; k < n_outputs; ++k) {
            double current_total_gradient = 0.0;
            
            // Efficient column access (no copy if ColMajor)
            const auto current_adj = adj_matrix.col(k); 

            for (int e=0; e<ne; ++e) {
                current_total_gradient += FlowSolverList[e]->calcFlowTotalGradient(
                                                        sview.fstate_mat(e), 
                                                        current_adj.segment(e*block_size, block_size),
                                                        flow_bcs_star_bar,
                                                        g_star_bar.segment(e*nxi, nxi), 
                                                        Twall_star_bar.segment(e*nxi, nxi),
                                                        xi_star_bar.segment(e*nxi, nxi), 
                                                        this->flow_reorder);
            }

            current_total_gradient += ThermalSolver->calcThermalTotalGradient(
                                                        sview.Ths(), 
                                                        current_adj.segment(ne*block_size, nhs), 
                                                        vxyz_star_bar, 
                                                        qblf_star_bar,
                                                        qlf_star_bar);

            current_total_gradient += J_bar_vec(k);        

            total_gradients(k) = current_total_gradient;
        }
    }

    std::pair<double, double>
    calcHeatSinkOutputs(const Eigen::Ref<const Eigen::VectorXd> state) 
    {
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();

        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        double thermal_flux_star = this->ThermalSolver->calcThermalOutputs(sview.Ths()).first;
        
        double vol_flow_rate = 0.0;
        double pressure_drop_avg = 0.0;
        for (int e=0; e<ne; ++e) {
            vol_flow_rate += FlowSolverList[e]->calcNDInletVolumetricFlowRate(sview.fstate_mat(e));
            pressure_drop_avg += FlowSolverList[e]->calcNDPressureDrop(sview.fstate_mat(e));
        }   
        pressure_drop_avg /= (double)(ne);
        
        auto fbcs = FlowSolverList[0]->getFlowBCs();

        double norm_thermal_flux_star;
        if (this->solve_for_mdot) {
            norm_thermal_flux_star = thermal_flux_star / ( vol_flow_rate * (this->flow_bcs_star_ - fbcs.pout_star) );
        }
        else {
            norm_thermal_flux_star = thermal_flux_star / ( ne*this->flow_bcs_star_ * pressure_drop_avg );
        }
        return std::make_pair(thermal_flux_star, norm_thermal_flux_star);
    }

    void calcHeatSinkOutputsJacobian(const Eigen::Ref<const Eigen::VectorXd> state,
                                 Eigen::Ref<Eigen::VectorXd> dflux_star_dstate,
                                 Eigen::Ref<Eigen::VectorXd> dnormFlux_star_dstate) 
    {   
        dflux_star_dstate.setZero();
        dnormFlux_star_dstate.setZero();

        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();
        const int block_size = nf * nxi;

        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        // 1. Get current values
        auto [tf_star, ntf_star] = this->calcHeatSinkOutputs(state);
        
        // Total Denominator G = tf_star / ntf_star
        double G = tf_star / ntf_star; 

        // 2. Component: d(NTF)/d(ThermalState)
        // d(NTF)/dThs = (1/G) * d(tf)/dThs
        dflux_star_dstate.segment(ne * block_size, nhs) = ThermalSolver->calcThermalOutputsJacobian(sview.Ths()).first;
        dnormFlux_star_dstate.segment(ne * block_size, nhs) = dflux_star_dstate.segment(ne * block_size, nhs) / G;

        // 3. Component: d(NTF)/d(FlowState)
        // d(NTF)/du_flow = - (tf_star / G^2) * dG/du_flow
        for (int e=0; e<ne; ++e) {
            Eigen::VectorXd dG_de(block_size); dG_de.setZero();
            if (solve_for_mdot) {
                // G = sum(Q_dot * Dp) -> dG/du = Dp * dQdot/du
                double Dp = this->flow_bcs_star_ - FlowSolverList[e]->getFlowBCs().pout_star;
                FlowSolverList[e]->calcNDInletVolumetricFlowRateGradient(sview.fstate_mat(e), dG_de);
                dnormFlux_star_dstate.segment(e*block_size, block_size) = (-ntf_star / G) * (Dp * dG_de);
            } else {
                // G = sum(ne * BC * Dp_avg) -> dG/du = BC * dDp/du
                FlowSolverList[e]->calcNDPressureDropGradient(sview.fstate_mat(e), dG_de);
                dnormFlux_star_dstate.segment(e*block_size, block_size) = (-ntf_star / G) * (this->flow_bcs_star_ * dG_de);
            }
        }
    }

    std::pair<double, double> 
    diffHeatSinkOutputs(const Eigen::Ref<const Eigen::VectorXd> state, 
                             const double& flow_bcs_star_bar)
    {
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();

        auto sview = make_state_view(state, ne, nf, nxi, nhs);
        
        auto [thermal_flux_star, norm_thermal_flux_star] = this->calcHeatSinkOutputs(state);
        double denominator = thermal_flux_star / norm_thermal_flux_star;

        double vol_flow_rate = 0.0;
        double pressure_drop_avg = 0.0;
        for (int e=0; e<ne; ++e) {
            vol_flow_rate += FlowSolverList[e]->calcNDInletVolumetricFlowRate(sview.fstate_mat(e));
            pressure_drop_avg += FlowSolverList[e]->calcNDPressureDrop(sview.fstate_mat(e));
        }
        pressure_drop_avg /= double(ne);

        double tf_star_bar = 0.0; 
        double ntf_star_bar= -(norm_thermal_flux_star/denominator);

        ntf_star_bar *= (this->solve_for_mdot) ? vol_flow_rate : (double)(ne) * pressure_drop_avg;
        ntf_star_bar *= flow_bcs_star_bar;
        return std::make_pair(tf_star_bar, ntf_star_bar);
    }

    void setInitialConditions(Eigen::Ref<Eigen::VectorXd> state, double fbcs_star) {
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();
        const int input_size = ne * nxi;
        // const int total_size = ne * (nf * nxi) + nhs;

        // a view to set and send the state vectors in the right format
        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        // initializing precondiitoner with reasonable decoupled solves 
        FlowBoundaryConditions<double> fbcs = FlowSolverList[0]->getFlowBCs();
        auto ref = createRefrigerant<double>(fbcs.refrigerant_name);
        double Tsat_star = ref->calc_sat_temperature(fbcs_star * char_props_.p_c) / char_props_.T_c;

        Eigen::VectorXd qblf_star(input_size), qlf_star(input_size);
        for(int e=0; e<ne; ++e) {
            std::cout << "------------- Preconditioner - Decoupled - FlowSolve - " << e+1 << " -------------\n";
            if (this->two_phase_){
                flow_load_list[e].load_star.setConstant(Tsat_star);
            }
            else {
                flow_load_list[e].load_star.setConstant(fbcs.Tin_star);
            }
            
            FlowSolverList[e]->set_bcs(fbcs_star);
            FlowSolverList[e]->setInputs(flow_load_list[e]);
            FlowSolverList[e]->solveForState(sview.fstate_mat(e));
            FlowSolverList[e]->calcFlowOutputs(sview.fstate_mat(e), 
                                                qblf_star.segment(e*nxi, nxi),
                                                qlf_star.segment(e*nxi, nxi));
        }

        std::cout << "------------- Preconditioner - Decoupled - ThermalSolve -------------\n";
        thermal_load.qblf_star = qblf_star;
        thermal_load.qlf_star  = qlf_star;
        ThermalSolver->setInputs(thermal_load);
        ThermalSolver->solveForState(sview.Ths());
        std::cout << "------------- Preconditioner Developed -------------\n";
    }

    void NonlinearGaussSiedel(Eigen::Ref<Eigen::VectorXd> state, const double fbcs_star) {
        
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();
        const int input_size = ne * nxi;

        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        const int max_iter = 20;
        Eigen::VectorXd qblf_star(input_size), qlf_star(input_size);
        qblf_star.setZero(); qlf_star.setZero();

        std::cout << "------------- Nonlinear Gauss Seidel Start -------------\n";
        FlowBoundaryConditions<double> fbcs = FlowSolverList[0]->getFlowBCs();
        auto ref = createRefrigerant<double>(fbcs.refrigerant_name);

        double Tsat_star = ref->calc_sat_temperature(fbcs_star * char_props_.p_c) / char_props_.T_c;

        Eigen::VectorXd Twall_star(input_size); Twall_star.setConstant(Tsat_star);
        bool useguess = false;
        
        double norm0, norm;
        Eigen::VectorXd res(state.size()); res.setZero();

        for(int iter=1; iter<=max_iter; ++iter){

            for (int e=0; e<ne; e++) {
                this->flow_load_list[e].load_star = Twall_star.segment(e*nxi, nxi);
                this->FlowSolverList[e]->set_bcs(fbcs_star);
                this->FlowSolverList[e]->setInputs(this->flow_load_list[e]);
                this->FlowSolverList[e]->solveForState(sview.fstate_mat(e), this->flow_reorder, useguess);
                this->FlowSolverList[e]->calcFlowOutputs(sview.fstate_mat(e),
                                                        qblf_star.segment(e*nxi, nxi),
                                                        qlf_star.segment(e*nxi, nxi));
            }
            
            this->thermal_load.qblf_star = qblf_star;
            this->thermal_load.qlf_star  = qlf_star;
            this->ThermalSolver->setInputs(this->thermal_load);
            this->ThermalSolver->solveForState(sview.Ths(), true);
            Twall_star =  this->ThermalSolver->calcThermalOutputs(sview.Ths()).second;

            this->calcHeatSinkResidual(state, res);
            norm = res.norm();
            if (iter == 1) { 
                useguess = true; 
                norm0 = norm;
            }
            
            if (norm/norm0 < 1.0e-08 && iter <= max_iter) {
                std::cout << "NLGS iteration " << iter << ": ||r|| = " << norm << ", ||r||/||r_0|| = " << norm/norm0 << "\n";
                std::cout << "------------- Nonlinear Gauss Seidel converged -------------\n";
                break;
            }
            else{
                std::cout << "-------------------------------------------------------------------------------------------\n";
                std::cout << "NLGS iteration " << iter << ": ||r|| = " << norm << ", ||r||/||r_0|| = " << norm/norm0 << "\n";
                std::cout << "-------------------------------------------------------------------------------------------\n";
            }
        }

    }

    void solveForState(Eigen::Ref<Eigen::VectorXd> state, bool use_eigen = false, bool prec_solve = true) {

        state.setZero(); // initial without garbage values

        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();
        // const int input_size = ne * nxi;
        const int total_size = ne * (nf * nxi) + nhs;
        assert(state.size() == total_size);

        // non-linear residual function
        auto res_fun = [this](const Eigen::VectorXd& x, Eigen::VectorXd& r) {
            this->calcHeatSinkResidual(x, r);
            return r.norm();
        };
        // non-linear jacobian function
        auto jac_fun = [this](const Eigen::VectorXd& x, Eigen::SparseMatrix<double>& jac) {
            this->calcHeatSinkJacobian(x, jac);
        };

        if (prec_solve) {
            if (!this->two_phase_){
                setInitialConditions(state, this->flow_bcs_star_);
            }
            else {
                // NonlinearGaussSiedel(state, this->flow_bcs_star_);
                setInitialConditions(state, this->flow_bcs_star_);
            }
        }

        Eigen::SparseMatrix<double> prec(total_size, total_size); 
        this->calcHeatSinkJacobian(state, prec);
        
        bool converged;
        if (use_eigen){
            EigenSolver::NewtonSolver<double> solver(res_fun, jac_fun, true);
            converged = solver.solve(state);
        }
        else {
            // use Inexact Newton
            mfem::EigenNonlinearOperator oper(total_size, res_fun, jac_fun, prec);
            this->prec_matrix.reset(new mfem::HypreParMatrix(oper.GetStaticPreconditionerMatrix()));

            if (!this->real_amg) {
                this->real_amg = std::make_unique<mfem::HypreBoomerAMG>(*this->prec_matrix);
                this->real_amg->SetPrintLevel(0);
                this->real_amg->SetSystemsOptions(1);

                this->frozen_amg = std::make_unique<mfem::FrozenPreconditioner>(*real_amg.get());
                this->gmres->SetPreconditioner(*frozen_amg.get());
            }
            else {
                this->real_amg->SetOperator(*this->prec_matrix);
            }

            this->newton->SetOperator(oper);
            // 5. Solve
            mfem::Vector x_mfem(total_size);
            CopyEigenToMFEMVector(x_mfem, state);
            
            mfem::Vector b_zero(total_size); b_zero = 0.0;
            std::cout << "------------- Inexact Newton Start -------------\n";
            this->newton->Mult(b_zero, x_mfem); // Solve F(x) = 0

            converged = this->newton->GetConverged();

            if(!converged){
                this->NonlinearGaussSiedel(state, this->flow_bcs_star_);
            }
            else{
                CopyMFEMVectorToEigen(state, x_mfem);
                std::cout << "------------- Inexact Newton End -------------\n";
            }
        }
    }

    void solveForAdjoint(const Eigen::Ref<const Eigen::VectorXd> state,
                         const Eigen::Ref<const Eigen::VectorXd> dJdstate,
                         Eigen::Ref<Eigen::VectorXd> adjoint,
                         bool use_eigen_solver = false
                        ) 
    {

        Eigen::SparseMatrix<double> jac(state.size(), state.size());
        this->calcHeatSinkJacobian(state, jac);

        if (use_eigen_solver) {
            Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
            solver.compute(jac.transpose());
            if(solver.info() != Eigen::Success) {
                // decomposition failed
                return;
            }

            adjoint = solver.solve(-dJdstate);
        }
        else {
            Eigen::SparseMatrix<double> jacT = jac.transpose(); jac.makeCompressed();
            mfem::SparseMatrix jacT_mfem;
            CopyEigenSparseToMFEM(jacT, jacT_mfem);

            this->adj_matrix = EigenToHypreParMatrix(jacT, hypre_MPI_COMM_WORLD);

            if(!this->real_amg_adj) {
                this->real_amg_adj = std::make_unique<mfem::HypreBoomerAMG>(*this->adj_matrix.get());
                this->real_amg_adj->SetPrintLevel(0);
                this->real_amg_adj->SetSystemsOptions(1);

                this->frozen_amg_adj = std::make_unique<mfem::FrozenPreconditioner>(*real_amg_adj.get());
                this->gmres_adj->SetPreconditioner(*frozen_amg_adj.get());
            }
            else {
                this->real_amg_adj->SetOperator(*this->adj_matrix.get());
            }
            this->gmres_adj->SetOperator(*this->adj_matrix.get());

            mfem::Vector adjoint_mfem(state.size()); adjoint_mfem = 0.0;
            mfem::Vector b(state.size()); 
            CopyEigenToMFEMVector(b, -dJdstate);

            this->gmres_adj->Mult(b, adjoint_mfem);
            CopyMFEMVectorToEigen(adjoint, adjoint_mfem);
        }
    }

    // ------------------------ Constraint 1 - Tsrc Aggregate -------------------------------

    double calcAggregatedSourceTemperature(const Eigen::Ref<const Eigen::VectorXd> state,
                                           const int boundary_att_id = 1)
    {   
        assert(!this->dirichlet_ && "this method is only available for Robin and Flux boudnary condition thermal solver!!");
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();

        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        Eigen::VectorXd Tbdr_star = this->ThermalSolver->getBoundaryStates(sview.Ths(), boundary_att_id);
        const int N = Tbdr_star.size();

        Eigen::Index max_idx;
        double Tmax_star = Tbdr_star.maxCoeff(&max_idx);

        double exp_avg = 0.0;
        for (int i=0; i<N; ++i) {
            exp_avg += std::exp(this->rho_agg * (Tbdr_star(i) - Tmax_star) );
        }
        exp_avg /= (double)(N);

        return Tmax_star + (std::log(exp_avg) / this->rho_agg);
    }

    void calcAggregateSourceTemperatureJacobian(const Eigen::Ref<const Eigen::VectorXd> state,
                                                Eigen::Ref<Eigen::VectorXd> dTagg_star_dstate,
                                                const int boundary_att_id = 1)
    {   
        assert(!this->dirichlet_ && "this method is only available for Robin and Flux boudnary condition thermal solver!!");
        dTagg_star_dstate.setZero();

        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();
        const int block_size = nf * nxi;

        auto sview = make_state_view(state, ne, nf, nxi, nhs);
        auto Tbdr_star = this->ThermalSolver->getBoundaryStates(sview.Ths(), boundary_att_id);
        auto Tbdr_star_idx = this->ThermalSolver->getBoundaryStatesIdx(sview.Ths(), boundary_att_id);

        Eigen::Index max_idx;
        double Tmax_star = Tbdr_star.maxCoeff(&max_idx); // Valid, but index not strictly needed for logic

        const int N = Tbdr_star.size();
        if (N == 0) return;

        double exp_sum = 0.0;
        Eigen::VectorXd exp_val(N);
        for (int i=0; i<N; ++i) {
            exp_val(i) = std::exp(this->rho_agg * (Tbdr_star(i)-Tmax_star) );
            exp_sum += exp_val(i);
        }

        // Optimization: Precompute inverse sum
        double inv_sum = 1.0 / exp_sum;

        for (int i=0; i<N; ++i) {
            int idx = ne * block_size + Tbdr_star_idx(i);
            dTagg_star_dstate(idx) = exp_val(i) * inv_sum; 
        }
    }

    double diffAggregateSourceTemperature() {return 0.0;}

    // ------------------------ Constraint 2 - xout Aggregate -------------------------------

    double calcAggregateOutletVaporQuality(const Eigen::Ref<const Eigen::VectorXd> state) {
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();

        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        double exp_avg = 0.0;
        for (int e=0; e<ne; ++e) {
            double xout = this->FlowSolverList[e]->calcOutletVaporQuality(sview.fstate_mat(e));
            exp_avg += std::exp(this->rho_agg * (xout - 1.0));
        }
        exp_avg /= (double)(ne);

        double xagg = 1.0 + (std::log(exp_avg) / this->rho_agg);
        return xagg;
    }

    void calcAggregateOutletVaporQualityJacobian(const Eigen::Ref<const Eigen::VectorXd> state,
                                                 Eigen::Ref<Eigen::VectorXd> dxagg_dstate)
    {   
        dxagg_dstate.setZero();

        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();
        const int block_size = nf * nxi;

        auto sview = make_state_view(state, ne, nf, nxi, nhs);
        Eigen::VectorXd exp_val(ne);
        double exp_sum = 0.0;
        for(int e=0; e<ne; ++e) {
            double xout = this->FlowSolverList[e]->calcOutletVaporQuality(sview.fstate_mat(e));
            exp_val(e) = std::exp( this->rho_agg * (xout - 1.0) );
            exp_sum += exp_val(e);
        }

        // Optimization: Precompute inverse sum
        double inv_sum = 1.0 / exp_sum;

        Eigen::VectorXd dxout_dstate(block_size); 
        for (int e=0; e<ne; ++e) {
            this->FlowSolverList[e]->calcOutletVaporQualityJacobian(sview.fstate_mat(e), 
                                                                    dxout_dstate);
            dxagg_dstate.segment(e*block_size, block_size) = exp_val(e) * inv_sum * dxout_dstate;
        }

    }

    double diffAggregateOutletVaporQuality() {return 0.0;}

    // ------------------------ Constraint 3 - Pump constraint -------------------------------

    double calcNDPumpSpeed(const Eigen::Ref<const Eigen::VectorXd> state) {
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();

        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        double vol_flow_rate_star = 0.0;
        double pressure_drop_star_avg = 0.0;
        for (int e=0; e<ne; ++e) {
            vol_flow_rate_star += this->FlowSolverList[e]->calcNDInletVolumetricFlowRate(sview.fstate_mat(e));
            pressure_drop_star_avg += this->FlowSolverList[e]->calcNDPressureDrop(sview.fstate_mat(e));
        }
        pressure_drop_star_avg /= (double)(ne);

        auto fbcs = this->FlowSolverList[0]->getFlowBCs();
        double Nstar;  // non-dimensional pump speed
        if(solve_for_mdot) {
            double Dp_star = this->flow_bcs_star_ - fbcs.pout_star;
            Nstar = this->PumpCons->calcNDPumpSpeed(Dp_star, vol_flow_rate_star);
        }
        else {
            double hin = fbcs.hin_star * this->char_props_.h_c;
            double rho_star = this->FlowSolverList[0]->getFluidDensity(hin) / this->char_props_.rho_c;
            Nstar = this->PumpCons->calcNDPumpSpeed(pressure_drop_star_avg, 
                                        (double)(ne)*this->flow_bcs_star_/rho_star);
        }
        return Nstar;
    }

    void calcNDPumpSpeedJacobian(const Eigen::Ref<const Eigen::VectorXd> state,
                                 Eigen::Ref<Eigen::VectorXd> dNstar_dstate)
    {   
        dNstar_dstate.setZero();
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();
        const int block_size = nf * nxi;

        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        double vol_flow_rate_star = 0.0;
        double pressure_drop_star_avg = 0.0;
        for (int e=0; e<ne; ++e) {
            vol_flow_rate_star += this->FlowSolverList[e]->calcNDInletVolumetricFlowRate(sview.fstate_mat(e));
            pressure_drop_star_avg += this->FlowSolverList[e]->calcNDPressureDrop(sview.fstate_mat(e));
        }
        pressure_drop_star_avg /= (double)(ne);

        auto fbcs = this->FlowSolverList[0]->getFlowBCs();
        Eigen::VectorXd dNstar(2);
        if (solve_for_mdot) {
            double Dp_star = this->flow_bcs_star_ - fbcs.pout_star;
            dNstar = this->PumpCons->calcNDPumpSpeedGradient(Dp_star, vol_flow_rate_star);
        }
        else {
            double hin = fbcs.hin_star * this->char_props_.h_c;
            double rho_star = this->FlowSolverList[0]->getFluidDensity(hin) / this->char_props_.rho_c;
            dNstar = this->PumpCons->calcNDPumpSpeedGradient(pressure_drop_star_avg, (double)(ne)*this->flow_bcs_star_/rho_star);
        }

        Eigen::VectorXd dNduf(block_size); 
        for (int e=0; e<ne; ++e) {
            dNduf.setZero();
            if (solve_for_mdot) {
                this->FlowSolverList[e]->calcNDInletVolumetricFlowRateGradient(sview.fstate_mat(e), dNduf);
                dNstar_dstate.segment(e*block_size, block_size) = dNstar(1) * dNduf;
            }
            else {
                this->FlowSolverList[e]->calcNDPressureDropGradient(sview.fstate_mat(e), dNduf);
                dNstar_dstate.segment(e*block_size, block_size) = (dNstar(0) / (double)(ne)) * dNduf;
            }
        }        
    }

    double diffNDPumpSpeed(const Eigen::Ref<const Eigen::VectorXd> state,
                           const double& flow_bcs_star_bar) 
    {
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();

        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        double vol_flow_rate_star = 0.0;
        double pressure_drop_star_avg = 0.0;
        for (int e=0; e<ne; ++e) {
            vol_flow_rate_star += this->FlowSolverList[e]->calcNDInletVolumetricFlowRate(sview.fstate_mat(e));
            pressure_drop_star_avg += this->FlowSolverList[e]->calcNDPressureDrop(sview.fstate_mat(e));
        }
        pressure_drop_star_avg /= (double)(ne);

        auto fbcs = this->FlowSolverList[0]->getFlowBCs();
        Eigen::VectorXd dNstar(2);
        double Nbar = 0.0;
        if (solve_for_mdot) {
            double Dp_star = this->flow_bcs_star_ - fbcs.pout_star;
            dNstar = this->PumpCons->calcNDPumpSpeedGradient(Dp_star, vol_flow_rate_star);
            Nbar   = dNstar(0) * flow_bcs_star_bar;
        }
        else {
            double hin = fbcs.hin_star * this->char_props_.h_c;
            double rho_star = this->FlowSolverList[0]->getFluidDensity(hin) / this->char_props_.rho_c;
            dNstar = this->PumpCons->calcNDPumpSpeedGradient(pressure_drop_star_avg, (double)(ne)*this->flow_bcs_star_/rho_star);
            Nbar   = dNstar(1) * (double)(ne) * flow_bcs_star_bar / rho_star;
        }
        return Nbar;
    }

    // ------------------------ Constraint 4 - Modified Thermal Resistance -------------------------------

    double calcModifiedThermalResistance(const Eigen::Ref<const Eigen::VectorXd> state,
                                         const double Tsrc_star,
                                         const double alpha,
                                         const double beta,
                                         const int boundary_att_id = 1) 
    {   
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();

        auto sview = make_state_view(state, ne, nf, nxi, nhs);
    
        double Tagg_star= this->calcAggregatedSourceTemperature(state, boundary_att_id);
        double Rth_star = this->ThermalSolver->calcSourceSurfaceUniformity(sview.Ths(), boundary_att_id);

        double vol_flow_rate = 0.0;
        double pressure_drop_avg = 0.0;
        for (int e=0; e<ne; ++e) {
            vol_flow_rate += FlowSolverList[e]->calcNDInletVolumetricFlowRate(sview.fstate_mat(e));
            pressure_drop_avg += FlowSolverList[e]->calcNDPressureDrop(sview.fstate_mat(e));
        }   
        pressure_drop_avg /= (double)(ne);

        double pump_power_star;
        if (this->solve_for_mdot) {
            pump_power_star = vol_flow_rate * (this->flow_bcs_star_ - this->FlowSolverList[0]->getFlowBCs().pout_star);
        }
        else {
            pump_power_star = ne * this->flow_bcs_star_ * pressure_drop_avg;
        }
        return alpha * (Tagg_star/Tsrc_star) + (1.0 - alpha) * Rth_star + (beta * pump_power_star);
    }

    void calcModifiedThermalResistanceJacobian(const Eigen::Ref<const Eigen::VectorXd> state,
                                               Eigen::Ref<Eigen::VectorXd> dmtr_star_dstate,
                                               const double Tsrc_star,
                                               const double alpha,
                                               const double beta,
                                               const int boundary_att_id = 1)
    {      
        dmtr_star_dstate.setZero();

        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();
        const int block_size = nf * nxi;

        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        this->calcAggregateSourceTemperatureJacobian(state, dmtr_star_dstate, boundary_att_id);
        dmtr_star_dstate *= (alpha / Tsrc_star) ;

        Eigen::VectorXd dRth(nhs); dRth.setZero();
        this->ThermalSolver->calcSurfaceUniformityGradient(sview.Ths(), dRth, boundary_att_id);
        dmtr_star_dstate.segment(ne*block_size, nhs) += (1.0 - alpha) * dRth;


        for (int e=0; e<ne; ++e) {
            Eigen::VectorXd dG_de(block_size); dG_de.setZero();
            if (this->solve_for_mdot) {
                double Dp = this->flow_bcs_star_ - FlowSolverList[e]->getFlowBCs().pout_star;
                FlowSolverList[e]->calcNDInletVolumetricFlowRateGradient(sview.fstate_mat(e), dG_de);
                dmtr_star_dstate.segment(e*block_size, block_size) +=  beta * Dp * dG_de;
            } else {
                FlowSolverList[e]->calcNDPressureDropGradient(sview.fstate_mat(e), dG_de);
                dmtr_star_dstate.segment(e*block_size, block_size) +=  beta * this->flow_bcs_star_ * dG_de;
            }
        }

    }

    double diffModifiedThermalResistance(const Eigen::Ref<const Eigen::VectorXd> state,
                                         const double Tsrc_star,
                                         const double flow_bcs_star_bar,
                                         const double alpha,
                                         const double beta,
                                         const int boundary_att_id = 1)
    {
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();

        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        double vol_flow_rate = 0.0;
        double pressure_drop_avg = 0.0;
        for (int e=0; e<ne; ++e) {
            vol_flow_rate += FlowSolverList[e]->calcNDInletVolumetricFlowRate(sview.fstate_mat(e));
            pressure_drop_avg += FlowSolverList[e]->calcNDPressureDrop(sview.fstate_mat(e));
        }   
        pressure_drop_avg /= (double)(ne);

        double mtr_bar;
        if (this->solve_for_mdot) {
            mtr_bar = vol_flow_rate * flow_bcs_star_bar;
        }
        else {
            mtr_bar = ne * pressure_drop_avg * flow_bcs_star_bar;
        }
        mtr_bar *= beta;
        return mtr_bar;
    }

    // ------------------------  Pumping Power -------------------------------

    double calcNDPumpingPower(const Eigen::Ref<const Eigen::VectorXd> state){
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();

        auto sview = make_state_view(state, ne, nf, nxi, nhs);
        double vol_flow_rate = 0.0;
        double pressure_drop_avg = 0.0;
        for (int e=0; e<ne; ++e) {
            vol_flow_rate += FlowSolverList[e]->calcNDInletVolumetricFlowRate(sview.fstate_mat(e));
            pressure_drop_avg += FlowSolverList[e]->calcNDPressureDrop(sview.fstate_mat(e));
        }   
        pressure_drop_avg /= (double)(ne);
        
        auto fbcs = FlowSolverList[0]->getFlowBCs();

        double pump_power_star;
        if (this->solve_for_mdot) {
            pump_power_star =  ( vol_flow_rate * (this->flow_bcs_star_ - fbcs.pout_star) );
        }
        else {
            pump_power_star =  ( ne*this->flow_bcs_star_ * pressure_drop_avg );
        }
        return pump_power_star;
    }

    void calcNDPumpingPowerJacobian(const Eigen::Ref<const Eigen::VectorXd> state,
                                 Eigen::Ref<Eigen::VectorXd> dpower_dstate)
    {   
        assert(state.size() == dpower_dstate.size() && "state.size() != dpower_dstate.size(), should be the same");
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();
        const int block_size = nf * nxi;

        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        dpower_dstate.setZero();

        for (int e=0; e<ne; ++e) {
            Eigen::VectorXd dG_de(block_size); dG_de.setZero();
            if (solve_for_mdot) {
                // G = sum(Q_dot * Dp) -> dG/du = Dp * dQdot/du
                double Dp = this->flow_bcs_star_ - FlowSolverList[e]->getFlowBCs().pout_star;
                FlowSolverList[e]->calcNDInletVolumetricFlowRateGradient(sview.fstate_mat(e), dG_de);
                dpower_dstate.segment(e*block_size, block_size) =  Dp * dG_de;
            } else {
                // G = sum(ne * BC * Dp_avg) -> dG/du = BC * dDp/du
                FlowSolverList[e]->calcNDPressureDropGradient(sview.fstate_mat(e), dG_de);
                dpower_dstate.segment(e*block_size, block_size) =  this->flow_bcs_star_ * dG_de;
            }
        }
    }

    double diffNDPumpingPower(const Eigen::Ref<const Eigen::VectorXd> state, const double flow_bcs_star_bar)
    {
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();

        auto sview = make_state_view(state, ne, nf, nxi, nhs);
        double vol_flow_rate = 0.0;
        double pressure_drop_avg = 0.0;
        for (int e=0; e<ne; ++e) {
            vol_flow_rate += FlowSolverList[e]->calcNDInletVolumetricFlowRate(sview.fstate_mat(e));
            pressure_drop_avg += FlowSolverList[e]->calcNDPressureDrop(sview.fstate_mat(e));
        }   
        pressure_drop_avg /= (double)(ne);
        
        auto fbcs = FlowSolverList[0]->getFlowBCs();

        double pump_power_star_bar;
        if (this->solve_for_mdot) {
            pump_power_star_bar =  ( vol_flow_rate * flow_bcs_star_bar );
        }
        else {
            pump_power_star_bar =  ( ne * flow_bcs_star_bar * pressure_drop_avg );
        }
        return pump_power_star_bar;
    }

    // ------------------------  Constraint: Dp_needed - Dp_available -------------------------------

    double calcPressureDropConstraint(const Eigen::Ref<const Eigen::VectorXd> state) 
    {
        assert(this->solve_for_mdot && "only available when solving for mdot!");
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();

        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        auto fbcs = this->FlowSolverList[0]->getFlowBCs();
        double vol_flow_rate = 0.0;
        for (int e=0; e<ne; ++e) {
            vol_flow_rate += FlowSolverList[e]->calcNDInletVolumetricFlowRate(sview.fstate_mat(e));
        }
        double max_head_available = this->PumpCons->calcNDMaxHeadAvailable(vol_flow_rate);
        return (this->flow_bcs_star_ - fbcs.pout_star) - max_head_available;
    }

    void calcPressureDropConstraintJacobian(const Eigen::Ref<const Eigen::VectorXd> state,
                                                  Eigen::Ref<Eigen::VectorXd> dpdcons_dstate)
    {
        dpdcons_dstate.setZero();
        assert(this->solve_for_mdot && "only available when solving for mdot!");
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();
        const int block_size = nf * nxi;

        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        auto fbcs = this->FlowSolverList[0]->getFlowBCs();
        double vol_flow_rate = 0.0;
        Eigen::VectorXd dG_de(nf * nxi);
        for (int e=0; e<ne; ++e) {
            dG_de.setZero();
            vol_flow_rate += FlowSolverList[e]->calcNDInletVolumetricFlowRate(sview.fstate_mat(e));
            FlowSolverList[e]->calcNDInletVolumetricFlowRateGradient(sview.fstate_mat(e), dG_de);
            dpdcons_dstate.segment(e*block_size, block_size) += dG_de;
        }
        double max_head_gradient = this->PumpCons->calcNDMaxHeadAvailableGradient(vol_flow_rate);
        dpdcons_dstate *= -max_head_gradient;
    }

    double diffPressureDropConstraint(const Eigen::Ref<const Eigen::VectorXd> state, 
                                      const double flow_bcs_star_bar)
    {
        return flow_bcs_star_bar;
    }

    // ------------------------ My Own Objective on Source Temperature Maintanence ------------------

    double calcSafeSourceTemperature(const Eigen::Ref<const Eigen::VectorXd> state,
                                     const double Tsrc_star,
                                     const double alpha,
                                     const int boundary_att_id = 1)
    {
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();

        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        double Tagg_star= this->calcAggregatedSourceTemperature(state, boundary_att_id);
        double Rth_star = this->ThermalSolver->calcSourceSurfaceUniformity(sview.Ths(), boundary_att_id);
        return alpha * (Tagg_star/Tsrc_star) + (1.0 - alpha) * Rth_star;
    }

    void calcSafeSourceTemperatureJacobian(const Eigen::Ref<const Eigen::VectorXd> state,
                                            Eigen::Ref<Eigen::VectorXd> dJ_dstate,
                                            const double Tsrc_star,
                                            const double alpha,
                                            const int boundary_att_id = 1)
    {
        dJ_dstate.setZero();
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();
        const int block_size = nf * nxi;

        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        this->calcAggregateSourceTemperatureJacobian(state, dJ_dstate, boundary_att_id);
        dJ_dstate *= alpha/Tsrc_star;

        Eigen::VectorXd dRth_dstate(nhs); dRth_dstate.setZero();
        this->ThermalSolver->calcSurfaceUniformityGradient(sview.Ths(), dRth_dstate, boundary_att_id);

        dJ_dstate.segment(ne * block_size, nhs) += (1.0 - alpha) * dRth_dstate;
    }

    double diffSafeSourceTemperature(const Eigen::Ref<const Eigen::VectorXd> state, 
                                     const double Tsrc_star,
                                     const double flow_bcs_star_bar,
                                     const double alpha,
                                     const int boundary_att_id = 1)
    {
        return 0.0;
    }

    // ------------------------  Save Stuff -------------------------------

    void SaveToParaview( const Eigen::Ref<const Eigen::VectorXd> state, 
                         const std::string& tcollection = "thermal",
                         const std::string& fcollection = "flow" ) 
    {
        const int ne = design_params_.getNumOfChannels();
        const int nf = FlowSolverList[0]->getNumFlowStates();
        const int nxi= design_params_.nxi;
        const int nhs= ThermalSolver->getNumThermalStates();
        const int input_size = ne * nxi;

        auto sview = make_state_view(state, ne, nf, nxi, nhs);

        const int mesh_dim  = 1;
        const int space_dim = 3;
        const int forder = 1;

        auto all_xstar = this->thermal_load.vxyz_star.segment(0, input_size);
        auto all_ystar = this->thermal_load.vxyz_star.segment(input_size, input_size);
        auto all_zstar = this->thermal_load.vxyz_star.segment(2*input_size, input_size);
        
        mfem::Mesh fmesh(mesh_dim, 0, 0, 0, space_dim);

        for(int e = 0; e < ne; ++e) {
            auto x_channel = this->char_props_.L_c * all_xstar.segment(e*nxi, nxi);
            auto y_channel = this->char_props_.L_c * all_ystar.segment(e*nxi, nxi);
            auto z_channel = this->char_props_.L_c * all_zstar.segment(e*nxi, nxi);
            
            // Add Vertices
            for (int j = 0; j < nxi; ++j){
                fmesh.AddVertex(x_channel(j), y_channel(j), z_channel(j));
            }

            // Add Elements
            int vertex_offset = e * nxi; 
            for (int k = 0; k < nxi - 1; ++k) {
                int v1 = vertex_offset + k;
                int v2 = vertex_offset + k + 1;
                // Attribute is e+1
                fmesh.AddElement(new mfem::Segment(v1, v2, e + 1));
            }
        }

        fmesh.FinalizeTopology();
        // 3. Finalization Sequence
        // This calculates the internal topology tables.
        // If this passes, GetNumGeometries(1) should be > 0.
        fmesh.Finalize(false); 

        // 4. Ensure Nodes
        // This is required to create a FiniteElementSpace on the mesh later.
        fmesh.EnsureNodes();

        mfem::H1_FECollection fec(forder, mesh_dim);
        mfem::FiniteElementSpace fespace(&fmesh, &fec);
        
        mfem::GridFunction u_gf(&fespace);
        mfem::GridFunction p_gf(&fespace);
        mfem::GridFunction h_gf(&fespace);
        mfem::GridFunction T_gf(&fespace);

        Eigen::MatrixXd fstate(nf, nxi); fstate.setZero();
        int global_dof = 0;
        for (int e=0; e<ne; ++e) {
            auto fbcs = this->FlowSolverList[e]->getFlowBCs();
            auto ref = createRefrigerant<double>(fbcs.refrigerant_name);
            fstate = sview.fstate_mat(e);
            auto u_vec = this->char_props_.u_c * fstate.row(0); // extract velocity
            auto p_vec = this->char_props_.p_c * fstate.row(1); // extract pressure
            auto h_vec = this->char_props_.h_c * fstate.row(2); // extract enthalpy
            for (int j=0; j<nxi; ++j){
                u_gf(global_dof) = u_vec(j);
                p_gf(global_dof) = p_vec(j);
                h_gf(global_dof) = h_vec(j);
                T_gf(global_dof) = ref->calc_homogeneous_temperature(h_vec(j), p_vec(j));
                global_dof++;
            }
        }

        // Save using ParaViewDataCollection
        auto pd = std::make_unique<mfem::ParaViewDataCollection>(fcollection, &fmesh);
        pd->SetPrefixPath("ParaView");
        pd->SetLevelsOfDetail(forder);
        pd->SetDataFormat(mfem::VTKFormat::BINARY);
        pd->SetHighOrderOutput(true);
        pd->SetTime(0.0);
        pd->SetCycle(0);
        pd->RegisterField("Ref:Velocity", &u_gf);
        pd->RegisterField("Ref:Pressure", &p_gf);
        pd->RegisterField("Ref:Enthalpy", &h_gf);
        pd->RegisterField("Ref:Temperature", &T_gf);
        pd->Save();

        this->ThermalSolver->SaveToParaView(sview.Ths(), tcollection);
    }

};


#endif // HEATSINK_SOLVER_HPP

