#ifndef THERMAL_SOLVER_HPP
#define THERMAL_SOLVER_HPP

#include "mfem.hpp"
#include "../ref/HomogeneousRefrigerant.hpp"
#include "SingularLoadIntegrators.hpp"
#include "ThermalBCCoefficient.hpp"
#include "ThermalOutputs.hpp"
#include "../utils/utils.hpp"
#include "../utils/NewtonSolver.hpp"
#include "../utils/InexactNewton.hpp"
#include "../utils/LinearAlgebra.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;
using namespace mfem;

using SparseMatrixCol = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using SparseMatrixRow = Eigen::SparseMatrix<double, Eigen::RowMajor>;


// --- Base Class ---

class BaseThermalSolver {
private:
    std::unique_ptr<mfem::FindPointsGSLIB> fp;
    std::unique_ptr<mfem::HypreBoomerAMG> real_amg;
    std::unique_ptr<mfem::FrozenPreconditioner> frozen_amg;
    std::unique_ptr<mfem::GMRESSolver> gmres;
    std::unique_ptr<InexactNewton> newton;

    std::unique_ptr<mfem::HypreParMatrix> P_hypre;

protected:
    static constexpr int dim = 3;
    static constexpr int order = 1;

    DesignParameters design_params;
    MeshParameters mesh_params;
    ThermalBoundaryConditions bcs;
    CharacteristicProperties<double> char_props;

    // Mesh and FEM
    mfem::Mesh mesh;
    std::unique_ptr<mfem::FiniteElementCollection> fec;
    std::unique_ptr<mfem::FiniteElementSpace> fespace;

    // Common Coefficient
    std::unique_ptr<mfem::Coefficient> QCoeff; // Conductivity

    // Solutions and Adjoints
    std::unique_ptr<mfem::GridFunction> Tstar_gf;
    
    // Load Data
    mfem::Vector vxyz_star;
    mfem::Vector qlf_star;
    mfem::Vector qblf_star;

    Eigen::SparseMatrix<double> Keig;

    // Virtual methods for specific BC implementation
    virtual void AddBoundaryBilinearTerms(mfem::BilinearForm &a) = 0;
    virtual void AddBoundaryLinearTerms(mfem::LinearForm &b) = 0;

public:
    BaseThermalSolver(const DesignParameters& design_params_, const MeshParameters& mesh_params_, 
                      const ThermalBoundaryConditions& bcs_, const CharacteristicProperties<double>& char_props_)
        : design_params(design_params_), mesh_params(mesh_params_), bcs(bcs_), char_props(char_props_) 
    {
        // 1. Mesh Construction
        mfem::Element::Type element = (mesh_params.elem_type == "HEXAHEDRON") ? mfem::Element::HEXAHEDRON : mfem::Element::TETRAHEDRON;
        
        double xL_star = mesh_params.x_domain / double(char_props.L_c);
        double yL_star = mesh_params.y_domain / double(char_props.L_c);
        double zL_star = mesh_params.z_domain / double(char_props.L_c);

        mesh = mfem::Mesh::MakeCartesian3D(mesh_params.num_x, mesh_params.num_y, mesh_params.num_z, 
                                            element, xL_star, yL_star, zL_star);
        mesh.EnsureNodes();

        // 2. Spaces
        fec = std::make_unique<mfem::H1_FECollection>(order, dim); 
        fespace = std::make_unique<mfem::FiniteElementSpace>(&mesh, fec.get());
        Tstar_gf = std::make_unique<mfem::GridFunction>(fespace.get());

        // 3. Coefficients
        QCoeff = std::make_unique<mfem::ConstantCoefficient>(design_params.kappa / char_props.kappa_c);

        // 4. Load Vectors Initialization
        vxyz_star.SetSize(dim * design_params.ne * design_params.nxi); vxyz_star = 0.0;
        qlf_star.SetSize(design_params.ne * design_params.nxi); qlf_star = 0.0;
        qblf_star.SetSize(design_params.ne * design_params.nxi); qblf_star = 0.0;

        this->fp = std::make_unique<mfem::FindPointsGSLIB>();
        this->fp->Setup(this->mesh);
        this->initSolvers();
    }

    void initSolvers() {
        this->gmres = std::make_unique<mfem::GMRESSolver>(hypre_MPI_COMM_WORLD);
        this->gmres->SetKDim(50);
        this->gmres->SetMaxIter(100);
        this->gmres->SetRelTol(1e-08);

        this->newton = std::make_unique<InexactNewton>();
        this->newton->SetSolver(*gmres.get());
        this->newton->SetPrintLevel(1);
        this->newton->SetRelTol(1.0e-06);
        this->newton->SetAbsTol(1.0e-10);
        this->newton->SetMaxIter(10);
    }

    virtual ~BaseThermalSolver() {this->fp->FreeData();}

    void setInputs(LoadParameters load) {
        CopyEigenToMFEMVector(vxyz_star, load.vxyz_star);
        CopyEigenToMFEMVector(qblf_star, load.qblf_star);
        CopyEigenToMFEMVector(qlf_star, load.qlf_star);
    }

    const int getNumThermalStates() const { return fespace->GetVSize(); }
    const ThermalBoundaryConditions getThermalBCs() const { return bcs; }
    const MeshParameters getMeshParams() const {return this->mesh_params;}

    virtual void updateThermalBCs(const double source_load_star) = 0;

    // --- Core Physics Calculation ---
    
    void calcThermalResidual(const Eigen::Ref<const Eigen::VectorXd>& Ths_star_, 
                            Eigen::Ref<Eigen::VectorXd> resT) 
    {
        mfem::BilinearForm a(fespace.get());
        
        // 1. Domain Integrators (Standard Diffusion + Singular Load)
        a.AddDomainIntegrator(new mfem::DiffusionIntegrator(*QCoeff.get()));
        a.AddDomainIntegrator(new SingularBLFIntegrator(this->fp, this->vxyz_star, this->qblf_star));

        // 2. Specific Boundary Terms (Virtual Call)
        AddBoundaryBilinearTerms(a);
        
        a.Assemble();

        mfem::LinearForm b(fespace.get());
        
        // 3. Linear Integrators (Singular Load)
        b.AddDomainIntegrator(new SingularLFIntegrator(this->fp, this->vxyz_star, this->qlf_star));
        
        // 4. Specific Boundary Sources (Virtual Call)
        AddBoundaryLinearTerms(b);
        
        b.Assemble();

        // 5. Form Residual: Res = K * T - b
        // Note: We use -1.0 * b to match residual definition Res = K*u - f = 0
        mfem::Vector res(b);
        res *= -1.0;

        mfem::Array<int> ess_tdof_list; // Empty, as boundaries are weakly enforced in DG
        mfem::SparseMatrix K;
        a.FormSystemMatrix(ess_tdof_list, K);
    
        CopyMFEMSparseToEigen(K, Keig);
        CopyEigenToMFEMGridFunction(Tstar_gf, Ths_star_);
        
        K.AddMult(*Tstar_gf.get(), res, 1.0);
        CopyMFEMVectorToEigen(resT, res);
    }

    // --- Jacobian Calculations ---

    std::vector<Eigen::Triplet<double>> 
    calcThermalStateJacobian(const Eigen::Ref<const Eigen::VectorXd>& Ths_star_) 
    {
        return EigenSparsetoTriplets(Keig);
    }

    std::vector<Eigen::Triplet<double>> 
    calcThermalInputJacobian(const Eigen::Ref<const Eigen::VectorXd>& Ths_star_, 
                             const Eigen::SparseMatrix<double>& dqblf_star,
                             const Eigen::SparseMatrix<double>& dqlf_star) 
    {
        CopyEigenToMFEMGridFunction(Tstar_gf, Ths_star_);

        SingularBLFIntegrator blf_integ(this->fp, vxyz_star, qblf_star);
        Eigen::SparseMatrix<double> dKThs_dqblf = blf_integ.calcLoadJacobian(this->fespace, this->Tstar_gf);

        SingularLFIntegrator lf_integ(this->fp, vxyz_star, qlf_star);
        Eigen::SparseMatrix<double> db_dqlf = lf_integ.calcLoadJacobian(this->fespace);
        
        Eigen::SparseMatrix<double> dresT_duf = (dKThs_dqblf * dqblf_star) - (db_dqlf * dqlf_star);
        return EigenSparsetoTriplets(dresT_duf);
    }

    // --- Output Calculations ---

    std::pair<double, Eigen::VectorXd> 
    calcThermalOutputs(const Eigen::Ref<const Eigen::VectorXd>& Ths_star_) 
    {
        CopyEigenToMFEMGridFunction(Tstar_gf, Ths_star_);
        mfem::Array<int> bdr_arr(mesh.bdr_attributes.Max());
        bdr_arr = 0; 
        bdr_arr[0] = 1; // Assuming primary output on attribute 1
        double thermal_flux_star = calcBoundaryFlux(mesh, fespace.get(), *QCoeff.get(), *Tstar_gf.get(), bdr_arr);
        Eigen::VectorXd Twall_star = calcEvaporatorWallTemperature(this->fp, this->vxyz_star, this->mesh, *Tstar_gf.get());
        return std::make_pair(thermal_flux_star, Twall_star);
    }

    using ListOf_TripletList = std::vector<std::vector<Eigen::Triplet<double>>>;
    
    std::pair<Eigen::VectorXd, ListOf_TripletList> 
    calcThermalOutputsJacobian(const Eigen::Ref<const Eigen::VectorXd>& Ths_star_) 
    {
        CopyEigenToMFEMGridFunction(Tstar_gf, Ths_star_);
        mfem::Array<int> bdr_arr(mesh.bdr_attributes.Max());
        bdr_arr = 0; bdr_arr[0] = 1;
        Eigen::VectorXd dflux = calcBoundaryFluxGradient(mesh, fespace.get(), *QCoeff.get(), bdr_arr);
        ListOf_TripletList dTwall = calcEvaporatorWallTemperatureJacobian(this->fp, this->vxyz_star, this->mesh, this->fespace);
        return std::make_pair(dflux, dTwall);
    }

    void diffThermalOutputs(const Eigen::Ref<const Eigen::VectorXd> Ths_star_,
                            const Eigen::Ref<const Eigen::VectorXd> vxyz_star_bar,
                            Eigen::Ref<Eigen::VectorXd> Twall_star_bar)
    {   
        CopyEigenToMFEMGridFunction(this->Tstar_gf, Ths_star_);
        diffEvaporatorWallTemperature(this->fp, this->vxyz_star, this->mesh, this->fespace, this->Tstar_gf, vxyz_star_bar, Twall_star_bar);
    }

    double calcThermalTotalGradient(const Eigen::Ref<const Eigen::VectorXd> Ths_star_, 
                                    const Eigen::Ref<const Eigen::VectorXd> adjT_,
                                    const Eigen::Ref<const Eigen::VectorXd> vxyz_star_bar,
                                    const Eigen::Ref<const Eigen::VectorXd> qblf_star_bar,
                                    const Eigen::Ref<const Eigen::VectorXd> qlf_star_bar)
    {
        CopyEigenToMFEMGridFunction(Tstar_gf, Ths_star_);
        auto adjT_gf = std::make_unique<mfem::GridFunction>(this->fespace.get());
        CopyEigenToMFEMGridFunction(adjT_gf, adjT_);

        SingularBLFIntegrator blf_integ(this->fp, this->vxyz_star, this->qblf_star);
        double KThs_bar = blf_integ.calcTotalGradient(this->fespace, vxyz_star_bar, qblf_star_bar, adjT_gf, this->Tstar_gf);

        SingularLFIntegrator lf_integ(this->fp, vxyz_star, qlf_star);
        double b_bar = lf_integ.calcTotalGradient(this->fespace, vxyz_star_bar, qlf_star_bar, adjT_gf);

        return KThs_bar - b_bar;
    }

    // --- Solver Logic ---

    void solveForState(Eigen::Ref<Eigen::VectorXd> Ths_star_, bool use_eigen = false) {
        const int nhs = this->getNumThermalStates();
        assert(Ths_star_.size() == nhs);

        auto res_fun = [this](const Eigen::VectorXd& current_x, Eigen::VectorXd& r) {
            this->calcThermalResidual(current_x, r);
            return r.norm();
        };

        auto jac_fun = [this](const Eigen::VectorXd& current_x, Eigen::SparseMatrix<double>& jac) {
            jac.resize(current_x.size(), current_x.size());
            std::vector<Eigen::Triplet<double>> jtrips = this->calcThermalStateJacobian(current_x);
            jac.setFromTriplets(jtrips.begin(), jtrips.end()); jac.makeCompressed();
        };

        if (use_eigen){
            EigenSolver::NewtonSolver<double> solver(res_fun, jac_fun, true);
            solver.solve(Ths_star_);
        }
        else {
            // Construct Preconditioner Matrix that matches the physics
            mfem::BilinearForm prec_form(fespace.get());
            prec_form.AddDomainIntegrator(new mfem::DiffusionIntegrator(*QCoeff.get()));
                
            // IMPORTANT: Add boundary terms to preconditioner so AMG sees the full stiffness (crucial for Robin)
            AddBoundaryBilinearTerms(prec_form); 
            prec_form.Assemble();

            mfem::Array<int> ess_tdof_list; 
            mfem::SparseMatrix P_mat;
            prec_form.FormSystemMatrix(ess_tdof_list, P_mat);

            Eigen::SparseMatrix<double> Prec; 
            CopyMFEMSparseToEigen(P_mat, Prec);

            // Setup Solvers
            mfem::EigenNonlinearOperator oper(nhs, res_fun, jac_fun, Prec);
            this->P_hypre.reset(new mfem::HypreParMatrix(oper.GetStaticPreconditionerMatrix()));

            if(!this->real_amg) {
                this->real_amg = std::make_unique<mfem::HypreBoomerAMG>(*this->P_hypre);
                this->real_amg->SetPrintLevel(0);
                this->real_amg->SetSystemsOptions(1);

                this->frozen_amg = std::make_unique<mfem::FrozenPreconditioner>(*real_amg.get());
                this->gmres->SetOperator(*frozen_amg.get());
            }
            else {
                this->real_amg->SetOperator(*this->P_hypre);
            }
            this->newton->SetOperator(oper);
            mfem::Vector x_sol(nhs); CopyEigenToMFEMVector(x_sol, Ths_star_);
            mfem::Vector b_zero(nhs); b_zero = 0.0;

            this->newton->Mult(b_zero, x_sol);
            CopyMFEMVectorToEigen(Ths_star_, x_sol);
        }
    }

    void SaveToParaView(const Eigen::Ref<const Eigen::VectorXd> Ths_star_, const std::string collection="sol") {
        auto mesh_dim = std::make_unique<mfem::Mesh>(mesh, true);
        if (mesh_dim->GetNodes()) {
            *mesh_dim->GetNodes() *= char_props.L_c;
        }
        auto fes_dim = std::make_unique<mfem::FiniteElementSpace>(mesh_dim.get(), fec.get());
        auto T_gf = std::make_unique<mfem::GridFunction>(fes_dim.get());
        CopyEigenToMFEMGridFunction(T_gf, char_props.T_c * Ths_star_);

        auto pd = std::make_unique<mfem::ParaViewDataCollection>(collection, mesh_dim.get());
        pd->SetPrefixPath("ParaView");
        pd->SetLevelsOfDetail(order);
        pd->SetDataFormat(mfem::VTKFormat::BINARY);
        pd->SetHighOrderOutput(true);
        pd->RegisterField("Temperature", T_gf.get());
        pd->Save();
    }


    Eigen::VectorXd getBoundaryStates(const Eigen::Ref<const Eigen::VectorXd> Ths_star_,
                                  const int boundary_att_id = 1)
    {   
        // 1. Prepare to collect unique DoFs
        // Use a set to handle shared nodes between elements automatically
        std::set<int> unique_dof_indices;
        mfem::Array<int> dofs;

        // 2. Iterate ONLY over boundary elements
        // This ignores volume elements completely
        for (int i = 0; i < this->fespace->GetNBE(); i++) 
        {
            // Check if this boundary face matches our ID
            if (this->mesh.GetBdrAttribute(i) == boundary_att_id) 
            {
                // Get the DoF indices for this specific face
                this->fespace->GetBdrElementVDofs(i, dofs);
                
                // Insert into set (auto-sorts and removes duplicates)
                for (int j = 0; j < dofs.Size(); j++) {
                    // Handle MFEM's potential negative indices (orientation/constraints)
                    // Decode: index = (dof >= 0) ? dof : -1 - dof;
                    int idx = (dofs[j] >= 0) ? dofs[j] : -1 - dofs[j];
                    unique_dof_indices.insert(idx);
                }
            }
        }

        // 3. Handle Empty Case
        if (unique_dof_indices.empty()) {
            std::cout << "Warning: No DoFs found for boundary attribute " << boundary_att_id << "\n";
            return Eigen::VectorXd(0);
        }

        // 4. Extract Values
        Eigen::VectorXd Tbdr_star(unique_dof_indices.size());
        int count = 0;

        for (int global_idx : unique_dof_indices) 
        {
            // Safety Check
            if (global_idx < 0 || global_idx >= Ths_star_.size()) {
                std::cerr << "Error: DoF " << global_idx << " out of bounds.\n";
                continue; 
            }
            
            Tbdr_star(count) = Ths_star_(global_idx);
            count++;
        }

        return Tbdr_star;
    }

    Eigen::VectorXi getBoundaryStatesIdx(const Eigen::Ref<const Eigen::VectorXd> Ths_star_, 
                                        const int boundary_att_id = 1)
    {
        // 1. Prepare to collect unique DoFs
        // Use a set to handle shared nodes between elements automatically
        std::set<int> unique_dof_indices;
        mfem::Array<int> dofs;

        // 2. Iterate ONLY over boundary elements
        // This ignores volume elements completely
        for (int i = 0; i < this->fespace->GetNBE(); i++) 
        {
            // Check if this boundary face matches our ID
            if (this->mesh.GetBdrAttribute(i) == boundary_att_id) 
            {
                // Get the DoF indices for this specific face
                this->fespace->GetBdrElementVDofs(i, dofs);
                
                // Insert into set (auto-sorts and removes duplicates)
                for (int j = 0; j < dofs.Size(); j++) {
                    // Handle MFEM's potential negative indices (orientation/constraints)
                    // Decode: index = (dof >= 0) ? dof : -1 - dof;
                    int idx = (dofs[j] >= 0) ? dofs[j] : -1 - dofs[j];
                    unique_dof_indices.insert(idx);
                }
            }
        }

        // 3. Handle Empty Case
        if (unique_dof_indices.empty()) {
            std::cout << "Warning: No DoFs found for boundary attribute " << boundary_att_id << "\n";
            Eigen::VectorXi temp(1); temp(0) = -1;
            return temp;
        }

        // 4. Extract Values
        Eigen::VectorXi Tbdr_star_idx(unique_dof_indices.size());
        int count = 0;

        for (int global_idx : unique_dof_indices) 
        {
            // Safety Check
            if (global_idx < 0 || global_idx >= Ths_star_.size()) {
                std::cerr << "Error: DoF " << global_idx << " out of bounds.\n";
                continue; 
            }
            
            Tbdr_star_idx(count) = global_idx;
            count++;
        }

        return Tbdr_star_idx;
    }

    void calcBdrNormal(const int boundary_att_id, mfem::Vector& n) {
        n = 0.0;
        if (boundary_att_id==1) {
            // z = 0
            n(2) = -1;
        }
        else if(boundary_att_id == 2) {
            // y = 0
            n(1) = -1;
        }
        else if (boundary_att_id == 3){
            // x = 1
            n(0) = 1;
        }
        else if (boundary_att_id == 4) {
            // y = 1
            n(1) = 1;
        }
        else if (boundary_att_id == 5) {
            // x = 0
            n(0) = -1;
        }
        else {
            // z = 1
            n(2) = 1;
        }    
    }

    double calcSourceSurfaceUniformity(const Eigen::Ref<const Eigen::VectorXd> Ths_star_,
                                 const int boundary_att_id = 1)
    {
        CopyEigenToMFEMGridFunction(this->Tstar_gf, Ths_star_);
        double integral = 0.0;
        mfem::Vector n(this->dim); n = 0.0; calcBdrNormal(boundary_att_id, n);
        mfem::Vector grad_T(this->dim); grad_T = 0.0;

        for (int i = 0; i < this->fespace->GetNBE(); ++i) {
            if (this->mesh.GetBdrAttribute(i) == boundary_att_id) {

                mfem::ElementTransformation *tr =  this->fespace->GetBdrElementTransformation(i);
                
                int order = 2 * this->fespace->GetBE(i)->GetOrder() + 1;
                const mfem::IntegrationRule& ir = mfem::IntRules.Get(tr->GetGeometryType(), order);

                for (int j=0; j<ir.GetNPoints(); ++j){
                    const mfem::IntegrationPoint& ip = ir.IntPoint(j);
                    tr->SetIntPoint(&ip);
                    this->Tstar_gf->GetGradient(*tr, grad_T);
                    
                    // ||grad_t T||^2 = ||grad T||^2 - (grad T . n)^2
                    double grad_norm_sq = grad_T * grad_T;   
                    double T_n = grad_T * n;                 
                    double integrand = grad_norm_sq - (T_n * T_n);
                    
                    // 5. Integrate
                    // tr->Weight() is the Jacobian of the 2D surface element (Area)
                    integral += ip.weight * tr->Weight() * integrand;
                }
            }
        }
        return integral;
    }

    void calcSurfaceUniformityGradient(const Eigen::Ref<const Eigen::VectorXd> Ths_star_,
                                        Eigen::Ref<Eigen::VectorXd> res_grad,
                                        const int boundary_att_id = 1)
    {
        // 1. Update MFEM GridFunction
        mfem::Vector Ths_star(Ths_star_.size()); Ths_star = 0.0;
        CopyEigenToMFEMVector(Ths_star, Ths_star_);
        
        // 2. Setup Boundary Marker
        // MFEM markers are 0-based arrays where 1 means "active"
        int max_bdr_attr = this->mesh.bdr_attributes.Max();
        mfem::Array<int> bdr_marker(max_bdr_attr);
        bdr_marker = 0;
        if (boundary_att_id <= max_bdr_attr) {
            bdr_marker[boundary_att_id - 1] = 1; // Activate only the specific surface
        }

        // 3. Define the Surface Diffusion Operator
        // "DiffusionIntegrator" computes (grad u, grad v). 
        // When added as a *BoundaryIntegrator*, it computes (grad_t u, grad_t v).
        mfem::BilinearForm surface_stiffness(this->fespace.get());
        
        // Coefficient = 2.0 because derivative of T^2 is 2*T
        mfem::ConstantCoefficient coeff(2.0); 
        surface_stiffness.AddBoundaryIntegrator(new mfem::DiffusionIntegrator(coeff), bdr_marker);

        // 4. Assemble the Operator
        // If your mesh geometry doesn't change, you can move this setup 
        // to the class initialization and do it only ONCE (huge speedup).
        surface_stiffness.Assemble();
        surface_stiffness.Finalize();

        // 5. Compute Gradient: g = K_surf * T
        // We compute the matrix-vector product
        mfem::Vector grad_vec(this->getNumThermalStates());

        surface_stiffness.Mult(Ths_star, grad_vec);

        // 6. Copy result to Eigen
        res_grad.setZero();
        CopyMFEMVectorToEigen(res_grad, grad_vec);
    }

    double calcBdrMeanTemperature(const Eigen::Ref<const Eigen::VectorXd> Ths_star_,
                                        const int boundary_att_id = 1)
    {   
        CopyEigenToMFEMGridFunction(this->Tstar_gf, Ths_star_);

        mfem::Array<int> bdr_arr(this->mesh.bdr_attributes.Max());
        bdr_arr = 0;
        bdr_arr[boundary_att_id-1] = 1;

        return calcBoundaryMeanTemperature(this->mesh, this->fespace, this->Tstar_gf, bdr_arr);
    }

    void calcBdrMeanTemperatureGradient(const Eigen::Ref<const Eigen::VectorXd> Ths_star_,
                                        Eigen::Ref<Eigen::VectorXd> dTmean_dThs,
                                        const int boundary_att_id = 1)
    {
        CopyEigenToMFEMGridFunction(this->Tstar_gf, Ths_star_);

        mfem::Array<int> bdr_arr(this->mesh.bdr_attributes.Max());
        bdr_arr = 0;
        bdr_arr[boundary_att_id-1] = 1;
        
        dTmean_dThs = calcBoundaryMeanTemperatureGradient(this->mesh, this->fespace, this->Tstar_gf, bdr_arr);
    }

};

// --- Derived Class: Dirichlet BCs (Original Logic) ---

class DirichletThermalSolver : public BaseThermalSolver {
private:
    std::unique_ptr<mfem::Coefficient> dbcCoeff;
    mfem::Array<int> dbc_bdr;
    double sigma = -1.0;
    double alpha;

public:
    DirichletThermalSolver(const DesignParameters& dp, const MeshParameters& mp, 
                           const ThermalBoundaryConditions& bcs, const CharacteristicProperties<double>& cp)
        : BaseThermalSolver(dp, mp, bcs, cp) 
    {
        // Setup Dirichlet Coefficients
        if (bcs.f_in_bcs) {
            dbcCoeff = std::make_unique<ForcingCoefficient>(set_dbc_forcing, dim, bcs.Tin_star, bcs.Tsrc_star);
        } else {
            dbcCoeff = std::make_unique<mfem::ConstantCoefficient>(bcs.Tsrc_star);
        }

        // Setup Attributes
        int max_bdr_attr = mesh.bdr_attributes.Max();
        dbc_bdr.SetSize(max_bdr_attr);
        dbc_bdr = 0; 
        if (bcs.Tsrc_bdr_id > 0) dbc_bdr[bcs.Tsrc_bdr_id - 1] = 1;
        if (bcs.f_in_bcs && bcs.Tin_bdr_id > 0) dbc_bdr[bcs.Tin_bdr_id - 1] = 1;

        alpha = 10.0 * (double(order) + 1.0) * (double(order) + 1.0);
    }

    void updateThermalBCs(const double source_load_star) override {
        this->bcs.Tsrc_star = source_load_star;
        if (this->bcs.f_in_bcs) {
            static_cast<ForcingCoefficient*>(this->dbcCoeff.get())->SetSourceTemperature(source_load_star);
        }
        else {
            static_cast<mfem::ConstantCoefficient*>(this->dbcCoeff.get())->constant = source_load_star;
        }
    }

    void AddBoundaryBilinearTerms(mfem::BilinearForm &a) override {
        // Enforce T = Fixed via SIPG Penalty
        a.AddBdrFaceIntegrator(new mfem::DGDiffusionIntegrator(*QCoeff.get(), sigma, alpha), dbc_bdr);
    }

    void AddBoundaryLinearTerms(mfem::LinearForm &b) override {
        // Source for T = Fixed
        b.AddBdrFaceIntegrator(new mfem::DGDirichletLFIntegrator(*dbcCoeff.get(), *QCoeff.get(), sigma, alpha), dbc_bdr);
    }
};

// --- Derived Class: Robin & Flux BCs (Adjoint Consistent) ---

class RobinFluxThermalSolver : public BaseThermalSolver {
private:
    std::unique_ptr<mfem::Coefficient> h_coeff;
    std::unique_ptr<mfem::Coefficient> Tinf_coeff;
    std::unique_ptr<mfem::Coefficient> hTinf_coeff;
    std::unique_ptr<mfem::Coefficient> q_coeff;

    mfem::Array<int> robin_marker;
    mfem::Array<int> flux_marker;
    

public:
    RobinFluxThermalSolver(const DesignParameters& dp, const MeshParameters& mp, 
                           const ThermalBoundaryConditions& bcs, const CharacteristicProperties<double>& cp)
        : BaseThermalSolver(dp, mp, bcs, cp) 
    {
        // Setup Coefficients
        h_coeff = std::make_unique<mfem::ConstantCoefficient>(bcs.h_star);
        Tinf_coeff = std::make_unique<mfem::ConstantCoefficient>(bcs.Tinf_star);
        q_coeff = std::make_unique<mfem::ConstantCoefficient>(bcs.qflux_star);
        hTinf_coeff = std::make_unique<mfem::ProductCoefficient>(*h_coeff.get(), *Tinf_coeff.get());

        // Setup Markers
        int max_bdr = mesh.bdr_attributes.Max();
        robin_marker.SetSize(max_bdr); robin_marker = 0;
        flux_marker.SetSize(max_bdr);  flux_marker = 0;

        if (bcs.robin_bdr_id > 0) robin_marker[bcs.robin_bdr_id - 1] = 1;
        if (bcs.flux_bdr_id > 0)  flux_marker[bcs.flux_bdr_id - 1] = 1;
        
    }

    void updateThermalBCs(const double source_load_star) override {
        this->bcs.qflux_star = source_load_star;
        static_cast<mfem::ConstantCoefficient*>(this->q_coeff.get())->constant = source_load_star;
    }

    void AddBoundaryBilinearTerms(mfem::BilinearForm &a) override {
        // ONLY apply the physical Convection Stiffness
        // This adds the +h*T term to the LHS
        if (bcs.robin_bdr_id > 0) {
            a.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(*h_coeff.get()), robin_marker);
        }
    }

    void AddBoundaryLinearTerms(mfem::LinearForm &b) override {
        // 1. Convection Source: \int (h * Tinf) * v * ds
        // Use BoundaryLFIntegrator for scalar boundary sources
        if (bcs.robin_bdr_id > 0) {
            b.AddBdrFaceIntegrator(new mfem::BoundaryLFIntegrator(*hTinf_coeff.get()), robin_marker);
        }

        // 2. Constant Flux Source: \int q'' * v * ds
        // Since q_coeff is a scalar representing the normal flux magnitude, 
        // we use BoundaryLFIntegrator here as well.
        if (bcs.flux_bdr_id > 0) {
            b.AddBdrFaceIntegrator(new mfem::BoundaryLFIntegrator(*q_coeff.get()), flux_marker);
        }
    }
};

#endif // THERMAL_SOLVER_HPP