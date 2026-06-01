#ifndef NEWTON_SOLVER_HPP
#define NEWTON_SOLVER_HPP

#include "mfem.hpp"
#include <mpi.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <functional>
#include <string>
#include <iostream>
#include <iomanip>
#include <complex>
#include "utils.hpp"

// --- Configuration Constants ---
static constexpr double NEWTON_REL_TOL = 1.0E-12; // Relative tolerance
static constexpr double NEWTON_ABS_TOL = 1.0E-10; // Absolute tolerance
static constexpr int MAX_NEWTONS_ITER = 50;
static constexpr double LINE_SEARCH_STEP = 0.5; // Step reduction factor for line search
static constexpr int MAX_LINE_SEARCH_ITER = 10;


// --- Helper Display Functions ---
// Since the solver is now a template class, the main implementation
// has been moved to NewtonSolver.hpp.
// These non-templated helper functions remain here.

inline void DisplayIterationHistory(const std::string& type, const std::string& solver, int iter, double norm, bool display) {
    if (display) {
        std::cout << std::left << std::setw(12) << type << " | "
                  << std::setw(12) << solver << " | "
                  << "Iter: " << std::setw(3) << iter << " | "
                  << "Residual Norm: " << std::scientific << std::setprecision(4) << norm << std::endl;
    }
}

inline void DisplayConvergence(const std::string& type, const std::string& solver, int iter, double norm, bool display) {
    if (display) {
        std::cout << "SUCCESS: Converged in " << iter << " iterations." << std::endl;
        std::cout << "Final Residual Norm: " << norm << std::endl;
    }
}

inline void DisplayErrorMessage(const std::string& type, const std::string& solver, bool display) {
    if (display) {
        std::cerr << "ERROR: " << type << " " << solver << " solver failed to converge." << std::endl;
    }
}

inline void DisplayResidualRatio(const std::string& type, int iter, double new_norm, double old_norm, bool display) {
    if (display) {
        std::cout << "  > " << std::left << std::setw(12) << type << " | "
                  << "Step: " << std::setw(2) << iter << " | "
                  << "Residual reduced to " << std::scientific << new_norm << std::endl;
    }
}


/**
 * @namespace EigenSolver
 * @brief Namespace to avoid naming conflicts with external libraries like MFEM.
 */
namespace EigenSolver {

/**
 * @class NewtonSolver
 * @brief A templated class to solve nonlinear systems F(x) = 0.
 */
template <typename T>
class NewtonSolver {
public:
    // --- Type Aliases ---
    using TemplateVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using TemplateMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using SparseMat = Eigen::SparseMatrix<T>;
    
    // Define the Sparse Solver Type
    // Using IncompleteLUT (ILU) for the preconditioner
    // using Preconditioner = Eigen::IncompleteLUT<T>;
    // using SparseSolver = Eigen::BiCGSTAB<SparseMat, Preconditioner>;
    using SparseSolver = Eigen::SparseLU<SparseMat>;

    using ResidualFunction = std::function<double(const TemplateVector& x, TemplateVector& res)>;
    using JacobianFunction = std::function<void(const TemplateVector& x, TemplateMatrix& jac)>;
    using SpJacobianFunction = std::function<void(const TemplateVector& x, SparseMat& jac)>;

    // --- Constructors ---

    NewtonSolver(ResidualFunction res_fun, JacobianFunction jac_fun, bool display_history = false)
        : res_fun_(res_fun), jac_fun_(jac_fun), sp_jac_fun_(nullptr), display_history_(display_history), solver_pattern_analyzed_(false) {
        if (!res_fun_ || !jac_fun_) {
            throw std::invalid_argument("Residual and Jacobian functions must be provided.");
        }
    }

    NewtonSolver(ResidualFunction res_fun, SpJacobianFunction jac_fun, bool display_history = false)
        : res_fun_(res_fun), jac_fun_(nullptr), sp_jac_fun_(jac_fun), display_history_(display_history), solver_pattern_analyzed_(false) {
        if (!res_fun_ || !sp_jac_fun_) {
            throw std::invalid_argument("Residual and Sparse Jacobian functions must be provided.");
        }
    }

    /**
     * @brief Solves the nonlinear system.
     */
    bool solve(Eigen::Ref<TemplateVector> x) {
        using Integer = Eigen::Index;
        Integer nd = x.size();
        if (nd == 0) {
            std::cerr << "Error: Initial guess vector 'x' has size 0." << std::endl;
            return false;
        }

        TemplateVector res(nd);
        TemplateMatrix jac(nd, nd);
        SparseMat sp_jac(nd, nd);
        TemplateVector dx(nd);

        bool newton_convergence = false;
        double norm0 = 0.0;

        for (int iter = 1; iter <= MAX_NEWTONS_ITER; ++iter) {
            
            // 1. Calculate residual and check for convergence
            double norm_res = res_fun_(x, res);
            
            if (iter == 1) {
                norm0 = norm_res;
                if (norm0 == 0.0) norm0 = 1.0;
            }

            DisplayIterationHistory("Nonlinear", "Newton", iter, norm_res, display_history_);

            if (norm_res < NEWTON_ABS_TOL || norm_res < norm0 * NEWTON_REL_TOL) {
                DisplayConvergence("Nonlinear", "Newton", iter, norm_res, display_history_);
                newton_convergence = true;
                break;
            }

            // 2. Solve Linear System J * dx = -res
            
            // --- DENSE PATH ---
            if (jac_fun_) {
                jac_fun_(x, jac);
                
                Eigen::PartialPivLU<TemplateMatrix> lu(jac);
                // Use determinant check for compatibility with older Eigen versions
                if (std::abs(lu.determinant()) < 1.0e-12) {
                    std::cerr << "Warning: Jacobian matrix is singular in Newton iteration (Dense LU)." << std::endl;
                    break;
                }
                dx = lu.solve(-res);
            }
            // --- SPARSE PATH ---
            else {
                sp_jac_fun_(x, sp_jac);
                sp_jac.makeCompressed(); // Ensure standard CSR format
                
                // // --- STEP 2: CONFIGURE ITERATIVE SOLVER ---
                // sparse_solver_.setTolerance(1e-6); // Or relative to norm_res if preferred
                // sparse_solver_.setMaxIterations(nd * 2);

                // 3. Analyze sparse pattern
                if (iter == 1) {
                    sparse_solver_.analyzePattern(sp_jac);
                    // // Optional: Configure Preconditioner (e.g. fill factor for ILUT)
                    // sparse_solver_.preconditioner().setFillfactor(7);
                    // sparse_solver_.preconditioner().setDroptol(1e-4);
                }

                // 4. Factorize and Solve
                sparse_solver_.factorize(sp_jac);
                if (sparse_solver_.info() != Eigen::Success) {
                    std::cerr << "Error: Factorization failed (Singular Matrix?)" << std::endl;
                    break;
                }
                
                // 3. Solve
                dx = sparse_solver_.solve(-res);

                if (sparse_solver_.info() != Eigen::Success) {
                    std::cerr << "Warning: Sparse solver failed to converge. " << std::endl; 
                }

                // DisplayIterationHistory("Sparse Solve", "SparseLU", sparse_solver_.iterations(), sparse_solver_.error(), display_history_);
            }

            // 4. Backtracking Line Search
            double alpha = 1.0; 
            bool step_found = false;
            for (int i = 1; i <= MAX_LINE_SEARCH_ITER; ++i) {
                TemplateVector x_ls = x + alpha * dx; 
                double norm_rls = res_fun_(x_ls, res);

                if (norm_rls < norm_res * (1.0 - 1e-4 * alpha)) {
                    x = x_ls; 
                    DisplayResidualRatio("Line Search", i, norm_rls, norm_res, display_history_);
                    step_found = true;
                    break;
                } else {
                    alpha *= LINE_SEARCH_STEP;
                }
            }
            
            if (!step_found) {
                DisplayErrorMessage("Warning: Line search failed to find a better solution.", "Newton", display_history_);
                break; 
            }
        }

        if (!newton_convergence) {
            DisplayErrorMessage("Nonlinear", "Newton", display_history_);
        }

        return newton_convergence;
    }

private:
    ResidualFunction res_fun_;
    JacobianFunction jac_fun_;
    SpJacobianFunction sp_jac_fun_;
    bool display_history_;
    
    // Sparse Solver State
    SparseSolver sparse_solver_;
    bool solver_pattern_analyzed_;
};


/**
 * @brief A Newton Solver with customized constraints for Single-Phase/Two-Phase Flow
 * Constraints enforced:
 * 1. Pressure > 0 (Absolute)
 * 2. Quality < 1.0 (Upper Bound)
 * 3. Pressure[i] > Pressure[Outlet] (Monotonic Drop Check)
 */
template<typename T>
class FlowConstrainedNewtonSolver {

public:
    // --- Type Aliases ---
    using TemplateVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using TemplateMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using SparseMat = Eigen::SparseMatrix<T>;
    
    // Define the Sparse Solver Type
    // Using IncompleteLUT (ILU) for the preconditioner
    // using Preconditioner = Eigen::IncompleteLUT<T>;
    // using SparseSolver = Eigen::BiCGSTAB<SparseMat, Preconditioner>;
    using SparseSolver = Eigen::SparseLU<SparseMat>;

    using ResidualFunction = std::function<double(const TemplateVector& x, TemplateVector& res)>;
    using JacobianFunction = std::function<void(const TemplateVector& x, TemplateMatrix& jac)>;
    using SpJacobianFunction = std::function<void(const TemplateVector& x, SparseMat& jac)>;
    using StateValidator = std::function<bool(const TemplateVector&)>;

    // --- Constructors ---

    FlowConstrainedNewtonSolver(ResidualFunction res_fun, JacobianFunction jac_fun, bool display_history = false)
        : res_fun_(res_fun), jac_fun_(jac_fun), sp_jac_fun_(nullptr), display_history_(display_history), solver_pattern_analyzed_(false) {
        if (!res_fun_ || !jac_fun_) {
            throw std::invalid_argument("Residual and Jacobian functions must be provided.");
        }
    }

    FlowConstrainedNewtonSolver(ResidualFunction res_fun, SpJacobianFunction jac_fun, bool display_history = false)
        : res_fun_(res_fun), jac_fun_(nullptr), sp_jac_fun_(jac_fun), display_history_(display_history), solver_pattern_analyzed_(false) {
        if (!res_fun_ || !sp_jac_fun_) {
            throw std::invalid_argument("Residual and Sparse Jacobian functions must be provided.");
        }
    }

    void setValidator(StateValidator v) { validator_ = v; }

    /**
     * @brief Constraint-Aware Step Calculation
     * * Calculates the maximum alpha (0 < alpha <= 1) such that x + alpha*delta
     * respects specific physical bounds.
     */
    double compute_constraint_aware_step(const Vector& x, const Vector& delta) {
        double alpha = 1.0;
        
        // --- PROBLEM CONFIGURATION ---
        const int nf = 3; // Stride: [u, p, h]
        const int n_nodes = x.size() / nf;
        
        // Identify Last Node Indices
        int idx_p_last = (n_nodes - 1) * nf + 1;
        double P_last = std::real(x(idx_p_last));
        double dP_last = std::real(delta(idx_p_last));

        for (int i = 0; i < n_nodes; ++i) {
            int idx_p = i * nf + 1; // Pressure

            double P_curr = std::real(x(idx_p));
            double dP_curr = std::real(delta(idx_p));
            
            // ---------------------------------------------------------
            // 1. ABSOLUTE POSITIVITY: P > 0
            // ---------------------------------------------------------
            if (dP_curr < (0.0)) {
                // Prevent P from crossing 0. Target min: 1% of current value or 1e-5
                double target_min = std::max(std::abs(P_curr) * 0.01, 1.0e-5);
                double alpha_limit = (target_min - P_curr) / dP_curr;
                if (alpha_limit < alpha && alpha_limit > 0.0) alpha = alpha_limit;
            }


            // ---------------------------------------------------------
            // 3. PRESSURE DROP: P[i] > P[last] (for i < last)
            // ---------------------------------------------------------
            if (i < n_nodes - 1) {
                // We want: (P_curr + a*dP_curr) > (P_last + a*dP_last)
                // Rearranged: a * (dP_curr - dP_last) > (P_last - P_curr)
                
                double diff_P_curr = P_curr - P_last;      // Current Drop (Should be > 0)
                double diff_dP     = dP_curr - dP_last;    // Change in Drop

                // If diff_P_curr is already negative (Bad State), we rely on Line Search to fix it.
                // We only limit alpha if we are currently VALID, but the step would make us INVALID.
                if (diff_P_curr > (1e-8)) {
                    // If the step reduces the pressure drop (diff_dP < 0)
                    if (diff_dP < (0.0)) {
                        // Prevent drop from hitting 0. Keep 10% of current drop.
                        double target_drop = diff_P_curr * (0.1);
                        
                        // alpha limit = (target_drop - current_drop) / change_in_drop
                        //             = (target_drop - diff_P_curr) / diff_dP
                        double alpha_limit = (target_drop - diff_P_curr) / diff_dP;
                        
                        if (alpha_limit < alpha && alpha_limit > 0.0) alpha = alpha_limit;
                    }
                }
            }
        }

        return alpha;
    }

    /**
     * @brief Solves the nonlinear system.
     */
    bool solve(Eigen::Ref<TemplateVector> x) {
        using Integer = Eigen::Index;
        Integer nd = x.size();
        if (nd == 0) {
            std::cerr << "Error: Initial guess vector 'x' has size 0." << std::endl;
            return false;
        }

        TemplateVector res(nd);
        TemplateMatrix jac(nd, nd);
        SparseMat sp_jac(nd, nd);
        TemplateVector dx(nd);

        bool newton_convergence = false;
        double norm0 = (0.0);

        for (int iter = 1; iter <= MAX_NEWTONS_ITER; ++iter) {
            
            // 1. Calculate residual and check for convergence
            double norm_res = res_fun_(x, res);
            
            if (iter == 1) {
                norm0 = norm_res;
                if (norm0 == 0.0) norm0 = (1.0);
            }

            DisplayIterationHistory("Nonlinear", "Newton", iter, norm_res, display_history_);

            if (norm_res < NEWTON_ABS_TOL || norm_res < norm0 * NEWTON_REL_TOL) {
                DisplayConvergence("Nonlinear", "Newton", iter, norm_res, display_history_);
                newton_convergence = true;
                break;
            }

            // 2. Solve Linear System J * dx = -res
            
            // --- DENSE PATH ---
            if (jac_fun_) {
                jac_fun_(x, jac);
                
                Eigen::PartialPivLU<TemplateMatrix> lu(jac);
                // Use determinant check for compatibility with older Eigen versions
                if (std::abs(lu.determinant()) < 1.0e-12) {
                    std::cerr << "Warning: Jacobian matrix is singular in Newton iteration (Dense LU)." << std::endl;
                    break;
                }
                dx = lu.solve(-res);
            }
            // --- SPARSE PATH ---
            else {
                sp_jac_fun_(x, sp_jac);
                sp_jac.makeCompressed(); // Ensure standard CSR format

                // 3. Analyze sparse pattern
                if (iter == 1) sparse_solver_.analyzePattern(sp_jac);
                
                // 4. Factorize and Solve
                sparse_solver_.factorize(sp_jac);
                if (sparse_solver_.info() != Eigen::Success) {
                    std::cerr << "Error: Factorization failed (Singular Matrix?)" << std::endl;
                    break;
                }
                
                // 3. Solve
                dx = sparse_solver_.solve(-res);

                if (sparse_solver_.info() != Eigen::Success) {
                    std::cerr << "Warning: Sparse solver failed to converge. " << std::endl; 
                }

            }

            // --- CONSTRAINT AWARE STEP ---
            double alpha = compute_constraint_aware_step(x, dx);
            // T alpha = T(1.0);
            bool step_found = false;
            for (int i = 1; i <= MAX_LINE_SEARCH_ITER; ++i) {
                TemplateVector x_ls = x + alpha * dx; 

                if (validator_ && !validator_(x_ls)) {
                    alpha *= LINE_SEARCH_STEP;
                    continue;
                }

                double norm_rls = res_fun_(x_ls, res);

                if (norm_rls < norm_res * (1.0 - 1e-4 * alpha)) {
                    x = x_ls; 
                    DisplayResidualRatio("Line Search", i, norm_rls, norm_res, display_history_);
                    step_found = true;
                    break;
                } else {
                    alpha *= LINE_SEARCH_STEP;
                }
            }
            
            if (!step_found) {
                DisplayErrorMessage("Warning: Line search failed to find a better solution.", "Newton", display_history_);
                break; 
            }
        }

        if (!newton_convergence) {
            DisplayErrorMessage("Nonlinear", "Newton", display_history_);
        }

        return newton_convergence;
    }


private:
    ResidualFunction res_fun_;
    JacobianFunction jac_fun_;
    SpJacobianFunction sp_jac_fun_;
    StateValidator validator_;
    bool display_history_;
    
    // Sparse Solver State
    SparseSolver sparse_solver_;
    bool solver_pattern_analyzed_;    

};


} // End namespace EigenSolver

namespace mfem {

class EigenNonlinearOperator : public mfem::Operator {
    using SparseMatCol = Eigen::SparseMatrix<double, Eigen::ColMajor>;
    using SparseMatRow = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using Vector = Eigen::VectorXd;

    using ResidualFunction = std::function<void(const Vector&, Vector&)>;
    using JacobianFunction = std::function<void(const Vector&, SparseMatCol&)>;
private:
    int n_dofs;

    // NATIVE Storage (What your physics logic fills)
    mutable Vector x_eigen;
    mutable Vector r_eigen;
    mutable SparseMatCol J_eigen_col; 

    // TARGET Storage (Buffer for Hypre)
    mutable SparseMatRow J_eigen_row; 

    // MFEM/Hypre Wrappers
    mutable std::unique_ptr<mfem::HypreParMatrix> J_hypre;
    std::unique_ptr<mfem::HypreParMatrix> P_hypre;

    // User Functions
    ResidualFunction res_fun;
    JacobianFunction jac_fun;

public:
    EigenNonlinearOperator(int size, ResidualFunction res_f, JacobianFunction jac_f, SparseMatCol& P_col)
                        : mfem::Operator(size), n_dofs(size), res_fun(res_f), jac_fun(jac_f) {
        x_eigen.resize(n_dofs);
        r_eigen.resize(n_dofs);
        J_eigen_col.resize(n_dofs, n_dofs);
        
        SparseMatRow P_row = P_col;
        P_row.makeCompressed();
        P_hypre = EigenToHypreParMatrix(P_row, hypre_MPI_COMM_WORLD);
    }

    ~EigenNonlinearOperator() {}

    mfem::HypreParMatrix& GetStaticPreconditionerMatrix() {
        return *P_hypre;
    }

    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const override {
        CopyMFEMVectorToEigen(x_eigen, x);
        res_fun(x_eigen, r_eigen);
        CopyEigenToMFEMVector(y, r_eigen);
    }

    virtual mfem::Operator& GetGradient(const mfem::Vector& x) const override {
        CopyMFEMVectorToEigen(x_eigen, x);
        jac_fun(x_eigen, J_eigen_col);
        J_eigen_row = J_eigen_col;
        J_hypre = EigenToHypreParMatrix(J_eigen_row, hypre_MPI_COMM_WORLD);
        return *J_hypre;
    }

};

}


#endif // NEWTON_SOLVER_HPP