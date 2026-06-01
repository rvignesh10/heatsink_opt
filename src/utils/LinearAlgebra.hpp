#ifndef LIN_ALGEBRA_HPP
#define LIN_ALGEBRA_HPP

#include <iostream>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>

// MFEM Headers
#include "mfem.hpp"

// MPI
#include <mpi.h>

// --- Crucial: Use RowMajor for Eigen to match MFEM/Hypre CSR layout ---
using Scalar = double;
using SparseMatCol = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
using SparseMatRow = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;
using Vector = Eigen::VectorXd;

namespace mfem {

// class MfemSolverWrapper {
// public:
//     struct Options {
//         double rel_tol = 1e-8;
//         double abs_tol = 0.0;
//         int max_iters = 1000;
//         int restart = 30; // Krylov dim for FGMRES
//         int print_level = 1;
//     };

//     /**
//      * @brief Solves Ax = b using MFEM's FGMRES wrapper around Hypre.
//      */
//     static Vector solve(SparseMatCol& A_eigen_col, Vector& b_eigen, const Options& opts = Options()) {
        
//         // 1. MPI Boilerplate (Required for Hypre)
//         int num_procs, myid;
//         MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
//         MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

//         // Compress Eigen matrix to ensure CSR pointers are valid
//         SparseMatRow A_eigen = A_eigen_col;
//         A_eigen.makeCompressed(); 

//         int rows = A_eigen.rows();
//         int cols = A_eigen.cols();
//         int nnz = A_eigen.nonZeros();

//         // 2. Wrap Eigen data into MFEM SparseMatrix
//         // MFEM's SparseMatrix constructor can take existing CSR arrays.
//         // Note: Eigen uses 'int' for storage index by default, which matches MFEM.
//         // We do NOT give ownership to MFEM (last 3 params: false, false, true for verify)
//         mfem::SparseMatrix mfem_A(
//             A_eigen.outerIndexPtr(), 
//             A_eigen.innerIndexPtr(), 
//             A_eigen.valuePtr(), 
//             rows, cols, false, false, true
//         );

//         // 3. Convert to HypreParMatrix
//         // Even if running on 1 core, Hypre solvers require the Parallel Matrix format.
//         // We split the rows evenly among processors (trivial if 1 proc).
//         mfem::HypreParMatrix par_A(hypre_MPI_COMM_WORLD, &mfem_A, NULL, NULL);

//         // 4. Wrap RHS and Solution vectors
//         // MFEM Vector wraps the raw double pointer from Eigen
//         mfem::Vector mfem_b(b_eigen.data(), rows);
        
//         Vector x_eigen = Vector::Zero(rows);
//         mfem::Vector mfem_x(x_eigen.data(), rows);
        
//         // Promote vectors to HypreParVectors matches the par_A partitioning
//         mfem::HypreParVector par_b(hypre_MPI_COMM_WORLD, mfem_A.GetGlobalNumRows(), mfem_b.GetData(), par_A.GetRowPart());
//         mfem::HypreParVector par_x(hypre_MPI_COMM_WORLD, mfem_A.GetGlobalNumRows(), mfem_x.GetData(), par_A.GetRowPart());

//         // 5. Setup the Preconditioner (BoomerAMG)
//         // This is the "nuclear option" of preconditioners - handles almost anything.
//         mfem::HypreBoomerAMG amg(par_A);
//         amg.SetPrintLevel(0);
//         amg.SetSystemsOptions(1); // Helper if system of equations (e.g. vector field)

//         // 6. Setup MFEM FGMRES Solver
//         mfem::FGMRESSolver fgmres(hypre_MPI_COMM_WORLD);
//         fgmres.SetKDim(opts.restart);
//         fgmres.SetMaxIter(opts.max_iters);
//         fgmres.SetRelTol(opts.rel_tol);
//         fgmres.SetAbsTol(opts.abs_tol);
//         fgmres.SetPrintLevel(opts.print_level);
        
//         // Link operator and preconditioner
//         fgmres.SetPreconditioner(amg);
//         fgmres.SetOperator(par_A);

//         // 7. Solve
//         // The result is written directly into par_x, which wraps mfem_x, which wraps x_eigen.
//         // Zero-copy magic!
//         fgmres.Mult(par_b, par_x);

//         // Ensure we check for convergence
//         if (!fgmres.GetConverged()) {
//             std::cerr << "FGMRES did not converge!" << std::endl;
//         }

//         return x_eigen;
//     }
// };

// class SplitPreconditionedSolver {
// public:
//     struct Options {
//         double rel_tol = 1e-8;
//         int max_iters = 1000;
//         int restart = 30; 
//         int print_level = 1;
//     };

//     /**
//      * @brief Solves Ax = b, using Matrix P to build the preconditioner.
//      * @param A_eigen The exact system matrix (Operator).
//      * @param P_eigen The approximate matrix (Preconditioner Basis).
//      * @param b_eigen The RHS.
//      */
//     static Vector solve(SparseMatCol& A_eigen_col, SparseMatCol& P_eigen_col, Vector& b_eigen, const Options& opts = Options()) {
        
//         int num_procs;
//         MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

//         // 1. Compress both matrices to ensure CSR arrays are packed and valid
//         SparseMatRow A_eigen = A_eigen_col;
//         SparseMatRow P_eigen = P_eigen_col;
//         A_eigen.makeCompressed(); 
//         P_eigen.makeCompressed();

//         int rows = A_eigen.rows();

//         // 2. Wrap Eigen A (The Operator)
//         mfem::SparseMatrix mfem_A(
//             A_eigen.outerIndexPtr(), A_eigen.innerIndexPtr(), A_eigen.valuePtr(), 
//             rows, A_eigen.cols(), false, false, true
//         );
//         mfem::HypreParMatrix par_A(hypre_MPI_COMM_WORLD, &mfem_A, NULL, NULL);

//         // 3. Wrap Eigen P (The Preconditioner Matrix)
//         mfem::SparseMatrix mfem_P(
//             P_eigen.outerIndexPtr(), P_eigen.innerIndexPtr(), P_eigen.valuePtr(), 
//             rows, P_eigen.cols(), false, false, true
//         );
//         mfem::HypreParMatrix par_P(hypre_MPI_COMM_WORLD, &mfem_P, NULL, NULL);

//         // 4. Setup Vectors
//         mfem::Vector mfem_b(b_eigen.data(), rows);
//         Vector x_eigen = Vector::Zero(rows);
//         mfem::Vector mfem_x(x_eigen.data(), rows);
        
//         mfem::HypreParVector par_b(hypre_MPI_COMM_WORLD, mfem_A.GetGlobalNumRows(), mfem_b.GetData(), par_A.GetRowPart());
//         mfem::HypreParVector par_x(hypre_MPI_COMM_WORLD, mfem_A.GetGlobalNumRows(), mfem_x.GetData(), par_A.GetRowPart());

//         // ---------------------------------------------------------
//         // 5. The Magic: Build AMG on P, but Solve for A
//         // ---------------------------------------------------------
        
//         // We initialize BoomerAMG using par_P (The approximate matrix)
//         // Hypre will analyze P to determine coarse grids and smoothing weights.
//         mfem::HypreBoomerAMG amg(par_P);
//         amg.SetPrintLevel(0);
//         amg.SetSystemsOptions(1); 

//         // 6. Setup FGMRES
//         mfem::FGMRESSolver fgmres(hypre_MPI_COMM_WORLD);
//         fgmres.SetKDim(opts.restart);
//         fgmres.SetMaxIter(opts.max_iters);
//         fgmres.SetRelTol(opts.rel_tol);
//         fgmres.SetPrintLevel(opts.print_level);
        
//         // Connect the parts:
//         fgmres.SetPreconditioner(amg); // Use P's hierarchy to precondition
//         fgmres.SetOperator(par_A);     // Use A for the Matrix-Vector multiply (Calculates true residual)

//         // 7. Solve
//         fgmres.Mult(par_b, par_x);

//         if (!fgmres.GetConverged()) {
//             std::cerr << "Warning: FGMRES did not converge." << std::endl;
//         }

//         return x_eigen;
//     }
// };

class FrozenPreconditioner : public mfem::Solver {
private:
    mfem::Solver& real_preconditioner;
public:
    // Constructor takes the real AMG (which is already set up with P)
    FrozenPreconditioner(mfem::Solver& prec) 
        : mfem::Solver(prec.Height()), real_preconditioner(prec) { 
        // Important: Inherit iterative_mode from parent to ensure correct solver behavior
        this->iterative_mode = prec.iterative_mode;
    }

    // 1. THE TRICK: Ignore the update!
    // FGMRES will try to pass J here. We simply do nothing.
    virtual void SetOperator(const mfem::Operator& op) override {
        // Do NOT call real_preconditioner.SetOperator(op);
        // We want the inner preconditioner to stay stuck on Matrix P.
    }

    // 2. Forward the Solve call
    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const override {
        real_preconditioner.Mult(x, y);
    }
};

}



#endif // LIN_ALGEBRA_HPP

// int main(int argc, char *argv[]) {
//     // Initialize MPI (Required for MFEM/Hypre)
//     MPI_Init(&argc, &argv);
    
//     // --- Create a Test Problem in Eigen ---
//     int n = 100;
//     SparseMat A(n, n);
//     Vector b = Vector::Ones(n);

//     // Simple 1D Laplacian stencil
//     std::vector<Eigen::Triplet<Scalar>> tri;
//     for(int i=0; i<n; ++i) {
//         tri.push_back({i, i, 2.0});
//         if(i>0) tri.push_back({i, i-1, -1.0});
//         if(i<n-1) tri.push_back({i, i+1, -1.0});
//     }
//     A.setFromTriplets(tri.begin(), tri.end());

//     // --- Solve using MFEM/Hypre Wrapper ---
//     try {
//         std::cout << "Starting MFEM FGMRES..." << std::endl;
        
//         MfemSolverWrapper::Options opts;
//         opts.rel_tol = 1e-12;
        
//         Vector x = MfemSolverWrapper::solve(A, b, opts);

//         std::cout << "Solution head: " << x.head(5).transpose() << std::endl;
//         std::cout << "Residual norm: " << (A*x - b).norm() << std::endl;

//     } catch (std::exception& e) {
//         std::cout << "Error: " << e.what() << std::endl;
//     }

//     MPI_Finalize();
//     return 0;
// }