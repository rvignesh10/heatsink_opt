#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstring>
#include "mfem.hpp"
#include "HYPRE.h"
#include "_hypre_parcsr_mv.h"
#include <cassert> // For assert
#include <complex>
#include <cmath>
#include <type_traits>
#include <limits>

// Type alias for clarity, similar to np.ndarray
using Vector = Eigen::VectorXd;
using Triplet = Eigen::Triplet<double>;
using SparseMatRow = Eigen::SparseMatrix<double, Eigen::RowMajor>;

/**
 * @brief Performs 1-D trapezoidal integration of a function f(z).
 * @param fval An Eigen vector containing the function values at several z locations.
 * @param dz An Eigen vector containing the distance between the z locations.
 * @return The result of the integration \int_{z=a}^{z=b}{f(z) dz}.
 */
inline double trapz1D(const Vector& fval, const Vector& dz) {
    assert(fval.size() == dz.size() && "Function value and discretization length mismatch");
    long n = fval.size(); // .size() returns a long integer in Eigen
    
    double integral = 0.0;

    for (long i = 0; i < n - 1; ++i) {
        double fmean = 0.5 * (fval(i) + fval(i + 1));
        double h = dz(i); // Assuming dz(i) is the width of the i-th segment
        integral += fmean * h;
    }
    
    return integral;
}


/**
 * @brief A continuous and smooth absolute value function |x|_s.
 * Inaccurate near x=0 where it is smoothed.
 * @param x The value to evaluate the function at.
 * @return The smoothed absolute value.
 */
inline double sabs(double x) {
    constexpr double eps = 1e-10;
    if (std::abs(x) < eps) {
        return (0.5 / eps) * std::pow(x, 2.0) + (0.5 * eps);
    } else if (x >= eps) {
        return x;
    } else {
        // x < -eps
        return -x;
    }
}

/**
 * @brief Calculates the derivative of the smooth absolute value function, d|x|_s/dx.
 * @param x The value at which the derivative is evaluated.
 * @return The first derivative with respect to x.
 */
inline double Dsabs(double x) {
    constexpr double eps = 1e-10;
    if (std::abs(x) < eps) {
        return x / eps;
    } else if (x >= eps) {
        return 1.0;
    } else {
        // x < -eps
        return -1.0;
    }
}

template <typename T>
inline T complex_abs(T val) {
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        return (val.real() >= 0.0) ? val : -val;
    } else {
        using std::abs;
        return abs(val);
    }
}

// Helper to print Real and Imaginary parts cleanly
template <typename T>
inline void print_term(std::string name, T val) {
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        std::cout << name << ": " << std::setw(12) << val.real() 
                    << " | Imag (Sens): " << std::setw(12) << val.imag() << "\n";
    } else {
        std::cout << name << ": " << std::setw(12) << val << "\n";
    }
}

/**
 * @brief A blending or smooth step function.
 * @return The blended value, between 0 and 1.
 * Returns 0 if x << x0 (left of transition)
 * Returns 1 if x >> x0 (right of transition)
 */
template <typename T>
inline T blend(T x, T x0, T k)
{
    using std::exp;
    using std::real; 

    // Prevent division by zero
    constexpr double k_eps = 1e-12;
    const T k_safe = k + static_cast<T>(k_eps);
    
    // v = -(x - x0) / k
    const T v = -(x - x0) / k_safe;

    // --- FIX: Unified Clamping for Double AND Complex ---
    // We check the real part of the exponent. 
    // If Re(v) > 700, exp(v) overflows to Infinity.
    // If Re(v) < -700, exp(v) underflows to Zero.
    
    double v_real = std::real(v); 

    // Case 1: x << x0. v is huge positive. exp(v) is Inf. 
    // Result = 1 / (1 + Inf) = 0.0
    if (v_real > 700.0) {
        return static_cast<T>(0.0);
    }

    // Case 2: x >> x0. v is huge negative. exp(v) is 0. 
    // Result = 1 / (1 + 0) = 1.0
    if (v_real < -700.0) {
        return static_cast<T>(1.0);
    }
    // ---------------------------------------------------

    const T exp_val = exp(v);
    return static_cast<T>(1.0) / (static_cast<T>(1.0) + exp_val);
}


// /**
//  * @brief A blending or smooth step function.
//  * @param x The input value.
//  * @param x0 The center of the blend (transition point).
//  * @param k The steepness/width of the blend.
//  * @return The blended value, between 0 and 1.
//  */
// template <typename T>
// inline T blend(T x, T x0, T k)
// {
//     using std::exp;
//     using std::real;
//     using std::isnan;

//     constexpr double k_eps = 1e-12;
//     const T k_safe = k + static_cast<T>(k_eps);
//     const T val = -(x - x0) / k_safe;

//     // For real types, clamp to avoid overflow
//     if constexpr (std::is_floating_point_v<T>) {
//         if (val > 700.0) return 0.0;
//         if (val < -700.0) return 1.0;
//     }

//     const T exp_val = exp(val);
//     const T result = static_cast<T>(1.0) / (static_cast<T>(1.0) + exp_val);


//     if constexpr (std::is_floating_point_v<T>) {
//         return (isnan(result)) ? 0.0 : result;
//     } else {
//         return result;
//     }
// }

template <typename T>
inline T Dblend(T x, T x0, T k)
{
    // Prevent division by zero in k
    constexpr double k_eps = 1e-12;
    const T k_safe = k + static_cast<T>(k_eps);
    const T Dv = -static_cast<T>(1.0) / k_safe; // Chain rule factor d(v)/dx

    // 1. Calculate the sigmoid value using the robust blend function
    // This handles all the exp() overflow checks internally.
    T sigma = blend(x, x0, k);

    // 2. Use the Sigmoid Derivative Identity: d(sig)/dv = sig * (1 - sig)
    // Chain rule: d(sig)/dx = d(sig)/dv * dv/dx
    T result = -sigma * (static_cast<T>(1.0) - sigma) * Dv;

    return result;
}


// template <typename T>
// inline T Dblend(T x, T x0, T k)
// {
//     using std::exp;
//     using std::real; // Import std::real to handle both complex and scalar

//     constexpr double k_eps = 1e-12;
//     const T k_safe = k + static_cast<T>(k_eps);

//     const T v = -(x - x0) / k_safe;
//     const T Dv = -static_cast<T>(1.0) / k_safe;

//     // --- FIX: Safe Check for BOTH double and complex ---
//     // Extract the real part to check for exponential overflow
//     if (std::abs(real(v)) > 700.0) {
//         return static_cast<T>(0.0);
//     }
//     // ---------------------------------------------------

//     const T exp_v = exp(v);
//     const T den = static_cast<T>(1.0) + exp_v;
//     const T result = (-exp_v / (den * den)) * Dv;

//     return result;
// }

// /**
//  * @brief Calculates the derivative of the blend function.
//  * @param x The input value.
//  * @param x0 The center of the blend.
//  * @param k The steepness/width of the blend.
//  * @return The derivative of the blend function.
//  */
// template <typename T>
// inline T Dblend(T x, T x0, T k)
// {
//     using std::exp;
//     using std::isnan;

//     constexpr double k_eps = 1e-12;
//     const T k_safe = k + static_cast<T>(k_eps);

//     const T v = -(x - x0) / k_safe;
//     const T Dv = -static_cast<T>(1.0) / k_safe;

//     // Skip clipping for complex types
//     if constexpr (std::is_floating_point_v<T>) {
//         if (v > 700.0 || v < -700.0)
//             return static_cast<T>(0.0);
//     }

//     const T exp_v = exp(v);
//     const T den = static_cast<T>(1.0) + exp_v;
//     const T result = (-exp_v / (den * den)) * Dv;

//     if constexpr (std::is_floating_point_v<T>) {
//         return (isnan(result)) ? 0.0 : result;
//     } else {
//         return result;
//     }
// }


/**
 * @brief Copies data from an Eigen::VectorXd into an mfem::GridFunction.
 * * This is the preferred, efficient method using Eigen::Map to view the
 * GridFunction's memory and perform a deep copy from the Eigen vector.
 * @param gf_ptr A unique_ptr to the destination mfem::GridFunction.
 * @param eigen_vec The source Eigen::VectorXd containing the data to copy.
 */
inline void CopyEigenToMFEMGridFunction(std::unique_ptr<mfem::GridFunction>& gf_ptr, 
                                        const Eigen::Ref<const Eigen::VectorXd>& eigen_vec)
{
    // Ensure the GridFunction pointer is valid
    assert(gf_ptr && "GridFunction pointer must be initialized.");

    // 1. Safety Check: Ensure dimensions match
    MFEM_ASSERT(gf_ptr->Size() == eigen_vec.size(), 
        "GridFunction size (" << gf_ptr->Size() << 
        ") does not match Eigen vector size (" << eigen_vec.size() << ")!");

    // 2. Create an Eigen::Map that "views" the GridFunction's memory (no copy).
    //    We use the raw pointer (GetData()) from the GridFunction.
    Eigen::Map<Eigen::VectorXd> gf_map(gf_ptr->GetData(), gf_ptr->Size());

    // 3. Assign the Eigen vector to the map.
    //    This performs the deep copy of data from 'eigen_vec' into the 
    //    GridFunction's memory.
    gf_map = eigen_vec;
}

/**
 * @brief Copies data from an Eigen::VectorXd into an existing mfem::Vector.
 * * This uses Eigen::Map to efficiently view the target mfem::Vector's memory
 * and performs a deep copy from the Eigen source.
 * * NOTE: The mfem::Vector must be pre-sized before calling this function.
 * @param v_mfem The destination mfem::Vector (must be pre-sized).
 * @param eigen_vec The source Eigen::VectorXd containing the data to copy.
 */
inline void CopyEigenToMFEMVector(mfem::Vector& v_mfem, 
                                 const Eigen::Ref<const Eigen::VectorXd>& eigen_vec) 
{
    // 1. Safety Check: Ensure dimensions match
    MFEM_ASSERT(v_mfem.Size() == eigen_vec.size(), 
        "mfem::Vector size (" << v_mfem.Size() << 
        ") does not match Eigen vector size (" << eigen_vec.size() << ")!");

    // 2. Create an Eigen::Map that "views" the mfem Vector's memory (no copy).
    //    We use the raw pointer (GetData()) from the mfem Vector.
    Eigen::Map<Eigen::VectorXd> mfem_vec_map(v_mfem.GetData(), v_mfem.Size());

    // 3. Assign the Eigen vector to the map.
    //    This performs the deep copy of data from 'eigen_vec' into the 
    //    mfem Vector's memory.
    mfem_vec_map = eigen_vec;
}

/**
 * @brief Copies data from an mfem::GridFunction into an Eigen::VectorXd.
 * * This function handles resizing the destination Eigen vector and uses 
 * Eigen::Map for an efficient deep copy.
 * @param eigen_vec The destination Eigen::VectorXd (will be resized).
 * @param gf The source mfem::GridFunction.
 */
inline void CopyMFEMGridFunctionToEigen(Eigen::Ref<Eigen::VectorXd> eigen_vec, const mfem::GridFunction& gf)
{
    // 1. Resize the destination Eigen vector to match the GridFunction size.
    eigen_vec.resize(gf.Size());
    
    // 2. Create an Eigen::Map that "views" the GridFunction's data.
    //    We use GetData() to get the raw pointer and Size() for the length.
    //    The 'const' cast is necessary because GetData() returns a non-const pointer, 
    //    but we are only reading the data here.
    Eigen::Map<const Eigen::VectorXd> gf_map(gf.GetData(), gf.Size());

    // 3. Assign the map to the Eigen vector.
    //    This performs a deep copy of the data into the now-resized 'eigen_vec'.
    eigen_vec = gf_map;
}

/**
 * @brief Copies data from an mfem::Vector into an Eigen::VectorXd.
 * * This function handles resizing the destination Eigen vector and uses 
 * Eigen::Map for an efficient deep copy.
 * @param eigen_vec The destination Eigen::VectorXd (will be resized).
 * @param v_mfem The source mfem::Vector.
 */
inline void CopyMFEMVectorToEigen(Eigen::Ref<Eigen::VectorXd> eigen_vec, const mfem::Vector& v_mfem)
{
    // 1. Resize the destination Eigen vector to match the Vector size.
    // eigen_vec.resize(v_mfem.Size());
    assert(v_mfem.Size() == eigen_vec.size() && "size mismatch between mfem vector and eigen vector");
    
    // 2. Create an Eigen::Map that "views" the mfem::Vector's data.
    //    We use GetData() to get the raw pointer and Size() for the length.
    //    The 'const' cast is necessary because GetData() returns a non-const pointer, 
    //    but we are only reading the data here.
    Eigen::Map<const Eigen::VectorXd> mfem_vec_map(v_mfem.GetData(), v_mfem.Size());

    // 3. Assign the map to the Eigen vector.
    //    This performs a deep copy of the data into the now-resized 'eigen_vec'.
    eigen_vec = mfem_vec_map;
}

/**
 * @brief Copies over MFEM SparseMatrix to Eigen SparseMatrix
 * @param A_mfem Sparse BilinearForm Assembled matrix
 * @param A Eigen SparseMatrix
 */
inline void CopyMFEMSparseToEigen(const mfem::SparseMatrix& A_mfem, Eigen::SparseMatrix<double>& A) {
    const int nrows = A_mfem.Height();
    const int ncols = A_mfem.Width();

    const int* I      = A_mfem.GetI();      // row offsets (size nrows+1)
    const int* J      = A_mfem.GetJ();      // col indices (size nnz)
    const double* val = A_mfem.GetData();   // values (size nnz)

    std::vector<Triplet> triplets;
    triplets.reserve(I[nrows]);        // nnz

    for (int i = 0; i < nrows; ++i) {
        const int row_start = I[i];
        const int row_end   = I[i+1];
        for (int k = row_start; k < row_end; ++k) {
            const int    j = J[k];
            const double v = val[k];
            if (v != 0.0) {
                triplets.emplace_back(i, j, v);
            }
        }
    }

    A.resize(nrows, ncols);
    A.setFromTriplets(triplets.begin(), triplets.end());
}


// Templated to accept both ColMajor (default) and RowMajor Eigen matrices
template <typename Scalar, int Options, typename StorageIndex>
inline void CopyEigenSparseToMFEM(const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& A_eigen, 
                           mfem::SparseMatrix& A_mfem) 
{
    // 1. Convert to RowMajor (CSR)
    //    If A_eigen is already RowMajor, this is a cheap copy.
    //    If A_eigen is ColMajor (default), this performs the structural Transpose 
    //    needed to make the memory layout compatible with MFEM.
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor, int> A_row = A_eigen;

    // 2. Compress to ensure no unused space in vectors
    A_row.makeCompressed();

    int rows = A_row.rows();
    int cols = A_row.cols();
    int nnz = A_row.nonZeros();

    // 3. Allocate buffers for MFEM
    //    MFEM expects to own these arrays (via the constructor/MakeRef we will use),
    //    so we allocate them on the heap using 'new'.
    int* I = new int[rows + 1];
    int* J = new int[nnz];
    double* data = new double[nnz];

    // 4. Copy Data (Fast memcpy)
    //    We can now copy directly because A_row is guaranteed to be CSR.
    
    // Copy Row Offsets (Outer Index)
    std::memcpy(I, A_row.outerIndexPtr(), (rows + 1) * sizeof(int));
    
    // Copy Col Indices (Inner Index)
    std::memcpy(J, A_row.innerIndexPtr(), nnz * sizeof(int));
    
    // Copy Values
    std::memcpy(data, A_row.valuePtr(), nnz * sizeof(double));

    // 5. Populate MFEM Matrix
    //    We create a temporary wrapper that takes OWNERSHIP of the pointers 
    //    (true, true, true), and then Swap it into the destination.
    //    This is the safest way to replace an existing MFEM matrix.
    mfem::SparseMatrix temp(I, J, data, rows, cols, true, true, true);
    
    // Swap contents (A_mfem gets the new data, temp gets the old data and destroys it)
    A_mfem.Swap(temp);
}



inline std::unique_ptr<mfem::HypreParMatrix> EigenToHypreParMatrix(const SparseMatRow& eigen_mat, MPI_Comm comm) {
    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // 1. Get raw pointers from Eigen
    const int* I = eigen_mat.outerIndexPtr(); // Row offsets
    const int* J = eigen_mat.innerIndexPtr(); // Col indices
    const double* data = eigen_mat.valuePtr(); // Values
    
    HYPRE_BigInt local_rows = eigen_mat.rows();

    // 2. Calculate Global Offsets (Partitioning)
    HYPRE_BigInt row_start = 0;
    
    std::vector<HYPRE_BigInt> all_sizes(num_procs);
    long long my_size_long = local_rows;
    MPI_Allgather(&my_size_long, 1, MPI_LONG_LONG, all_sizes.data(), 1, MPI_LONG_LONG, comm);

    for (int i = 0; i < myid; ++i) row_start += all_sizes[i];
    HYPRE_BigInt row_end = row_start + local_rows - 1;

    // 3. Initialize Hypre IJ Matrix
    HYPRE_IJMatrix ij_matrix;
    HYPRE_IJMatrixCreate(comm, row_start, row_end, row_start, row_end, &ij_matrix);
    HYPRE_IJMatrixSetObjectType(ij_matrix, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(ij_matrix);

    // 4. Transfer Data
    for (int r = 0; r < local_rows; ++r) {
        HYPRE_BigInt global_row = row_start + r;
        int row_size = I[r+1] - I[r]; 

        std::vector<HYPRE_BigInt> global_cols(row_size);
        std::vector<double> values(row_size);

        for (int k = 0; k < row_size; ++k) {
            int idx = I[r] + k; 
            global_cols[k] = row_start + J[idx]; 
            values[k] = data[idx];
        }

        HYPRE_IJMatrixSetValues(ij_matrix, 1, &row_size, &global_row, global_cols.data(), values.data());
    }

    // 5. Finalize Assembly
    HYPRE_IJMatrixAssemble(ij_matrix);

    // 6. Extract ParCSR Object
    hypre_ParCSRMatrix* par_csr = NULL;
    HYPRE_IJMatrixGetObject(ij_matrix, (void**)&par_csr);

    // // 7. Detach and Destroy IJ Wrapper
    // // FIX: Set the object to NULL inside the wrapper before destroying it.
    // // This tells Hypre "The wrapper no longer owns any data".
    // // Then we can safely destroy the wrapper without touching 'par_csr'.
    // HYPRE_IJMatrixSetObject(ij_matrix, NULL); 

    // // Now it is safe to destroy the wrapper (fixes the struct leak)
    // HYPRE_IJMatrixDestroy(ij_matrix); 

    // 8. Return wrapped in MFEM object (MFEM owns the memory now)
    return std::make_unique<mfem::HypreParMatrix>(par_csr, true);
}

/**
 * @brief groups Triplets by row_id and creates a new vector out of it
 * @param trips is the list of Triplets that are ungrouped.
 * @param nrows is the num of rows we know exist
 * @return a vector of <vector of Triplets> where the <vector of Triplets> have the same row ID. 
 */
std::vector<std::vector<Triplet>>
inline groupTripletsByRow(const std::vector<Triplet>& trips, int nrows)
{
    std::vector<std::vector<Triplet>> by_row(nrows);

    for (const auto& t : trips) {
        int r = t.row();
        // optionally assert 0 <= r < nrows
        by_row[r].push_back(t);
    }

    return by_row;
}

inline std::vector<Triplet> EigenSparsetoTriplets(const Eigen::SparseMatrix<double>& A)
{
    std::vector<Triplet> trips;
    trips.reserve(A.nonZeros());   // avoid reallocations

    for (int outer = 0; outer < A.outerSize(); ++outer) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, outer); it; ++it) {
            trips.emplace_back(it.row(), it.col(), it.value());
        }
    }
    return trips;
}

#endif // UTILS_HPP


