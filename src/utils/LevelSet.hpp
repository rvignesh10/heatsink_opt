// #ifndef LEVELSET_HPP
// #define LEVELSET_HPP

// #include <Eigen/Dense>
// #include <vector>
// #include <memory>
// #include <complex>
// #include <stdexcept>
// #include <algorithm>
// #include <cmath>
// #include "nanoflann.hpp" // Assumes nanoflann is in the include path

// // // The point cloud adapter remains double-based as the geometry is real.
// // struct PointCloud {
// //     std::vector<Eigen::VectorXd> pts;
// //     inline size_t kdtree_get_point_count() const { return pts.size(); }
// //     inline double kdtree_get_pt(const size_t idx, const size_t dim) const { return pts[idx][dim]; }
// //     template <class BBOX>
// //     bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
// // };

// // The LevelSet class is templated on the scalar type 'T'.
// template <typename T>
// class LevelSet {
// public:
//     // Constructor uses double for geometry setup.
//     LevelSet(int dim,
//              const Eigen::MatrixXd& xcenter,
//              const Eigen::MatrixXd& normal,
//              const std::vector<Eigen::MatrixXd>& tangents,
//              double rho,
//              double normscale = 1000.0);

//     // "Copy" constructor to create a version with a new template type
//     // (e.g., creating LevelSet<std::complex<double>> from LevelSet<double>).
//     template <typename U>
//     LevelSet(const LevelSet<U>& other);

//     // Public evaluation methods operate on the template type T.
//     T evallevelset(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const;
//     Eigen::Matrix<T, Eigen::Dynamic, 1> difflevelset(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const;
//     T evallevelset_normalized(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const;
//     Eigen::Matrix<T, Eigen::Dynamic, 1> difflevelset_normalized(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const;

// public:
//     // Geometry / config (always stored in double space)
//     int dim;
//     int numbasis;
//     Eigen::MatrixXd xc;
//     std::vector<Eigen::MatrixXd> frame;
//     double delta = 1e-10;
//     double rho;
//     double normscale;

//     // // KD-Tree for nearest neighbor search (in real space).
//     // using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
//     //     nanoflann::L2_Simple_Adaptor<double, PointCloud>,
//     //     PointCloud,
//     //     -1 /* dim */
//     // >;
//     // PointCloud cloud;
//     // std::unique_ptr<my_kd_tree_t> tree;

// private:
//     // Private helper methods
//     // std::pair<std::vector<size_t>, double> getnearest(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const;

//     T locallevelset(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
//                     const Eigen::VectorXd& xc,
//                     const Eigen::MatrixXd& local_frame,
//                     const Eigen::VectorXd& kappa) const;

//     Eigen::Matrix<T, Eigen::Dynamic, 1> difflocallevelset(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
//                                                           const Eigen::VectorXd& xc,
//                                                           const Eigen::MatrixXd& local_frame,
//                                                           const Eigen::VectorXd& kappa,
//                                                           T perp_bar) const;

//     T expdist(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
//               const Eigen::VectorXd& xc,
//               T min_dist) const;

//     Eigen::Matrix<T, Eigen::Dynamic, 1> diffexpdist(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
//                                                     const Eigen::VectorXd& xc,
//                                                     T min_dist,
//                                                     T exp_bar) const;
// };


// // ========================= IMPLEMENTATION =========================

// template <typename T>
// LevelSet<T>::LevelSet(int p_dim,
//                       const Eigen::MatrixXd& xcenter,
//                       const Eigen::MatrixXd& normal,
//                       const std::vector<Eigen::MatrixXd>& tangents,
//                       double p_rho,
//                       double p_normscale)
//     : dim(p_dim),
//       numbasis(xcenter.cols()),
//       xc(xcenter),
//       rho(p_rho),
//       normscale(p_normscale)
// {
//     if (xcenter.cols() != normal.cols() ||
//         (tangents.size() > 0 && xcenter.cols() != tangents.size())) {
//         throw std::invalid_argument("x-centers, normals, and tangents must have the same number of columns.");
//     }
//     if (xcenter.rows() != dim) {
//         throw std::invalid_argument("x-center dimension does not match problem dimension.");
//     }

//     // Build orthonormal frames
//     frame.resize(numbasis, Eigen::MatrixXd(dim, dim));
//     for (int i = 0; i < numbasis; ++i) {
//         // first column = normalized normal
//         frame[i].col(0) = normal.col(i).normalized();

//         // tangent Gram-Schmidt
//         for (int j = 0; j < dim - 1; ++j) {
//             frame[i].col(j + 1) = tangents[i].col(j);
//             for (int k = 0; k < j + 1; ++k) {
//                 frame[i].col(j + 1) -=
//                     (frame[i].col(j + 1).dot(frame[i].col(k))) * frame[i].col(k);
//             }
//             frame[i].col(j + 1).normalize();
//         }
//     }

//     // // KD-tree build (geometry in real space)
//     // cloud.pts.resize(numbasis);
//     // for (int i = 0; i < numbasis; ++i) {
//     //     cloud.pts[i] = xc.col(i);
//     // }

//     // tree = std::make_unique<my_kd_tree_t>(
//     //     dim,
//     //     cloud,
//     //     nanoflann::KDTreeSingleIndexAdaptorParams(10)
//     // );
//     // tree->buildIndex();
// }

// template <typename T>
// template <typename U>
// LevelSet<T>::LevelSet(const LevelSet<U>& other)
//     : dim(other.dim),
//       numbasis(other.numbasis),
//       xc(other.xc),
//       frame(other.frame),
//       rho(other.rho),
//       normscale(other.normscale)
// {
//     // Rebuild KD-tree for this instance
//     // cloud.pts.resize(numbasis);
//     // for (int i = 0; i < numbasis; ++i) {
//     //     cloud.pts[i] = xc.col(i);
//     // }

//     // tree = std::make_unique<my_kd_tree_t>(
//     //     dim,
//     //     cloud,
//     //     nanoflann::KDTreeSingleIndexAdaptorParams(10)
//     // );
//     // tree->buildIndex();
// }

// // template <typename T>
// // std::pair<std::vector<size_t>, double>
// // LevelSet<T>::getnearest(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const
// // {
// //     // KDTree runs in real space. For complex T, ignore imaginary part.
// //     Eigen::VectorXd x_real;
// //     if constexpr (std::is_same_v<T, double>) {
// //         x_real = x;
// //     } else {
// //         x_real = x.real();
// //     }

// //     const size_t num_results = std::min(20, static_cast<int>(numbasis));
// //     std::vector<size_t> ret_index(num_results);
// //     std::vector<double> out_dist_sqr(num_results);

// //     nanoflann::KNNResultSet<double> resultSet(num_results);
// //     resultSet.init(ret_index.data(), out_dist_sqr.data());
// //     tree->findNeighbors(resultSet, x_real.data(), nanoflann::SearchParameters(10));

// //     return {ret_index, std::sqrt(out_dist_sqr[0])};
// // }

// template <typename T>
// T LevelSet<T>::locallevelset(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
//                              const Eigen::VectorXd& p_xc,
//                              const Eigen::MatrixXd& local_frame,
//                              const Eigen::VectorXd& kappa) const
// {
//     using std::pow;

//     // normal direction contribution
//     T perp = local_frame.col(0).template cast<T>().dot(x - p_xc.template cast<T>());

//     // curvature / quadratic contributions in tangent directions
//     for (int j = 0; j < dim - 1; ++j) {
//         T proj = local_frame.col(j + 1).template cast<T>().dot(x - p_xc.template cast<T>());
//         // perp += T(0.5) * T(kappa(j)) * pow(proj, 2.0);
//         perp += T(0.5) * T(kappa(j)) * proj * proj;
//     }

//     return perp;
// }

// template <typename T>
// Eigen::Matrix<T, Eigen::Dynamic, 1>
// LevelSet<T>::difflocallevelset(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
//                                const Eigen::VectorXd& p_xc,
//                                const Eigen::MatrixXd& local_frame,
//                                const Eigen::VectorXd& kappa,
//                                T perp_bar) const
// {
//     // gradient wrt x of locallevelset, scaled by adjoint perp_bar
//     Eigen::Matrix<T, Eigen::Dynamic, 1> x_bar =
//         Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(dim);

//     // tangent dirs
//     for (int j = 0; j < dim - 1; ++j) {
//         T proj = local_frame.col(j + 1).template cast<T>().dot(x - p_xc.template cast<T>());
//         x_bar += perp_bar * T(kappa(j)) * proj *
//                  local_frame.col(j + 1).template cast<T>();
//     }

//     // normal dir
//     x_bar += perp_bar * local_frame.col(0).template cast<T>();

//     return x_bar;
// }

// template <typename T>
// T LevelSet<T>::expdist(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
//                        const Eigen::VectorXd& local_xc,
//                        T min_dist) const
// {
//     using std::sqrt;
//     using std::exp;

//     // T dist_sq = (x - local_xc.template cast<T>()).squaredNorm();
//     T dist_sq = (x - local_xc.template cast<T>()).dot(x - local_xc.template cast<T>());
//     T dist    = sqrt(dist_sq + T(delta));

//     return exp(-T(rho) * (dist - T(min_dist)));
// }

// template <typename T>
// Eigen::Matrix<T, Eigen::Dynamic, 1>
// LevelSet<T>::diffexpdist(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
//                          const Eigen::VectorXd& local_xc,
//                          T min_dist,
//                          T exp_bar) const
// {
//     using std::sqrt;
//     using std::exp;

//     // T diff_vec_norm_sq = (x - local_xc.template cast<T>()).squaredNorm();
//     T diff_vec_norm_sq = (x - local_xc.template cast<T>()).dot(x - local_xc.template cast<T>());
//     T dist             = sqrt(diff_vec_norm_sq + T(delta));

//     // derivative of exp(-rho*(dist - min_dist)) wrt x
//     // chain rule: d/dx = (-rho * exp(...)) * d(dist)/dx
//     T dist_bar = -exp_bar * T(rho) *
//                  exp(-T(rho) * (dist - T(min_dist)));

//     // d(dist)/dx = (x - xc) / dist
//     return (dist_bar / dist) * (x - local_xc.template cast<T>());
// }

// template <typename T>
// T LevelSet<T>::evallevelset(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const
// {
//     using std::exp;   // not strictly needed here, but consistent
//     using std::sqrt;  // for downstream calls

//     T numer = T(0.0);
//     T denom = T(0.0);

//     // auto [nearest_indices, min_dist] = getnearest(x);
//     T min_dist = T(0.0);
//     Eigen::VectorXd zero_kappa = Eigen::VectorXd::Zero(dim - 1);

//     // for (int i : nearest_indices) 
//     for (int i=0; i < numbasis; ++i)
//     {
//         T perp   = locallevelset(x, xc.col(i), frame[i], zero_kappa);
//         T w      = expdist(x, xc.col(i), min_dist);

//         numer += perp * w;
//         denom += w;
//     }
    
//     return (denom == T(0.0)) ? T(0.0) : numer / denom;
// }

// template <typename T>
// Eigen::Matrix<T, Eigen::Dynamic, 1>
// LevelSet<T>::difflevelset(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const
// {
//     using std::exp;
//     using std::sqrt;

//     // We'll reconstruct the same weighted average, then backprop.
//     T numer = T(0.0);
//     T denom = T(0.0);

//     // auto [nearest_indices, min_dist] = getnearest(x);
//     T min_dist = T(0.0);
//     Eigen::VectorXd zero_kappa = Eigen::VectorXd::Zero(dim - 1);

//     // for (int i : nearest_indices) 
//     for (int i=0; i < numbasis; ++i)
//     {
//         T perp   = locallevelset(x, xc.col(i), frame[i], zero_kappa);
//         T w      = expdist(x, xc.col(i), min_dist);
//         numer   += perp * w;
//         denom   += w;
//     }

//     T ls = (denom == T(0.0)) ? T(0.0) : numer / denom;

//     // Reverse-mode accumulation
//     Eigen::Matrix<T, Eigen::Dynamic, 1> x_bar =
//         Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(dim);

//     T numer_bar = (denom == T(0.0)) ? T(0.0) : T(1.0) / denom;
//     T denom_bar = (denom == T(0.0)) ? T(0.0) : -ls / denom;

//     // for (int i : nearest_indices) 
//     for (int i=0; i < numbasis; ++i)
//     {
//         T perp   = locallevelset(x, xc.col(i), frame[i], zero_kappa);
//         T w      = expdist(x, xc.col(i), min_dist);

//         // contributions to w and perp from chain rule
//         T w_bar    = denom_bar + numer_bar * perp;
//         T perp_bar = numer_bar * w;

//         // backprop through expdist
//         x_bar += diffexpdist(x, xc.col(i), min_dist, w_bar);

//         // backprop through locallevelset
//         x_bar += difflocallevelset(x, xc.col(i), frame[i], zero_kappa, perp_bar);
//     }

//     return x_bar;
// }

// template <typename T>
// T LevelSet<T>::evallevelset_normalized(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const
// {
//     using std::atan;

//     T val = evallevelset(x);
//     // smooth map: (-∞,∞) -> (0,1) via atan
//     return (T(1.0) / T(M_PI)) * atan(T(normscale) * val) + T(0.5);
// }

// template <typename T>
// Eigen::Matrix<T, Eigen::Dynamic, 1>
// LevelSet<T>::difflevelset_normalized(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const
// {
//     using std::pow;

//     T val = evallevelset(x);
//     Eigen::Matrix<T, Eigen::Dynamic, 1> val_bar = difflevelset(x);

//     // d/dx [ (1/pi)*atan(normscale*val) + 0.5 ]
//     // = (1/pi) * (normscale / (1 + (normscale*val)^2)) * dval/dx
//     T val_scaled = T(normscale) * val;
//     // T denom = T(1.0) + pow(T(normscale) * val, 2.0);
//     T denom = T(1.0) + val_scaled * val_scaled;

//     return (T(1.0) / T(M_PI)) * (T(normscale) * val_bar / denom);
// }

// #endif // LEVELSET_HPP

#ifndef LEVELSET_HPP
#define LEVELSET_HPP

#include <Eigen/Dense>
#include <vector>
#include <complex>
#include <stdexcept>
#include <cmath> // for std::sqrt, std::tanh, std::exp

/**
 * @brief Implements the "distance * tanh(sign)" level set model.
 *
 * This class translates the 'calculate_sdf_analytic_v2' MATLAB function,
 * which produced the correct 'levset1.png' plot.
 *
 * It is fully analytic (complex-safe) and can be used for
 * complex-step differentiation.
 */
template <typename T>
class LevelSetV2 {
public:
    // --- Members ---
    int dim;
    int numbasis;
    Eigen::MatrixXd xc;    // Real-space geometry (center points)
    Eigen::MatrixXd frame; // Real-space geometry (normal vectors)
    double delta = 1e-10;  // Stabilization parameter
    double smooth_k; // Sharpness parameter 'k' for tanh

    // --- Constructor ---
    /**
     * @brief Constructs the LevelSetV2 object.
     * @param p_dim Dimension of the space (e.g., 2 for [h, p]).
     * @param xcenter An (dim x N) matrix of N center points.
     * @param normal An (dim x N) matrix of N normal vectors.
     * @param p_smooth_k The sharpness parameter 'k' for the tanh function.
     */
    LevelSetV2(int p_dim,
               const Eigen::MatrixXd& xcenter,
               const Eigen::MatrixXd& normal,
               double p_smooth_k)
        : dim(p_dim),
          numbasis(xcenter.cols()),
          xc(xcenter),
          smooth_k(p_smooth_k)
    {
        // Validation checks
        if (xcenter.cols() != normal.cols()) {
            throw std::invalid_argument("xcenter and normal must have the same number of columns.");
        }
        if (xcenter.rows() != dim) {
            throw std::invalid_argument("x-center dimension does not match problem dimension.");
        }

        // Store normalized normal vectors in 'frame'
        frame.resize(dim, numbasis);
        for (int i = 0; i < numbasis; ++i) {
            // Normalization is done in real-space (double), so .normalized() is safe.
            frame.col(i) = normal.col(i).normalized();
        }
    }
    
    /**
     * @brief Template "copy" constructor.
     * Creates a new <T> version from an existing <U> version.
     */
    template <typename U>
    LevelSetV2(const LevelSetV2<U>& other)
        : dim(other.dim),
          numbasis(other.numbasis),
          xc(other.xc),
          frame(other.frame),
          delta(other.delta),
          smooth_k(other.smooth_k)
    {}

    // --- Core Evaluation Function (calculate_sdf_analytic_v2) ---
    /**
     * @brief Calculates the analytic SDF based on the "distance * sign" model.
     * This is the C++ version of 'calculate_sdf_analytic_v2'.
     */
    T evallevelset(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const
    {
        using std::sqrt;
        using std::tanh;

        T numer = T(0.0);
        T denom = T(0.0);

        // Loop over ALL basis points
        for (int i = 0; i < numbasis; ++i) {
            // Get geometry (cast real geometry to type T)
            Eigen::Matrix<T, Eigen::Dynamic, 1> local_xc = xc.col(i).template cast<T>();
            Eigen::Matrix<T, Eigen::Dynamic, 1> local_normal = frame.col(i).template cast<T>();
            
            // 1. vec = query_point - center_point
            Eigen::Matrix<T, Eigen::Dynamic, 1> vec = x - local_xc;

            // 2. & 3. Analytic "Distance"
            // Use .dot() for analytic sum-of-squares (z1^2 + z2^2 + ...)
            T dist_sq_analytic = vec.dot(vec); 
            // std::sqrt is analytic-safe as long as the arg isn't on the negative real axis.
            // (vec.dot(vec) will be (positive_real, tiny_imag) for complex step)
            T distance = sqrt(dist_sq_analytic + T(delta));

            // 4. & 5. Analytic "Sign"
            T dot_product = vec.dot(local_normal);
            T smooth_sign = tanh(T(smooth_k) * dot_product);

            // 6. Local SDF (distance * sign)
            T local_sdf = distance * smooth_sign;

            // 7. Weight (inverse distance squared)
            T weight = T(1.0) / (dist_sq_analytic + T(delta));

            // 8. & 9. Accumulate
            numer += local_sdf * weight;
            denom += weight;
        }

        // 10. Final (safe) division
        return numer / (denom + T(delta));
    }

    // --- Normalization Function (to get 0 or 1) ---
    /**
     * @brief Normalizes the SDF to the range [0, 1].
     * This is the 'normalize_sdf' sigmoid function from your MATLAB code.
     * It maps:
     * SDF < 0 (Inside)  -> 0
     * SDF > 0 (Outside) -> 1
     */
    T evallevelset_normalized(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const
    {
        using std::exp;
        
        // 1. Get the base SDF value (which is negative inside, positive outside)
        T sdf_val = evallevelset(x);
        
        // 2. Apply the sigmoid: 1 / (1 + exp(-k * sdf))
        return T(1.0) / (T(1.0) + exp(-T(smooth_k) * sdf_val));
    }

    // --- NEW: Manual (Analytic) Derivative Functions ---

    /**
     * @brief Calculates the analytic derivative of evallevelset (d(sdf)/dx).
     * Implements reverse-mode AD (backpropagation).
     */
    Eigen::Matrix<T, Eigen::Dynamic, 1> diff_evallevelset(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const;

    /**
     * @brief Calculates the analytic derivative of evallevelset_normalized.
     */
    Eigen::Matrix<T, Eigen::Dynamic, 1> diff_evallevelset_normalized(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const;
};


// ========================= IMPLEMENTATION =========================

// --- NEW: Implementation of diff_evallevelset ---
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
LevelSetV2<T>::diff_evallevelset(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const
{
    using std::sqrt;
    using std::tanh;

    // --- 1. Forward Pass: Calculate all intermediate values ---
    T numer = T(0.0);
    T denom = T(0.0);

    // Store intermediates needed for the reverse pass
    std::vector<T> all_local_sdf(numbasis);
    std::vector<T> all_weight(numbasis);
    std::vector<T> all_distance(numbasis);
    std::vector<T> all_smooth_sign(numbasis);
    std::vector<T> all_dist_sq(numbasis);
    std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>> all_vec(numbasis);

    for (int i = 0; i < numbasis; ++i) {
        Eigen::Matrix<T, Eigen::Dynamic, 1> local_xc = xc.col(i).template cast<T>();
        Eigen::Matrix<T, Eigen::Dynamic, 1> local_normal = frame.col(i).template cast<T>();
        
        all_vec[i] = x - local_xc;
        all_dist_sq[i] = all_vec[i].dot(all_vec[i]);
        all_distance[i] = sqrt(all_dist_sq[i] + T(delta));
        
        T dot_product = all_vec[i].dot(local_normal);
        all_smooth_sign[i] = tanh(T(smooth_k) * dot_product);
        
        all_local_sdf[i] = all_distance[i] * all_smooth_sign[i];
        all_weight[i] = T(1.0) / (all_dist_sq[i] + T(delta));

        numer += all_local_sdf[i] * all_weight[i];
        denom += all_weight[i];
    }

    T denom_plus_delta = denom + T(delta);
    // T sdf_val = numer / denom_plus_delta;

    // --- 2. Reverse Pass: Backpropagate gradients ---
    Eigen::Matrix<T, Eigen::Dynamic, 1> x_bar = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(dim);
    
    // Adjoint of f = numer / (denom + delta)
    T sdf_val_bar = T(1.0); // Adjoint of the output
    T numer_bar = sdf_val_bar * (T(1.0) / denom_plus_delta);
    T denom_bar = sdf_val_bar * (-numer / (denom_plus_delta * denom_plus_delta)); // or (-sdf_val / denom_plus_delta)

    for (int i = 0; i < numbasis; ++i) {
        // Backprop through numer and denom sums
        T local_sdf_bar = numer_bar * all_weight[i];
        T weight_bar = numer_bar * all_local_sdf[i] + denom_bar;

        // Adjoint for x_bar (gradient accumulator)
        Eigen::Matrix<T, Eigen::Dynamic, 1> vec_bar = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(dim);

        // --- Backprop local_sdf_bar ---
        // local_sdf = distance * smooth_sign
        T distance_bar = local_sdf_bar * all_smooth_sign[i];
        T smooth_sign_bar = local_sdf_bar * all_distance[i];

        // --- Backprop weight_bar ---
        // weight = 1.0 / (dist_sq + delta)
        T dist_sq_bar = weight_bar * ( -all_weight[i] * all_weight[i] ); // -1 / (dist_sq + delta)^2
        
        // --- Backprop distance_bar ---
        // distance = sqrt(dist_sq + delta)
        dist_sq_bar += distance_bar * (T(0.5) / all_distance[i]); // 0.5 / sqrt(dist_sq + delta)

        // --- Backprop smooth_sign_bar ---
        // smooth_sign = tanh(k * dot_product)
        T tanh_val = all_smooth_sign[i];
        T sech_sq = T(1.0) - (tanh_val * tanh_val);
        T dot_product_bar = smooth_sign_bar * (T(smooth_k) * sech_sq);

        // --- Backprop dist_sq_bar ---
        // dist_sq = vec.dot(vec)
        vec_bar += dist_sq_bar * (T(2.0) * all_vec[i]);
        
        // --- Backprop dot_product_bar ---
        // dot_product = vec.dot(local_normal)
        vec_bar += dot_product_bar * frame.col(i).template cast<T>();

        // --- Accumulate gradient from vec ---
        // vec = x - local_xc
        x_bar += vec_bar;
    }

    return x_bar;
}

// --- NEW: Implementation of diff_evallevelset_normalized ---
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
LevelSetV2<T>::diff_evallevelset_normalized(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const
{
    // Get the base SDF value and its derivative
    T sdf_val = evallevelset(x);
    Eigen::Matrix<T, Eigen::Dynamic, 1> sdf_val_bar = diff_evallevelset(x);

    // Apply chain rule for: 1.0 / (1.0 + exp(-k * sdf_val))
    T exp_val = exp(-T(smooth_k) * sdf_val);
    T denom = T(1.0) + exp_val;
    T denom_sq = denom * denom;

    // d/dsdf [ 1 / (1 + exp(-k*s)) ] = -1/(...)^2 * (exp(-k*s) * -k)
    //                                = k * exp(-k*s) / (1 + exp(-k*s))^2
    T outer_deriv = (T(smooth_k) * exp_val) / denom_sq;
    
    return outer_deriv * sdf_val_bar;
}


#endif // LEVELSET_V2_HPP

