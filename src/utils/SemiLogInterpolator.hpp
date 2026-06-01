#ifndef SEMI_LOG_INTERPOLATOR_HPP
#define SEMI_LOG_INTERPOLATOR_HPP

#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <boost/math/special_functions/legendre.hpp> // For Legendre polynomials

// // ==================================================================================
// // SHARED HELPER FUNCTIONS (Moved out of class)
// // ==================================================================================

// /**
//  * @brief Evaluates Legendre polynomials P_n(x) up to degree N using Boost.
//  * Accepts any Eigen vector/expression (MatrixBase) to avoid mismatched expression types.
// */
// template <typename Derived>
// auto legendreP(int N, const Eigen::MatrixBase<Derived>& x_expr)
//     -> Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
// {
//     using S = typename Derived::Scalar;
//     Eigen::Matrix<S, Eigen::Dynamic, 1> x = x_expr.derived().template cast<S>();
//     Eigen::Index m = x.size();
//     Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic> L(m, N + 1);
//     for (Eigen::Index i = 0; i < m; ++i) {
//         for (int n = 0; n <= N; ++n) {
//             L(i, n) = static_cast<S>(boost::math::legendre_p(n, static_cast<double>(x(i))));
//         }
//     }
//     return L;
// }

// /**
//  * @brief Evaluates derivatives of Legendre polynomials P'_n(x) up to degree N.
//  * Accepts any Eigen vector/expression (MatrixBase).
// */
// template <typename Derived>
// auto legendrePDeriv(int N, const Eigen::MatrixBase<Derived>& x_expr)
//     -> Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
// {
//     using S = typename Derived::Scalar;
//     Eigen::Matrix<S, Eigen::Dynamic, 1> x = x_expr.derived().template cast<S>();
//     Eigen::Index m = x.size();
//     Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic> L_deriv(m, N + 1);
//     L_deriv.setZero();
//     if (N > 0) L_deriv.col(1).setOnes();

//     auto L = legendreP(N, x);

//     for (int n = 1; n < N; ++n) {
//         // L_deriv.col(n+1) = ( (2n+1) * L.col(n) + L_deriv.col(n-1) )  (elementwise)
//         L_deriv.col(n + 1) = (S(2.0 * n + 1.0) * L.col(n).array() + L_deriv.col(n - 1).array()).matrix();
//     }
//     return L_deriv;
// }


template <typename Scalar, typename Derived>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
legendreP_eigen(int N, const Eigen::MatrixBase<Derived>& x_expr)
{
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    Vec x = x_expr.derived().template cast<Scalar>();
    if (x.cols() > 1 && x.rows() == 1)
        x.transposeInPlace();
    Eigen::Index M = x.size();

    Mat L = Mat::Zero(M, N + 1);
    L.col(0).setOnes();
    if (N > 0)
        L.col(1) = x;

    for (int n = 1; n < N; ++n)
        L.col(n + 1) = ((2 * n + 1) * x.array() * L.col(n).array()
                      - n * L.col(n - 1).array()) / Scalar(n + 1);

    return L;
}

template <typename Scalar, typename Derived>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
legendrePDeriv_eigen(int N, const Eigen::MatrixBase<Derived>& x_expr)
{
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    Vec x = x_expr.derived().template cast<Scalar>();
    if (x.cols() > 1 && x.rows() == 1)
        x.transposeInPlace();
    Eigen::Index M = x.size();

    Mat L_deriv = Mat::Zero(M, N + 1);
    L_deriv.col(0).setZero();
    if (N > 0)
        L_deriv.col(1).setOnes();

    Mat L = legendreP_eigen<Scalar>(N, x);

    for (int n = 1; n < N; ++n)
        // L_deriv.col(n + 1) = (Scalar(2 * n + 1) * L.col(n).array()
        //                     + L_deriv.col(n - 1).array()).matrix();
        L_deriv.col(n+1) = ( (Scalar(2.0*n+1.0)/Scalar(n+1.0)) ) * ( L.col(n).array() + x.array()*L_deriv.col(n).array() ) -
                            ( Scalar(n)/(Scalar(n+1.0)) ) * L_deriv.col(n-1).array();

    return L_deriv;
}

// ==================================================================================
// 2D PREDICTION STRUCT AND CLASS
// ==================================================================================

template<typename T>
struct Prediction {
    T value;
    T d_dh;
    T d_dp;
};

/**
 * @brief 2D interpolator for f(log(h), log(p)).
 */
template<typename T>
class SemiLogInterpolator {
public:
    SemiLogInterpolator(
        const Eigen::Matrix<T, Eigen::Dynamic, 1>& coeffs,
        int h_degree,
        int p_degree,
        double log_h_min,
        double log_h_range,
        double log_p_min,
        double log_p_range
    );

    Prediction<T> predict(T h_new, T p_new) const;
    std::vector<Prediction<T>> predict(const std::vector<T>& h_new, const std::vector<T>& p_new) const;

    // Getters
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& getCoeffs() const { return coeffs_; }
    int getHDegree() const { return h_degree_; }
    int getPDegree() const { return p_degree_; }
    double getLogHMin() const { return log_h_min_; }
    double getLogHRange() const { return log_h_range_; }
    double getLogPMin() const { return log_p_min_; }
    double getLogPRange() const { return log_p_range_; }

private:
    Eigen::Matrix<T, Eigen::Dynamic, 1> coeffs_;
    int h_degree_;
    int p_degree_;
    double log_h_min_;
    double log_h_range_;
    double log_p_min_;
    double log_p_range_;
};


template<typename T>
SemiLogInterpolator<T>::SemiLogInterpolator(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& coeffs,
    int h_degree,
    int p_degree,
    double log_h_min,
    double log_h_range,
    double log_p_min,
    double log_p_range
) : coeffs_(coeffs),
    h_degree_(h_degree),
    p_degree_(p_degree),
    log_h_min_(log_h_min),
    log_h_range_(log_h_range),
    log_p_min_(log_p_min),
    log_p_range_(log_p_range)
{}


template<typename T>
Prediction<T> SemiLogInterpolator<T>::predict(T h_new, T p_new) const {
    return predict(std::vector<T>{h_new}, std::vector<T>{p_new})[0];
}

template<typename T>
std::vector<Prediction<T>> SemiLogInterpolator<T>::predict(const std::vector<T>& h_new, const std::vector<T>& p_new) const {
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> h_vec(h_new.data(), h_new.size());
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> p_vec(p_new.data(), p_new.size());

    auto log_h_new = h_vec.array().log();
    auto log_p_new = p_vec.array().log();

    auto log_h_scaled = (T(2.0) * (log_h_new - log_h_min_) / log_h_range_) - T(1.0);
    auto log_p_scaled = (T(2.0) * (log_p_new - log_p_min_) / log_p_range_) - T(1.0);
    

    // Call the free helper functions
    auto L_h = ::legendreP_eigen<T>(h_degree_, log_h_scaled.matrix());
    auto L_p = ::legendreP_eigen<T>(p_degree_, log_p_scaled.matrix());
    auto L_h_deriv = ::legendrePDeriv_eigen<T>(h_degree_, log_h_scaled.matrix());
    auto L_p_deriv = ::legendrePDeriv_eigen<T>(p_degree_, log_p_scaled.matrix());
    
    Eigen::Array<T, Eigen::Dynamic, 1> result = Eigen::Array<T, Eigen::Dynamic, 1>::Zero(h_new.size());
    Eigen::Array<T, Eigen::Dynamic, 1> d_result_dxs = Eigen::Array<T, Eigen::Dynamic, 1>::Zero(h_new.size());
    Eigen::Array<T, Eigen::Dynamic, 1> d_result_dys = Eigen::Array<T, Eigen::Dynamic, 1>::Zero(h_new.size());

    int k = 0;
    for (int i = 0; i <= h_degree_; ++i) {
        for (int j = 0; j <= p_degree_; ++j) {
            result += coeffs_(k) * (L_h.col(i).array() * L_p.col(j).array());
            d_result_dxs += coeffs_(k) * (L_h_deriv.col(i).array() * L_p.col(j).array());
            d_result_dys += coeffs_(k) * (L_h.col(i).array() * L_p_deriv.col(j).array());
            k++;
        }
    }

    auto d_result_d_log_h = d_result_dxs * (T(2.0) / log_h_range_);
    auto d_result_d_log_p = d_result_dys * (T(2.0) / log_p_range_);
    auto d_dh = d_result_d_log_h * (T(1.0) / h_vec.array());
    auto d_dp = d_result_d_log_p * (T(1.0) / p_vec.array());

    std::vector<Prediction<T>> predictions;
    predictions.reserve(h_new.size());
    for(size_t i = 0; i < h_new.size(); ++i) {
        predictions.push_back({result(i), d_dh(i), d_dp(i)});
    }
    return predictions;
}

// 

/**
 * @brief 2D interpolator for f(h, log(p)).
 */
template<typename T>
class SemiLogInterpolatorAlt {
public:
    SemiLogInterpolatorAlt(
        const Eigen::Matrix<T, Eigen::Dynamic, 1>& coeffs,
        int h_degree,
        int p_degree,
        double h_min,
        double h_range,
        double log_p_min,
        double log_p_range
    );

    Prediction<T> predict(T h_new, T p_new) const;
    std::vector<Prediction<T>> predict(const std::vector<T>& h_new, const std::vector<T>& p_new) const;

    // Getters
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& getCoeffs() const { return coeffs_; }
    int getHDegree() const { return h_degree_; }
    int getPDegree() const { return p_degree_; }
    double getHMin() const { return h_min_; }
    double getHRange() const { return h_range_; }
    double getLogPMin() const { return log_p_min_; }
    double getLogPRange() const { return log_p_range_; }

private:
    Eigen::Matrix<T, Eigen::Dynamic, 1> coeffs_;
    int h_degree_;
    int p_degree_;
    double h_min_;
    double h_range_;
    double log_p_min_;
    double log_p_range_;
    // legendreP and legendrePDeriv are now free functions
};


template<typename T>
SemiLogInterpolatorAlt<T>::SemiLogInterpolatorAlt(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& coeffs,
    int h_degree,
    int p_degree,
    double h_min,
    double h_range,
    double log_p_min,
    double log_p_range
) : coeffs_(coeffs),
    h_degree_(h_degree),
    p_degree_(p_degree),
    h_min_(h_min),
    h_range_(h_range),
    log_p_min_(log_p_min),
    log_p_range_(log_p_range)
{}


template<typename T>
Prediction<T> SemiLogInterpolatorAlt<T>::predict(T h_new, T p_new) const {
    return predict(std::vector<T>{h_new}, std::vector<T>{p_new})[0];
}

template<typename T>
std::vector<Prediction<T>> SemiLogInterpolatorAlt<T>::predict(const std::vector<T>& h_new, const std::vector<T>& p_new) const {
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> h_vec(h_new.data(), h_new.size());
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> p_vec(p_new.data(), p_new.size());
    
    auto h_new_arr = h_vec.array();
    auto log_p_new = p_vec.array().log();

    auto h_scaled = (T(2.0) * (h_new_arr.array() - h_min_) / h_range_) - T(1.0);
    auto log_p_scaled = (T(2.0) * (log_p_new - log_p_min_) / log_p_range_) - T(1.0);

    // Call the free helper functions
    auto L_h = ::legendreP_eigen<T>(h_degree_, h_scaled.matrix());
    auto L_p = ::legendreP_eigen<T>(p_degree_, log_p_scaled.matrix());
    auto L_h_deriv = ::legendrePDeriv_eigen<T>(h_degree_, h_scaled.matrix());
    auto L_p_deriv = ::legendrePDeriv_eigen<T>(p_degree_, log_p_scaled.matrix());
    
    Eigen::Array<T, Eigen::Dynamic, 1> result = Eigen::Array<T, Eigen::Dynamic, 1>::Zero(h_new_arr.size());
    Eigen::Array<T, Eigen::Dynamic, 1> d_result_dxs = Eigen::Array<T, Eigen::Dynamic, 1>::Zero(h_new_arr.size());
    Eigen::Array<T, Eigen::Dynamic, 1> d_result_dys = Eigen::Array<T, Eigen::Dynamic, 1>::Zero(h_new_arr.size());

    int k = 0;
    for (int i = 0; i <= h_degree_; ++i) {
        for (int j = 0; j <= p_degree_; ++j) {
            result += coeffs_(k) * (L_h.col(i).array() * L_p.col(j).array());
            d_result_dxs += coeffs_(k) * (L_h_deriv.col(i).array() * L_p.col(j).array());
            d_result_dys += coeffs_(k) * (L_h.col(i).array() * L_p_deriv.col(j).array());
            k++;
        }
    }

    auto d_result_d_h = d_result_dxs * (T(2.0) / h_range_);
    auto d_result_d_log_p = d_result_dys * (T(2.0) / log_p_range_);
    auto d_dh = d_result_d_h * ( T(1.0) ) ;
    auto d_dp = d_result_d_log_p * (T(1.0) / p_vec.array());

    std::vector<Prediction<T>> predictions;
    predictions.reserve(h_new_arr.size());
    for(size_t i = 0; i < h_new_arr.size(); ++i) {
        predictions.push_back({result(i), d_dh(i), d_dp(i)});
    }
    return predictions;
}


// ==================================================================================
// NEW 1D PREDICTION STRUCT AND CLASS
// ==================================================================================

/**
 * @brief Prediction struct for 1D interpolator.
 */
template<typename T>
struct Prediction1D {
    T value;
    T d_ds;
};

/**
 * @brief 1D interpolator for f(log(s)).
 */
template<typename T>
class SemiLogInterpolator1D {
public:
    SemiLogInterpolator1D(
        const Eigen::Matrix<T, Eigen::Dynamic, 1>& coeffs,
        int s_degree,
        double log_s_min,
        double log_s_range
    ) : coeffs_(coeffs),
        s_degree_(s_degree),
        log_s_min_(log_s_min),
        log_s_range_(log_s_range)
    {}

    /**
     * @brief Predicts value and derivative for a single point.
     */
    Prediction1D<T> predict(T s_new) const {
        return predict(std::vector<T>{s_new})[0];
    }

    /**
     * @brief C++ translation of the user's MATLAB function.
     * Predicts values and derivatives for a vector of points.
     */
    /**
     * @brief Predicts values and derivatives for a vector of points.
     */
    std::vector<Prediction1D<T>> predict(const std::vector<T>& s_new) const {
        // Map std::vector to Eigen vector
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> s_vec(s_new.data(), s_new.size());

        auto log_s_new = s_vec.array().log();
        auto log_s_scaled = (T(2.0) * (log_s_new - log_s_min_) / log_s_range_) - T(1.0);
        
        auto L_s_new = ::legendreP_eigen<T>(s_degree_, log_s_scaled.matrix());
        auto L_s_deriv_new = ::legendrePDeriv_eigen<T>(s_degree_, log_s_scaled.matrix());

        Eigen::Matrix<T, Eigen::Dynamic, 1> result = L_s_new * coeffs_;
        Eigen::Matrix<T, Eigen::Dynamic, 1> dT_dx_s = L_s_deriv_new * coeffs_;

        auto dT_dx = dT_dx_s.array() * (T(2.0) / log_s_range_);
        auto d_ds = dT_dx * (T(1.0) / s_vec.array());

        // Package results into std::vector
        std::vector<Prediction1D<T>> predictions;
        predictions.reserve(s_new.size());
        for(size_t i = 0; i < s_new.size(); ++i) {
            predictions.push_back({result(i), d_ds(i)});
        }
        return predictions;
    }

    // Getters
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& getCoeffs() const { return coeffs_; }
    int getSDegree() const { return s_degree_; }
    double getLogSMin() const { return log_s_min_; }
    double getLogSRange() const { return log_s_range_; }

private:
    Eigen::Matrix<T, Eigen::Dynamic, 1> coeffs_;
    int s_degree_;
    double log_s_min_;
    double log_s_range_;
};


#endif // SEMI_LOG_INTERPOLATOR_HPP

