#ifndef LEGENDRE_INTERPOLATOR_HPP
#define LEGENDRE_INTERPOLATOR_HPP

#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <boost/math/special_functions/legendre.hpp> // For Legendre polynomials

template<typename T>
struct Prediction2D {
    T value;
    T d_dh;
    T d_dp;
    T d2_dh2;
    T d2_dp2;
    T d2_dhdp;
};

template<typename T>
struct Prediction1D {
    T value;
    T d_ds;
    T d2_ds2;
};

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> scalar_legendreP(const int N, T x) {
    Eigen::Matrix<T, Eigen::Dynamic, 1> P(N+1);
    P(0) = T(1.0);
    if (N > 0) P(1) = x;

    for (int n=1; n < N; ++n) {
        // Recurrence relation: (n+1)P_{n+1}(x) = (2n+1)* x *P_n(x) - nP_{n-1}(x)
        P(n+1) = ( T(2.0*n+1.0) * x * P(n) - T(n) * P(n-1) ) / T(n+1.0);
    }
    return P;
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> scalar_legendrePDeriv(const int N, T x) {
    Eigen::Matrix<T, Eigen::Dynamic, 1> P_deriv(N + 1);
    P_deriv(0) = T(0.0);
    if (N > 0) P_deriv(1) = T(1.0);

    Eigen::Matrix<T, Eigen::Dynamic, 1> P = scalar_legendreP(N, x);
    for (int n=1; n < N; ++n) {
        // Derivative relation: (n+1)P'_{n+1}(x) = (2n+1)(P_n(x) + x*P'_n(x)) - nP'_{n-1}(x) 
        P_deriv(n+1) = ( T(2.0*n+1.0)*( P(n) + x*P_deriv(n) ) - T(n)*P_deriv(n-1) ) / T(n+1.0);
    }
    return P_deriv;
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> scalar_legendrePDeriv2( const int N, T x) {
    Eigen::Matrix<T, Eigen::Dynamic, 1> P_deriv2(N + 1);
    P_deriv2(0) = T(0.0);
    if (N>0) P_deriv2(1) = T(0.0);

    auto P_deriv = scalar_legendrePDeriv(N, x);
    for (int n=1; n<N; ++n) {
        // Second derivative relation: (n+1)P''_{n+1}(x) = (2n+1)(2P'_n(x) + x*P''_n(x)) - nP''_{n-1}(x)
        P_deriv2(n+1) = ( T(2.0*n+1.0) * (T(2.0)*P_deriv(n) + x*P_deriv2(n)) - T(n)*P_deriv2(n-1) ) / T(n+1.0);
    }
    return P_deriv2;
}

// ==================================================================================
// h-LOG(p) 2D LEGENDRE INTERPOLATOR CLASS 
// ==================================================================================

template<typename T>
class HLogP_LegendreInterpolator {
public:
    using TemplateVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    HLogP_LegendreInterpolator(const TemplateVector& coeffs, 
                             int h_degree, 
                             int p_degree, 
                             T h_min, 
                             T h_range, 
                             T log_p_min, 
                             T log_p_range)
        : coeffs_(coeffs),
          h_degree_(h_degree),
          p_degree_(p_degree),
          h_min_(h_min),
          h_range_(h_range),
          log_p_min_(log_p_min),
          log_p_range_(log_p_range)
    {}
    
    Prediction2D<T> predict(T h_new, T p_new);

private:
    const TemplateVector coeffs_;
    const int h_degree_;
    const int p_degree_;
    const T h_min_;
    const T h_range_;
    const T log_p_min_;
    const T log_p_range_;
};

template<typename T>
Prediction2D<T> HLogP_LegendreInterpolator<T>::predict(T h_new, T p_new) {
    // --- 1. Scaling ---
    T log_p_new = std::log(p_new);
    T h_scaled = (T(2.0)*(h_new - h_min_) / h_range_) - T(1.0);
    T log_p_scaled = (T(2.0)*(log_p_new - log_p_min_) / log_p_range_) - T(1.0);

    // --- 2. Legendre Polynomials & Derivatives (at scaled points) ---
    auto L_h = scalar_legendreP(h_degree_, h_scaled);
    auto L_p = scalar_legendreP(p_degree_, log_p_scaled);
    auto L_h_deriv = scalar_legendrePDeriv(h_degree_, h_scaled);
    auto L_p_deriv = scalar_legendrePDeriv(p_degree_, log_p_scaled);
    auto L_h_deriv2 = scalar_legendrePDeriv2(h_degree_, h_scaled);
    auto L_p_deriv2 = scalar_legendrePDeriv2(p_degree_, log_p_scaled);

    // --- 3. Accumulate Derivatives (w.r.t. scaled variables) ---
    // We use explicit names for clarity:
    T f_value = T(0.0);    // f(h_s, p_s)
    T df_dhs = T(0.0);     // df / dh_s
    T df_dps = T(0.0);     // df / dp_s
    T d2f_dhs2 = T(0.0);   // d^2f / dh_s^2
    T d2f_dps2 = T(0.0);   // d^2f / dp_s^2
    T d2f_dhsdps = T(0.0); // d^2f / dh_s dp_s

    int k = 0;
    for (int i=0; i<=h_degree_; ++i) {
        for (int j=0; j<= p_degree_; ++j) {
            f_value    += coeffs_(k) * L_h(i) * L_p(j);
            
            df_dhs     += coeffs_(k) * L_h_deriv(i) * L_p(j);
            df_dps     += coeffs_(k) * L_h(i) * L_p_deriv(j);

            d2f_dhs2   += coeffs_(k) * L_h_deriv2(i) * L_p(j);
            d2f_dps2   += coeffs_(k) * L_h(i) * L_p_deriv2(j);
            d2f_dhsdps += coeffs_(k) * L_h_deriv(i) * L_p_deriv(j); // Mixed partial
            k += 1;
        }
    }

    // --- 4. Chain Rule Scaling (from scaled to real variables) ---
    Prediction2D<T> pred;
    pred.value = f_value;

    // Scaling factors
    T dhs_dh = (T(2.0) / h_range_);
    T C_p = (T(2.0) / log_p_range_); // Constant part of p-scaling
    T dps_dp = C_p * (T(1.0) / p_new);
    
    // First derivatives
    pred.d_dh = df_dhs * dhs_dh;
    pred.d_dp = df_dps * dps_dp;

    // Second derivatives
    pred.d2_dh2 = d2f_dhs2 * (dhs_dh * dhs_dh); // (dhs_dh)^2
    
    // Mixed partial
    pred.d2_dhdp = d2f_dhsdps * dhs_dh * dps_dp;

    // Full product rule for d2_dp2
    T d2ps_dp2 = C_p * (-T(1.0) / (p_new * p_new)); // d/dp(C_p / p) = -C_p / p^2
    pred.d2_dp2 = d2f_dps2 * (dps_dp * dps_dp) + df_dps * d2ps_dp2;
    //           (Term 1: d2f/dps2 * (dps/dp)^2) (Term 2: df/dps * d2ps/dp2)

    return pred;
}

// ==================================================================================
// LOG(s) 1D LEGENDRE INTERPOLATOR CLASS 
// ==================================================================================

template<typename T>
class LogS_LegendreInterpolator1D{
public:
    using TemplateVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    LogS_LegendreInterpolator1D(const TemplateVector& coeffs,
                             int s_degree,
                             T log_s_min,
                             T log_s_range)
        : coeffs_(coeffs),
          s_degree_(s_degree),
          log_s_min_(log_s_min),
          log_s_range_(log_s_range)
    {}
    Prediction1D<T> predict(T s_new);
private:
    const TemplateVector coeffs_;
    const int s_degree_;
    const T log_s_min_;
    const T log_s_range_;
};

template<typename T>
Prediction1D<T> LogS_LegendreInterpolator1D<T>::predict(T s_new) {
    T log_s_new = std::log(s_new);
    T log_s_scaled = (T(2.0)*(log_s_new - log_s_min_) / log_s_range_) - T(1.0);

    auto L_s = scalar_legendreP(s_degree_, log_s_scaled);
    auto L_s_deriv = scalar_legendrePDeriv(s_degree_, log_s_scaled);
    auto L_s_deriv2 = scalar_legendrePDeriv2(s_degree_, log_s_scaled);

    T f_value = T(0.0);
    T df_dlog_s_scaled = T(0.0);
    T d2f_dlog_s_scaled2 = T(0.0);

    Prediction1D<T> pred;
    for (int i=0; i <= s_degree_; ++i) {
        f_value += coeffs_(i) * L_s(i);
        df_dlog_s_scaled += coeffs_(i) * L_s_deriv(i);
        d2f_dlog_s_scaled2 += coeffs_(i) * L_s_deriv2(i);
    }

    pred.value = f_value;

    T dlogs_scaled_dlog_s = T(2.0) / log_s_range_;
    T dlog_s_ds = T(1.0) / s_new;
    T d2logs_ds2 = -T(1.0) / (s_new * s_new);
    pred.d_ds = df_dlog_s_scaled * dlogs_scaled_dlog_s * dlog_s_ds;

    T dlog_s_scaled_ds = dlogs_scaled_dlog_s * dlog_s_ds;
    pred.d2_ds2 = dlog_s_scaled_ds * dlog_s_scaled_ds * d2f_dlog_s_scaled2 +
                  dlogs_scaled_dlog_s *  df_dlog_s_scaled * d2logs_ds2; 
    return pred;
}

// ==================================================================================
// LOG(h)Log(p) 2D LEGENDRE INTERPOLATOR CLASS 
// ==================================================================================

template<typename T>
class LogHLogP_LegendreInterpolator {
    using TemplateVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
public:
    LogHLogP_LegendreInterpolator(const TemplateVector& coeffs,
                             int h_degree,
                             int p_degree,
                             T log_h_min,
                             T log_h_range,
                             T log_p_min,
                             T log_p_range)
        : coeffs_(coeffs),
          h_degree_(h_degree),
          p_degree_(p_degree),
          log_h_min_(log_h_min),
          log_h_range_(log_h_range),
          log_p_min_(log_p_min),
          log_p_range_(log_p_range)
    {}

    Prediction2D<T> predict(T h_new, T p_new);
private:
    const TemplateVector coeffs_;
    const int h_degree_;
    const int p_degree_;
    const T log_h_min_;
    const T log_h_range_;
    const T log_p_min_;
    const T log_p_range_;
};

template<typename T>
Prediction2D<T> LogHLogP_LegendreInterpolator<T>::predict(T h_new, T p_new) {
    T log_h_new = std::log(h_new);
    T log_p_new = std::log(p_new);

    T log_h_scaled = (T(2.0)*(log_h_new - log_h_min_)/log_h_range_) - T(1.0);
    T log_p_scaled = (T(2.0)*(log_p_new - log_p_min_)/log_p_range_) - T(1.0);

    auto L_h = scalar_legendreP(h_degree_, log_h_scaled);
    auto L_h_deriv = scalar_legendrePDeriv(h_degree_, log_h_scaled);
    auto L_h_deriv2 = scalar_legendrePDeriv2(h_degree_, log_h_scaled);
    auto L_p = scalar_legendreP(p_degree_, log_p_scaled);
    auto L_p_deriv = scalar_legendrePDeriv(p_degree_, log_p_scaled);
    auto L_p_deriv2 = scalar_legendrePDeriv2(p_degree_, log_p_scaled);

    T f_value = T(0.0);
    T df_dhs = T(0.0);
    T df_dps = T(0.0);
    T d2f_dhs2 = T(0.0);
    T d2f_dps2 = T(0.0);
    T d2f_dhsdps = T(0.0);
    
    int k = 0;
    for (int i=0; i<=h_degree_; ++i) {
        for (int j=0; j<=p_degree_; ++j) {
            f_value += coeffs_(k) * L_h(i) * L_p(j);
            df_dhs += coeffs_(k) * L_h_deriv(i) * L_p(j);
            df_dps += coeffs_(k) * L_h(i) * L_p_deriv(j);
            d2f_dhs2 += coeffs_(k) * L_h_deriv2(i) * L_p(j);
            d2f_dps2 += coeffs_(k) * L_h(i) * L_p_deriv2(j);
            d2f_dhsdps += coeffs_(k) * L_h_deriv(i) * L_p_deriv(j);
            k += 1;
        }
    }

    T dhs_dh  = (T(2.0) / log_h_range_) * (T(1.0) / h_new);
    T d2hs_dh2 = (T(-2.0) / log_h_range_) * (T(1.0) / (h_new * h_new));
    T dps_dp  = (T(2.0) / log_p_range_) * (T(1.0) / p_new);
    T d2ps_dp2 = (T(-2.0) / log_p_range_) * (T(1.0) / (p_new * p_new));

    Prediction2D<T> pred;
    pred.value = f_value;
    pred.d_dh = df_dhs * dhs_dh;
    pred.d_dp = df_dps * dps_dp;
    pred.d2_dh2 = d2f_dhs2 * dhs_dh * dhs_dh + df_dhs * d2hs_dh2;
    pred.d2_dp2 = d2f_dps2 * dps_dp * dps_dp + df_dps * d2ps_dp2;
    pred.d2_dhdp = d2f_dhsdps * dhs_dh * dps_dp;
    return pred;
}

#endif