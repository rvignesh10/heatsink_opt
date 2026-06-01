#ifndef THERMAL_BC_COEFFICIENT_HPP
#define THERMAL_BC_COEFFICIENT_HPP

#include"mfem.hpp"
#include <functional>
#include <iostream>
#include <cmath>

/**
 * @brief Defines the function signature for the forcing term.
 * This is what the C++ function (or lambda) must look like.
 * It takes (x, y, z, T_in_star, T_src_star) and returns a double.
 */
using ForcingFunction = std::function<double(double, double, double, double, double)>;

/**
 * @brief C++ version of the PyCoefficient for a custom forcing term.
 *
 * This class inherits from mfem::Coefficient and holds a std::function
 * to evaluate the forcing term at any point p(x,y,z).
 */
class ForcingCoefficient : public mfem::Coefficient
{
public:
    /**
     * @brief Constructs the ForcingCoefficient.
     *
     * @param force_func A C++ function or lambda that matches the ForcingFunction signature.
     * @param dim The dimension of the problem (2 or 3).
     * @param T_in_star Non-dimensional inlet temperature.
     * @param T_src_star Non-dimensional source temperature.
     */
    ForcingCoefficient(ForcingFunction force_func, int dim,
                       double T_in_star, double T_src_star)
        : force_func_(force_func), // Store the std::function
          dim_(dim),
          T_in_star_(T_in_star),
          T_src_star_(T_src_star)
    {
        // You can optionally tell the base class the time
        // if your coefficient depends on it (it doesn't here).
        // SetTime(0.0); 
    }

    /**
     * @brief The C++ equivalent of EvalValue.
     *
     * MFEM calls this function to get the coefficient's value at a point p.
     * Note: This overrides the simpler Eval(Vector &p) method.
     */
    virtual double Eval(mfem::Vector &p)
    {
        // Ensure the point vector 'p' has the correct dimension.
        MFEM_ASSERT(p.Size() == dim_, 
            "ForcingCoefficient: Point dimension " << p.Size() 
            << " does not match specified dimension " << dim_);

        double x = p(0);
        double y = p(1);
        double z = (dim_ == 3) ? p(2) : 0.0; // Handle 2D or 3D

        // Call the stored std::function with the point and parameters
        return force_func_(x, y, z, T_in_star_, T_src_star_);
    }

    virtual double Eval(mfem::ElementTransformation &T,
                       const mfem::IntegrationPoint &ip) {
        mfem::Vector p;
        T.Transform(ip, p);

        // 2. Call the other Eval function with the physical coordinates.
        return Eval(p);
    }

    void SetSourceTemperature(const double source_load_star) {
        this->T_src_star_ = source_load_star;
    }

private:
    ForcingFunction force_func_; // The stored callable function
    int dim_;
    double T_in_star_;
    double T_src_star_;
};

/**
 * @brief C++ 'set_dbc' function, matching the ForcingFunction signature.
 */
inline double set_dbc_forcing(double x, double y, double z, double Tin_star, double Tsrc_star)
{
    // A small number to avoid division by zero, similar to Python
    const double epsilon = 1e-15; 
    // const double epsilon = std::numeric_limits<double>::epsilon(); // Alternative
    const double eps_x = 0.02;
    const double eps_z = 0.002;

    // Use (val * val) which is often faster than std::pow(val, 2.0)
    const double x_scaled = x / eps_x;
    const double z_scaled = z / eps_z;
    
    // std::exp is the C++ equivalent of np.exp
    const double wx = std::exp(-(x_scaled * x_scaled));
    const double wz = std::exp(-(z_scaled * z_scaled));
    
    return (wx * Tin_star + wz * Tsrc_star) / (wx + wz + epsilon);
}

#endif // THERMAL_BC_COEFFICIENT_HPP