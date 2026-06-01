#ifndef INEXACT_NEWTON_HPP
#define INEXACT_NEWTON_HPP

#include "mfem.hpp"
#include <mpi.h>
#include "utils.hpp"



using namespace mfem;

/// performs quadratic interpolation given x0, y0, dy0/dx0, x1, and y1.
inline double quadInterp(double x0, double y0, double dydx0, double x1, double y1)
{
   // Assume the fuction has the form y(x) = c0 + c1 * x + c2 * x^2
   // double c0 = (dydx0 * x0 * x0 * x1 + y1 * x0 * x0 - dydx0 * x0 * x1 * x1 -
   // 2 * y0 * x0 * x1 + y0 * x1 * x1) /
   //   (x0 * x0 - 2 * x1 * x0 + x1 * x1);
   double c1 = (2 * x0 * y0 - 2 * x0 * y1 - x0 * x0 * dydx0 + x1 * x1 * dydx0) /
               (x0 * x0 - 2 * x1 * x0 + x1 * x1);
   double c2 =
       -(y0 - y1 - x0 * dydx0 + x1 * dydx0) / (x0 * x0 - 2 * x1 * x0 + x1 * x1);
   return -c1 / (2 * c2);
}

/**
 * @brief Inexact Newton's method to solve Nonlinear equations F(x) = b with globalization
 */
class InexactNewton : public NewtonSolver {

public:

    /**
     * @brief Constructor for Inexact Newton Solver
     * @param eta_init - initial value of eta, the forcing parameter
     * @param eta_maximum - maximum value of eta
     * @param ared_scale - defines target actual reduction in the residual
     * @note this only defines the Inexact Newton parameters; the actual problem is defined by the operator `oper`
     */
    InexactNewton(double eta_init = 1.0e-04,
                  double eta_maximum = 1.0e-01,
                  double ared_scale = 1.0e-04) 
    : eta(eta_init), eta_max(eta_maximum), t(ared_scale)
    { }

    /**
     * @brief Constructor for Inexact Newton Solver
     * @param comm - a MPI communicator
     * @param eta_init - initial value of eta, the forcing parameter
     * @param eta_maximum - maximum value of eta
     * @param ared_scale - defines target actual reduction in the residual
     * @note this only defines the Inexact Newton parameters; the actual problem is defined by the operator `oper`
     */
    InexactNewton(MPI_Comm comm, 
                  double eta_init = 1.0e-04,
                  double eta_maximum = 1.0e-01,
                  double ared_scale = 1.0e-04)
    : NewtonSolver(comm), eta(eta_init), eta_max(eta_maximum), t(ared_scale)
    { }

    /**
     * @brief Set the operator that defines the nonlinear system
     * @param op - problem operator `r` in `r(x) = b`
     */
    void SetOperator(const Operator& op) override;

    
    /**
     * @brief Solve the nonlinear system with right-hand side b
     * @param b - the right hand side mfem::Vector (can be zero)
     * @param x - initial "guess" for solution
     */
    void Mult(const mfem::Vector& b, mfem::Vector& x) const override;

    Solver *GetSolver() { return prec; }

protected:
    /// Jacobian of the nonlinear operator; needed by ComputeStepSize();
    mutable mfem::Operator* jac;

    /// member mfem::Vector saves the new x position
    mutable mfem::Vector x_new;

    /// Parameters for inexact newton's method
    mutable double eta, eta_max, t;
    const double theta_min = 0.1;
    const double theta_max = 0.5;

private:
    // Explicitly hide NewtonSolver's const qualified Mult method
    using NewtonSolver::Mult;

    /**
     * @brief Back tracking globalization
     * @param x - current solution
     * @param b - the right-hand side mfem::Vector in `r(x) = b`
     * @param norm - norm of the current residual, `||r(x)||`
     * @returns the globalized step size
     * @warning `prec` must be 
     * @note See Pawlowski et al., doi:10.1137/S0036144504443511 for details 
     * regarding the line search method and its parameters.
     */
    double ComputeStepSize(const mfem::Vector& x,
                           const mfem::Vector& b,
                           double norm) const;
};


inline double InexactNewton::ComputeStepSize(const mfem::Vector& x,
                                      const mfem::Vector& b,
                                      const double norm) const 
{
    double theta = 0.0;
    double s = 1.0;
    // p0, p1, and p0p are used for quadratic interpolation p(s) in [0,1].
    // p0 is the value of p(0), p0p is the derivative p'(0), and
    // p1 is the value of p(1). */
    // A temporary mfem::Vector for calculating p0p.
    mfem::Vector temp(r.Size());

    double p0 = 0.5 * norm * norm;
    
    // temp = F'(x_i) * r(x_i)
    jac->Mult(r, temp);

    // c is the negative inexact newton step
    double p0p = -Dot(c, temp);

    // Calculate the new norm

    add(x, -1.0, c, x_new);
    oper->Mult(x_new, r);
    const bool have_b = (b.Size() == Height());
    if (have_b) {
        r -= b;
    }
    double err_new = Norm(r);

    // Globalization starts from here
    int itt = 0;
    while (err_new > (1.0 - t * ( 1.0 - theta)) * norm) {
        double p1 = 0.5 * err_new * err_new;
        // Quadratic interpolation between [0, 1]
        theta = quadInterp(0.0, p0, p0p, 1.0, p1);
        theta = (theta > theta_min) ? theta : theta_min;
        theta = (theta < theta_max) ? theta : theta_max;
        // set the new trial step size.
        s *= theta;
        // update eta
        eta = 1 - theta * (1 - eta);
        eta = (eta < eta_max) ? eta : eta_max;
        // re-evaluate the error norm at new x.
        add(x, -s, c, x_new);
        oper->Mult(x_new, r);
        if (have_b)
        {
            r -= b;
        }
        err_new = Norm(r);

        // Check the iteration counts.
        itt++;
        if (itt > max_iter)
        {
            // mfem::mfem_error("Fail to globalize: Exceed maximum iterations.\n");
            std::cerr << "Fail to globalize: Exceed maximum iterations.\n";
            // throw std::runtime_error("Fail to globalize: Exceed maximum iterations.\n");
            break;
        }
    }
    if (print_level >= 0) {
        mfem::out << " Globalization factors: theta= " << s << ", eta= " << eta
                << '\n';
    }
    return s;

}

inline void InexactNewton::SetOperator(const mfem::Operator& op) {
    oper = &op;
    height = op.Height();
    width = op.Width();
    MFEM_ASSERT(height == width, "square operation is required.");
    r.SetSize(width);
    c.SetSize(width);
    x_new.SetSize(width);
}

inline void InexactNewton::Mult(const mfem::Vector& b,
                         mfem::Vector& x) const 
{
    MFEM_ASSERT(oper != nullptr, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT(prec != nullptr, "the Solver is not set (use SetSolver).");

    const bool have_b = (b.Size() == Height());
    if(!iterative_mode) {
        x = 0.0;
    }
    oper->Mult(x, r);
    if (have_b) {
        r -= b;
    }

    double norm = Norm(r);
    double norm0 = norm;
    double norm_goal = std::max(rel_tol * norm, abs_tol);
    prec->iterative_mode = false;
    // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
    for (int it = 0; true; it++) {
        if( !mfem::IsFinite(norm) ) {
            std::cerr << "norm of the residual is not finite! \n";
            break;
        }
        // MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
        if (print_level >= 0) {
            mfem::out << "Inexact Newton iteration " << std::setw(2) << it << " : ||r|| = " << norm;
            if (it > 0) {
                mfem::out << ", ||r||/||r_0|| = " << norm / norm0;
            }
            mfem::out << "\n";
        }
        if (norm <= norm_goal) {
            converged = 1;
            final_iter= it;
            break;
        }
        if (it >= max_iter) {
            converged = 0;
            final_iter = it;
            break;
        }
        jac = &oper->GetGradient(x);
        prec->SetOperator(*jac);
        prec->Mult(r, c); // c = [DF(x_i)]^{-1} [F(x_i)-b]
        double c_scale = InexactNewton::ComputeStepSize(x, b, norm);
        if (c_scale == 0.0) {
            converged = 0;
            final_iter = it;
            break;
        }
        add(x, -c_scale, c, x);
        oper->Mult(x, r);
        if (have_b){
            r -= b;
        }
        norm = Norm(r);
    }
    final_norm = norm;
}


#endif // INEXACT_NEWTON_HPP