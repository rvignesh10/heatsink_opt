#ifndef THERMAL_OUTPUTS_HPP
#define THERMAL_OUTPUTS_HPP

#include "mfem.hpp"
#include "SingularLoadIntegrators.hpp" // Assumed to exist
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include "../utils/utils.hpp" // Assumed to exist

inline double calcBoundaryMeanTemperature(mfem::Mesh& mesh,
                                            std::unique_ptr<mfem::FiniteElementSpace>& fespace,
                                            std::unique_ptr<mfem::GridFunction>& Tstar_gf,
                                            mfem::Array<int>& bdr_arr)
{
    MFEM_ASSERT(mesh.bdr_attributes.Max() == bdr_arr.Size(), "Boundary array is not of the same size as mesh boundary attributes");

    mfem::ConstantCoefficient beta(1.0);

    mfem::LinearForm lf(fespace.get());
    lf.AddBdrFaceIntegrator(new mfem::BoundaryLFIntegrator(beta), bdr_arr);
    lf.Assemble();
    
    double Area = lf.Sum();
    return ( lf * (*Tstar_gf.get()) ) / Area;
}

inline Eigen::VectorXd calcBoundaryMeanTemperatureGradient(mfem::Mesh& mesh,
                                            std::unique_ptr<mfem::FiniteElementSpace>& fespace,
                                            std::unique_ptr<mfem::GridFunction>& Tstar_gf,
                                            mfem::Array<int>& bdr_arr)
{
    MFEM_ASSERT(mesh.bdr_attributes.Max() == bdr_arr.Size(), "Boundary array is not of the same size as mesh boundary attributes");

    mfem::ConstantCoefficient beta(1.0);

    mfem::LinearForm lf(fespace.get());
    lf.AddBdrFaceIntegrator(new mfem::BoundaryLFIntegrator(beta), bdr_arr);
    lf.Assemble();
    
    double Area = lf.Sum();

    mfem::Vector dTmean_star_dThs_star(Tstar_gf->Size());
    dTmean_star_dThs_star = lf;
    dTmean_star_dThs_star *= 1.0 / Area;

    Eigen::VectorXd dTmean_dThs(Tstar_gf->Size());
    CopyMFEMVectorToEigen(dTmean_dThs, dTmean_star_dThs_star);
    return dTmean_dThs;
}

inline double calcBoundaryFlux(mfem::Mesh& mesh, 
                                mfem::FiniteElementSpace* fespace,
                                mfem::Coefficient& QCoeff,
                                mfem::GridFunction& Tstar_gf,
                                mfem::Array<int>& bdr_arr) {
    MFEM_ASSERT(mesh.bdr_attributes.Max() == bdr_arr.Size(), "Boundary array is not of the same size as mesh boundary attributes");
    
    mfem::LinearForm lf(fespace);
    double sigma = -1.0;
    double kappa = 0.0;
    mfem::ConstantCoefficient beta(1.0);

    // 1. Integrator for Flux Term (Weak Dirichlet)
    lf.AddBdrFaceIntegrator(new mfem::DGDirichletLFIntegrator(beta, QCoeff, sigma, kappa), bdr_arr);
    lf.Assemble();

    // 2. Integrator for Area/Weighting Term (1.0 * v)
    mfem::LinearForm lf2(fespace);
    lf2.AddBdrFaceIntegrator(new mfem::BoundaryLFIntegrator(beta), bdr_arr);
    lf2.Assemble();

    // Calculate Area = <1, 1> over the boundary (assuming Tstar_gf is basis 1)
    mfem::Vector ones(Tstar_gf.Size());
    ones = 1.0;
    
    // Dot product: <1, lf2> (Incorrect) -> Use the assembled value
    double Area = lf2.Sum(); // Sum the load contributions, or compute a better normalization if needed
    
    // Calculate flux = <T, lf> / Area
    return (lf * Tstar_gf) / Area;
}

inline Eigen::VectorXd calcBoundaryFluxGradient(mfem::Mesh& mesh, 
                                mfem::FiniteElementSpace* fespace,
                                mfem::Coefficient& QCoeff,
                                mfem::Array<int>& bdr_arr) 
{
    MFEM_ASSERT(mesh.bdr_attributes.Max() == bdr_arr.Size(), 
            "Boundary array is not of the same size as mesh boundary attributes");
            
    mfem::LinearForm lf(fespace);
    double sigma = -1.0;
    double kappa = 0.0;
    mfem::ConstantCoefficient beta(1.0);

    // 1. Integrator for Flux Term (Weak Dirichlet)
    lf.AddBdrFaceIntegrator(new mfem::DGDirichletLFIntegrator(beta, QCoeff, sigma, kappa), bdr_arr);
    lf.Assemble();

    // 2. Integrator for Area/Weighting Term
    mfem::LinearForm lf2(fespace);
    lf2.AddBdrFaceIntegrator(new mfem::BoundaryLFIntegrator(beta), bdr_arr);
    lf2.Assemble();

    // Calculate Area/Normalization
    // Correct way to get the total area integral (assuming basis 1)
    double Area = lf2.Sum();

    // Derivative of (lf * T) / Area w.r.t T is just lf / Area
    Eigen::VectorXd d_dTstar(lf.Size());
    
    // CRITICAL FIX: Need to assign the MFEM data to the Eigen vector.
    // 1. Create a copy of the LinearForm vector, scaled by (1/Area).
    mfem::Vector d_dTstar_mfem = lf; // Copy lf to a new vector
    d_dTstar_mfem *= (1.0 / Area);  // Scale it

    // 2. Use the utility function to copy the scaled MFEM vector to Eigen.
    CopyMFEMVectorToEigen(d_dTstar, d_dTstar_mfem); 
    
    return d_dTstar;
}

inline Eigen::VectorXd 
calcEvaporatorWallTemperature(std::unique_ptr<mfem::FindPointsGSLIB>& fp,
                              mfem::Vector& vxyz_star,
                              mfem::Mesh& mesh,
                              mfem::GridFunction& Tstar_gf) 
{

    int dim = mesh.Dimension();
    int n_points = vxyz_star.Size() / dim; // Total number of points
    
    // mfem::FindPointsGSLIB fp;
    // fp.Setup(mesh); 
    fp->SetL2AvgType(mfem::FindPointsGSLIB::ARITHMETIC);
    fp->FindPoints(vxyz_star, mfem::Ordering::byNODES);

    mfem::Vector Twall_star(n_points);
    // Interpolate the GridFunction values at the found points
    fp->Interpolate(Tstar_gf, Twall_star);
    
    Eigen::VectorXd wall_temp(n_points);
    CopyMFEMVectorToEigen(wall_temp, Twall_star); 

    return wall_temp;
}

inline std::vector<std::vector<Eigen::Triplet<double>>> 
calcEvaporatorWallTemperatureJacobian(std::unique_ptr<mfem::FindPointsGSLIB>& fp,
                                      mfem::Vector& vxyz_star,
                                      mfem::Mesh& mesh,
                                      std::unique_ptr<mfem::FiniteElementSpace>& fespace) 
{
    using Triplet = Eigen::Triplet<double>;

    int dim = mesh.Dimension();
    int n_points = vxyz_star.Size() / dim; // Total number of points
    
    // mfem::FindPointsGSLIB fp;
    // fp.Setup(mesh); 
    fp->SetL2AvgType(mfem::FindPointsGSLIB::ARITHMETIC);
    fp->FindPoints(vxyz_star, mfem::Ordering::byNODES);
    
    mfem::Array<int> e_id = fp->GetElem();
    mfem::Vector ref_loc  = fp->GetReferencePosition();

    std::vector<std::vector<Triplet>> ListOf_TripletList;
    ListOf_TripletList.reserve(n_points);

    mfem::IntegrationPoint ip;
    const mfem::FiniteElement* el = nullptr;
    mfem::Vector shape_vec; // Declare outside loop for efficiency
    mfem::Array<int> dof_ids; // Declare outside loop for efficiency

    for (int i=0; i<e_id.Size(); ++i) { // Loop over each point (row in Jacobian)
        std::vector<Triplet> TripletList;
        int elemNo = e_id[i];
        
        double p[4];
        p[0] = ref_loc(dim*i);
        p[1] = ref_loc(dim*i + 1);
        p[2] = ref_loc(dim*i + 2);
        p[3] = 1.0;
        ip.Set3w(p);
        
        // 1. Get the FE and Element Transformation
        el = fespace->GetFE(elemNo);
        if (i == 0) {int ndof = el->GetDof(); shape_vec.SetSize(ndof);}
        shape_vec = 0.0;
        el->CalcShape(ip, shape_vec); 

        // 2. Get DOFs for assembly
        fespace->GetElementDofs(elemNo, dof_ids);

        // 3. Assemble (row 'i' of the Jacobian = shape_vec components)
        for (int j=0; j<dof_ids.Size(); ++j) {
            // Row is the point index (i), Column is the global DOF index
            TripletList.emplace_back(i, dof_ids[j], shape_vec(j));
        }
        ListOf_TripletList.push_back(TripletList);
    }

    return ListOf_TripletList;
}

inline void diffEvaporatorWallTemperature( std::unique_ptr<mfem::FindPointsGSLIB>& fp,
                                           mfem::Vector& vxyz_star,
                                           mfem::Mesh& mesh,
                                           std::unique_ptr<mfem::FiniteElementSpace>& fespace,
                                           std::unique_ptr<mfem::GridFunction>& Tstar_gf,
                                           const Eigen::Ref<const Eigen::VectorXd> vxyz_star_bar,
                                           Eigen::Ref<Eigen::VectorXd> Twall_star_bar )
{   
    int dim = mesh.Dimension();            // 3 dimensional heat sink
    int n_points = vxyz_star.Size() / dim; // Total number of points to calculate wall temperatures 

    MFEM_ASSERT(Twall_star_bar.Size() == n_points, "Gradient Vector Twall_star_bar.Size() != n_points (num_channels * num_nodes_per_channel)");

    // mfem::FindPointsGSLIB fp;
    // fp.Setup(mesh); 
    fp->SetL2AvgType(mfem::FindPointsGSLIB::ARITHMETIC);
    fp->FindPoints(vxyz_star, mfem::Ordering::byNODES);
    
    
    mfem::Array<int> e_id = fp->GetElem();
    mfem::Vector ref_loc  = fp->GetReferencePosition();

    mfem::IntegrationPoint ip;

    mfem::Vector shape_vec, shape_bar; 
    mfem::DenseMatrix dshape_dxi, dshape_dx;

    mfem::Vector Ths_star_elem;
    mfem::Vector xbar(dim); xbar = 0.0;

    for (int i=0; i<e_id.Size(); ++i) {
        int elemNo = e_id[i];

        double p[4];
        p[0] = ref_loc(dim*i);
        p[1] = ref_loc(dim*i + 1);
        p[2] = ref_loc(dim*i + 2);
        p[3] = 1.0;
        ip.Set3w(p);
        
        // 1. Get the FE and Element Transformation
        const mfem::FiniteElement* el = fespace->GetFE(elemNo);
        mfem::ElementTransformation* ElTr = fespace->GetElementTransformation(elemNo);
        ElTr->SetIntPoint(&ip);
        const mfem::DenseMatrix& iJac = ElTr->InverseJacobian();

        int ndof = el->GetDof(); 
        shape_vec.SetSize(ndof); shape_vec = 0.0;
        shape_bar.SetSize(ndof); shape_bar = 0.0;

        dshape_dxi.SetSize(ndof, dim); dshape_dxi = 0.0;
        dshape_dx.SetSize(ndof, dim);  dshape_dx = 0.0;

        Ths_star_elem.SetSize(ndof); Ths_star_elem = 0.0;

        el->CalcShape(ip, shape_vec);      // get shape N(xi) - ndof x 1
        el->CalcDShape(ip, dshape_dxi);    // get dNdxi       - ndof x dim

        // get xbar
        for (int d = 0; d < dim; ++d) {
            xbar(d) = vxyz_star_bar(d * n_points + i);
        }

        mfem::Mult(dshape_dxi, iJac, dshape_dx); // performs dNdxi * dxidx (inv Jac) = dNdx
        dshape_dx.Mult(xbar, shape_bar);            // performs dNdx * xbar (dx_dalpha_k) = Nbar

        Tstar_gf->GetElementDofValues(elemNo, Ths_star_elem); // get Ths_elem (nodal values pertaining to that element)

        Twall_star_bar(i) = shape_bar * Ths_star_elem;      // dot product of Nbar * Ths_elem = dTwall_dalpha_k
    }
    // fp.FreeData();
}

#endif