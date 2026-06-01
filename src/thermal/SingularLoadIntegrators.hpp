#ifndef SINGULAR_LOAD_INTEGRATORS
#define SINGULAR_LOAD_INTEGRATORS

#include "mfem.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <utility>


struct DesignParameters {
    int ne;
    int nxi;

    std::string material;
    double kappa;
    const int getNumOfChannels() const {return ne;}
};


struct MeshParameters {
    int num_x;
    int num_y;
    int num_z;

    double x_domain;
    double y_domain;
    double z_domain;

    std::string elem_type;
};


struct LoadParameters {
    Eigen::VectorXd vxyz_star;
    Eigen::VectorXd qlf_star;
    Eigen::VectorXd qblf_star;
};


struct ThermalBoundaryConditions {
    // For original Dirichlet Setup
    double Tsrc_star = 0.0;
    int Tsrc_bdr_id = 0; // 1-based index from mesh attributes
    
    bool f_in_bcs = false;
    double Tin_star = 0.0;
    int Tin_bdr_id = 0;

    // For Robin/Flux Setup
    double h_star = 0.0;       // Convection coeff
    double Tinf_star = 0.0;    // Ambient Temp
    double qflux_star = 0.0;  // Constant Flux
    
    // Boundary Attribute IDs (1-based from mesh)
    int robin_bdr_id = 0; // e.g., x=0
    int flux_bdr_id = 0;  // e.g., z=0
};

inline std::pair<mfem::Array<int>, mfem::Vector>
FindPointsAndReferenceLocations (std::unique_ptr<mfem::FindPointsGSLIB>& fp, mfem::Vector& vxyz_star) {

    // mfem::FindPointsGSLIB fp;
    // fp.Setup(mesh);
    fp->SetL2AvgType(mfem::FindPointsGSLIB::ARITHMETIC);
    fp->FindPoints(vxyz_star, mfem::Ordering::byNODES);
    mfem::Array<int> e_id = fp->GetElem();
    mfem::Vector ref_loc= fp->GetReferencePosition();

    return std::make_pair(e_id, ref_loc);
}

inline Eigen::VectorXd
calcIntegrationWeights(mfem::Vector& vxyz_star, const int dim) {
    int n = vxyz_star.Size() / dim;
    Eigen::VectorXd wxyz_star(n);
    wxyz_star.setZero();

    for (int i=0; i<n; ++i) {
        if (i == 0) {
            double x0 = vxyz_star(i);
            double y0 = vxyz_star(n+i);
            double z0 = vxyz_star(2*n+i);

            double x1 = vxyz_star(i+1);
            double y1 = vxyz_star(n+i+1);
            double z1 = vxyz_star(2*n+i+1);
            wxyz_star(i) = std::sqrt( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0) );
        }
        else if (i == n-1)
        {
            double x0 = vxyz_star(i-1);
            double y0 = vxyz_star(n+i-1);
            double z0 = vxyz_star(2*n+i-1);

            double x1 = vxyz_star(i);
            double y1 = vxyz_star(n+i);
            double z1 = vxyz_star(2*n+i);
            wxyz_star(i) = std::sqrt( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0) );
        }
        else {
            double xm = vxyz_star(i-1);
            double ym = vxyz_star(n+i-1);
            double zm = vxyz_star(2*n+i-1);

            double x = vxyz_star(i);
            double y = vxyz_star(n+i);
            double z = vxyz_star(2*n+i);

            double xp = vxyz_star(i+1);
            double yp = vxyz_star(n+i+1);
            double zp = vxyz_star(2*n+i+1);

            double w1 = std::sqrt( (x-xm)*(x-xm) + (y-ym)*(y-ym) + (z-zm)*(z-zm) );
            double w2 = std::sqrt( (xp-x)*(xp-x) + (yp-y)*(yp-y) + (zp-z)*(zp-z) );
            wxyz_star(i) = 0.5 * (w1 + w2);
        }
        
    }
    return wxyz_star;
}

inline Eigen::VectorXd
diffIntegrationWeights(mfem::Vector& vxyz_star, 
                       const Eigen::Ref<const Eigen::VectorXd> vbar_xyz, 
                       const int dim) {
    MFEM_ASSERT(vxyz_star.Size() == vbar_xyz.size(), 
                "vxyz_star.Size() != vxyz_star_bar.size(), they should have the same length");
    
    int n = vxyz_star.Size() / dim;
    Eigen::VectorXd wbar(n);
    wbar.setZero();

    for (int i=0; i<n; ++i) {
        if (i == 0) {
            double x0 = vxyz_star(i);      double dx0 = vbar_xyz(i);
            double y0 = vxyz_star(n+i);    double dy0 = vbar_xyz(n+i);
            double z0 = vxyz_star(2*n+i);  double dz0 = vbar_xyz(2*n+i);

            double x1 = vxyz_star(i+1);    double dx1 = vbar_xyz(i+1);
            double y1 = vxyz_star(n+i+1);  double dy1 = vbar_xyz(n+i+1);
            double z1 = vxyz_star(2*n+i+1);double dz1 = vbar_xyz(2*n+i+1);

            double wxyz_star = std::sqrt( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0) );
            wbar(i) = ( (x1-x0)*(dx1-dx0) + (y1-y0)*(dy1-dy0) + (z1-z0)*(dz1-dz0) ) / wxyz_star;
        }
        else if (i == n-1)
        {
            double x0 = vxyz_star(i-1);      double dx0 = vbar_xyz(i-1);
            double y0 = vxyz_star(n+i-1);    double dy0 = vbar_xyz(n+i-1);
            double z0 = vxyz_star(2*n+i-1);  double dz0 = vbar_xyz(2*n+i-1);

            double x1 = vxyz_star(i);    double dx1 = vbar_xyz(i);
            double y1 = vxyz_star(n+i);  double dy1 = vbar_xyz(n+i);
            double z1 = vxyz_star(2*n+i);double dz1 = vbar_xyz(2*n+i);

            double wxyz_star = std::sqrt( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0) );
            wbar(i) = ( (x1-x0)*(dx1-dx0) + (y1-y0)*(dy1-dy0) + (z1-z0)*(dz1-dz0) ) / wxyz_star;
        }
        else {
            double xm = vxyz_star(i-1);      double dxm = vbar_xyz(i-1);
            double ym = vxyz_star(n+i-1);    double dym = vbar_xyz(n+i-1);
            double zm = vxyz_star(2*n+i-1);  double dzm = vbar_xyz(2*n+i-1);

            double x = vxyz_star(i);         double dx = vbar_xyz(i);
            double y = vxyz_star(n+i);       double dy = vbar_xyz(n+i);
            double z = vxyz_star(2*n+i);     double dz = vbar_xyz(2*n+i);

            double xp = vxyz_star(i+1);      double dxp = vbar_xyz(i+1);
            double yp = vxyz_star(n+i+1);    double dyp = vbar_xyz(n+i+1);
            double zp = vxyz_star(2*n+i+1);  double dzp = vbar_xyz(2*n+i+1);

            double w1 = std::sqrt( (x-xm)*(x-xm) + (y-ym)*(y-ym) + (z-zm)*(z-zm) );
            double dw1= ( (x-xm)*(dx-dxm) + (y-ym)*(dy-dym) + (z-zm)*(dz-dzm) ) / w1;
            double w2 = std::sqrt( (xp-x)*(xp-x) + (yp-y)*(yp-y) + (zp-z)*(zp-z) );
            double dw2= ( (xp-x)*(dxp-dx) + (yp-y)*(dyp-dy) + (zp-z)*(dzp-dz) ) / w2;
            wbar(i) = 0.5 * (dw1 + dw2);
        }
        
    }
    return wbar;
}

/**
 * @brief This class performs the BilinearForm integration for the singular loads present in the heat-sink.
 * Since the integral is ∫ v(s) * q(s) * u(s) ds, which becomes Σ w_i(xi_i) * v(xi_i) * q_i(xi_i) * u(xi_i)
 */
class SingularBLFIntegrator : public mfem::BilinearFormIntegrator {
private:
    static constexpr int dim = 3;
    mfem::Vector vxyz_star;
    mfem::Vector qblf_star;
    
    Eigen::VectorXd wxyz_star;
    mfem::Array<int> e_id;
    mfem::Vector ref_loc;
public:
    SingularBLFIntegrator(std::unique_ptr<mfem::FindPointsGSLIB>& fp,
                          mfem::Vector& vxyz_star_, 
                          mfem::Vector& qblf_star_):
                          vxyz_star(vxyz_star_), qblf_star(qblf_star_) {
        auto [e_id, ref_loc] = FindPointsAndReferenceLocations(fp, vxyz_star);
        this->e_id = e_id; this->ref_loc = ref_loc;
        wxyz_star = calcIntegrationWeights(vxyz_star, dim);
    }

    ~SingularBLFIntegrator() {}

    virtual void AssembleElementMatrix(const mfem::FiniteElement &el, 
                                        mfem::ElementTransformation &Tr, 
                                        mfem::DenseMatrix &elmat) {
        // Find the Total DOF in the element domain.
        int dof = el.GetDof();
        // Find the element Number for which integartion is performed over the domain volume
        int ElemNumber = Tr.ElementNo;
        elmat.SetSize(dof);
        elmat = 0.0;
        
        // We can't pre-size, so we just let Append handle allocation.
        mfem::Array<int> idx_found_elem;
        idx_found_elem.SetSize(0); 

        for (int i = 0; i < e_id.Size(); i++) {
            if (e_id[i] == ElemNumber) {
                // Found an IP that belongs to this element.
                // Append its index 'i' to our list.
                idx_found_elem.Append(i);
            }
        }
        int num_ip = idx_found_elem.Size();

        // This ip object will be re-used
        mfem::IntegrationPoint ip;
        mfem::Vector shape_vec(dof);
        
        // 2. Loop over only the IPs that are inside this element
        for (int j = 0; j < num_ip; j++) {
            
            // This is the global index for this integration point
            int ip_index = idx_found_elem[j];
            double p[4];
            p[0] = ref_loc( dim * ip_index );     // ref_x
            p[1] = ref_loc( dim * ip_index + 1 ); // ref_y
            p[2] = ref_loc( dim * ip_index + 2 ); // ref_z
            p[3] = wxyz_star( ip_index );       // weight
            
            // Set the x,y,z, and weight of the integration point
            ip.Set3w(p);

            // 3. Now, do the assembly
            
            // Get the basis function values at this IP
            shape_vec = 0.0;
            el.CalcShape(ip, shape_vec);
            
            // Add to the local element matrix
            mfem::AddMult_a_VVt(ip.weight * qblf_star( ip_index ), shape_vec, elmat);
            
        }

    }

    Eigen::SparseMatrix<double> calcLoadJacobian(std::unique_ptr<mfem::FiniteElementSpace>& fespace,
                                                 std::unique_ptr<mfem::GridFunction>& Tstar_gf) {
        
        Eigen::SparseMatrix<double> dKThs_dqblf;
        dKThs_dqblf.resize(fespace->GetNVDofs(), e_id.Size());

        std::vector<Eigen::Triplet<double>> trips;
    
        mfem::IntegrationPoint ip;
        const mfem::FiniteElement* el = nullptr;
        mfem::DenseMatrix elmat;
        mfem::Vector shape_vec;
        mfem::Vector Tstar_elem;
        mfem::Vector wNNtThs;
        mfem::Array<int> dof_ids;
        
        // Loop over all load elements present 
        for (int j=0; j<e_id.Size(); ++j) {
            int elemNo = e_id[j];
            double p[4];
            p[0] = ref_loc( dim * j );
            p[1] = ref_loc( dim * j + 1 );
            p[2] = ref_loc( dim * j + 2 );
            p[3] = wxyz_star(j);
            ip.Set3w(p);

            el = fespace->GetFE(elemNo);
            int ndof = el->GetDof();

            if (j == 0) {
                // only set size once
                shape_vec.SetSize(ndof);
                wNNtThs.SetSize(ndof); 
                elmat.SetSize(ndof, ndof); 
            }

            shape_vec = 0.0;
            el->CalcShape(ip, shape_vec);

            fespace->GetElementDofs(elemNo, dof_ids);
            Tstar_gf->GetElementDofValues(elemNo, Tstar_elem);

            assert(shape_vec.Size() == Tstar_elem.Size());
            
            elmat = 0.0;
            mfem::AddMult_a_VVt(ip.weight, shape_vec, elmat);
            wNNtThs = 0.0;
            elmat.Mult(Tstar_elem, wNNtThs);
            for (int i=0; i<ndof; ++i) {
                trips.emplace_back(dof_ids[i], j, wNNtThs(i));
            }
        }

        dKThs_dqblf.setFromTriplets(trips.begin(), trips.end());
        dKThs_dqblf.makeCompressed();
        return dKThs_dqblf;
    }

    double calcTotalGradient( std::unique_ptr<mfem::FiniteElementSpace>& fespace,
                              const Eigen::Ref<const Eigen::VectorXd> vxyz_star_bar,
                              const Eigen::Ref<const Eigen::VectorXd> qblf_star_bar,
                              std::unique_ptr<mfem::GridFunction>& adjT_gf,
                              std::unique_ptr<mfem::GridFunction>& Tstar_gf ) 
    {   
        MFEM_ASSERT(this->wxyz_star.size() == qblf_star_bar.size(), 
                    "wxyz_star.size() != qblf_star.size(), load_bar and weights pertaining to the load should have the same length");

        double blf_bar = 0.0;
        auto wxyz_star_bar = diffIntegrationWeights(this->vxyz_star, vxyz_star_bar, this->dim);
        
        mfem::IntegrationPoint ip;
        const mfem::FiniteElement* el = nullptr;
        
        mfem::Vector shape_vec, shape_bar;
        mfem::DenseMatrix dshape_dxi, dshape_dx;

        mfem::Vector xbar(this->dim); xbar = 0.0;
        mfem::Vector Ths_elem, adjT_elem, a_elem;

        const int n_points = this->vxyz_star.Size() / this->dim;

        for (int j=0; j<n_points; ++j) {
            int elemNo = this->e_id[j];

            double p[4];
            p[0] = ref_loc( this->dim * j );
            p[1] = ref_loc( this->dim * j + 1 );
            p[2] = ref_loc( this->dim * j + 2 );
            p[3] = wxyz_star(j);
            ip.Set3w(p);

            // get xbar
            for (int d = 0; d < dim; ++d) {
                xbar(d) = vxyz_star_bar(d * n_points + j);
            }

            el = fespace->GetFE(elemNo);
            mfem::ElementTransformation* ElTr = fespace->GetElementTransformation(elemNo);
            ElTr->SetIntPoint(&ip);
            const mfem::DenseMatrix& iJac = ElTr->InverseJacobian();
            
            int ndof = el->GetDof();
            
            shape_vec.SetSize(ndof); shape_vec = 0.0; 
            shape_bar.SetSize(ndof); shape_bar = 0.0;

            dshape_dxi.SetSize(ndof, this->dim); dshape_dxi = 0.0;
            dshape_dx.SetSize(ndof, this->dim); dshape_dx = 0.0;

            el->CalcShape(ip, shape_vec);
            el->CalcDShape(ip, dshape_dxi);

            mfem::Mult(dshape_dxi, iJac, dshape_dx); // dN_dx = dN_dxi * iJac 
            dshape_dx.Mult(xbar, shape_bar);         // Nbar  = dN_dx * xbar

            Ths_elem.SetSize(ndof); Ths_elem = 0.0;
            Tstar_gf->GetElementDofValues(elemNo, Ths_elem);

            adjT_elem.SetSize(ndof); adjT_elem = 0.0;
            adjT_gf->GetElementDofValues(elemNo, adjT_elem);

            a_elem.SetSize(ndof); a_elem = 0.0;
            
            a_elem.Add(wxyz_star_bar(j) * this->qblf_star(j) * (shape_vec * Ths_elem), shape_vec);
            a_elem.Add(this->wxyz_star(j) * qblf_star_bar(j) * (shape_vec * Ths_elem), shape_vec);
            a_elem.Add(this->wxyz_star(j) * this->qblf_star(j) * (shape_vec * Ths_elem), shape_bar);
            a_elem.Add(this->wxyz_star(j) * this->qblf_star(j) * (shape_bar * Ths_elem), shape_vec);

            blf_bar += adjT_elem * a_elem;
        }
    
        return blf_bar;

    }

};


/**
 * @brief This class performs the linear form integration for singular loads present in the heat-sink.
 * Since the integral is ∫ v(s) * q(s) ds, which becomes Σ w_i(xi_i) * v(xi_i) * q_i(xi_i) 
 */
class SingularLFIntegrator : public mfem::LinearFormIntegrator {
private:
    static constexpr int dim = 3;
    mfem::Vector vxyz_star;
    mfem::Vector qlf_star;
    
    Eigen::VectorXd wxyz_star;
    mfem::Array<int> e_id;
    mfem::Vector ref_loc;
public:
    SingularLFIntegrator( std::unique_ptr<mfem::FindPointsGSLIB>& fp,
                          mfem::Vector& vxyz_star_, 
                          mfem::Vector& qlf_star_) :
                         vxyz_star(vxyz_star_), qlf_star(qlf_star_) {

        auto [e_id, ref_loc] = FindPointsAndReferenceLocations(fp, vxyz_star);
        this->e_id = e_id; this->ref_loc = ref_loc;
        wxyz_star = calcIntegrationWeights(vxyz_star, dim);
    }
    ~SingularLFIntegrator() {}
    virtual void AssembleRHSElementVect( const mfem::FiniteElement &el, 
                                        mfem::ElementTransformation &Tr, 
                                        mfem::Vector &elvect ) {
        // Find the Total DOF in the element domain.
        int dof = el.GetDof();
        // Find the element Number for which integartion is performed over the domain volume
        int ElemNumber = Tr.ElementNo;
        // resize elvect and set to 0
        elvect.SetSize(dof);
        elvect = 0.0;

        // We can't pre-size, so we just let Append handle allocation.
        mfem::Array<int> idx_found_elem;
        idx_found_elem.SetSize(0); 

        // Loop through ALL integration points from the line load
        for (int i = 0; i < e_id.Size(); i++) {
            if (e_id[i] == ElemNumber) {
                // Found an IP that belongs to this element.
                // Append its index 'i' to our list.
                idx_found_elem.Append(i);
            }
        }
    
        int num_ip = idx_found_elem.Size();

        // This ip object will be re-used
        mfem::IntegrationPoint ip;
        mfem::Vector shape_vec(dof);

        // 2. Loop over only the IPs that are inside this element
        for (int j = 0; j < num_ip; j++) {
            
            // This is the global index for this integration point
            int ip_index = idx_found_elem[j];
            double p[4];
            p[0] = ref_loc( dim * ip_index );     // ref_x
            p[1] = ref_loc( dim * ip_index + 1 ); // ref_y
            p[2] = ref_loc( dim * ip_index + 2 ); // ref_z
            p[3] = wxyz_star( ip_index );       // weight
            
            // Set the x,y,z, and weight of the integration point
            ip.Set3w(p);
            
            // 3. Now, do the assembly
            
            // Get the basis function values at this IP
            shape_vec = 0.0;
            el.CalcShape(ip, shape_vec);

            // Your "kernel" is just the basis functions,
            // since the integral is ∫ q(s) * u(s) ds,
            // which becomes Σ w_i * q_i * u(x_i)
            shape_vec *= ip.weight * qlf_star( ip_index );
            elvect += shape_vec;
        }
    }

    Eigen::SparseMatrix<double> calcLoadJacobian(std::unique_ptr<mfem::FiniteElementSpace>& fespace) {
        
        Eigen::SparseMatrix<double> db_dqlf;
        db_dqlf.resize(fespace->GetNVDofs(), e_id.Size());

        std::vector<Eigen::Triplet<double>> trips;
        mfem::IntegrationPoint ip;
        const mfem::FiniteElement* el = nullptr;
        mfem::Vector shape_vec;
        mfem::Array<int> dof_ids;

        // Loop over all load elements present 
        for (int j=0; j<e_id.Size(); ++j) {
            int elemNo = e_id[j];
            double p[4];
            p[0] = ref_loc( dim * j );
            p[1] = ref_loc( dim * j + 1 );
            p[2] = ref_loc( dim * j + 2 );
            p[3] = wxyz_star(j);
            ip.Set3w(p);

            el = fespace->GetFE(elemNo);

            int ndof = el->GetDof();
            if (j == 0) shape_vec.SetSize(ndof);

            shape_vec = 0.0;
            el->CalcShape(ip, shape_vec);
            
            fespace->GetElementDofs(elemNo, dof_ids);
            
            for (int i=0; i<ndof; ++i) {
                trips.emplace_back( dof_ids[i], j, ip.weight * shape_vec(i) );
            }
        }

        db_dqlf.setFromTriplets(trips.begin(), trips.end());
        db_dqlf.makeCompressed();
        return db_dqlf;
    }

    double calcTotalGradient( std::unique_ptr<mfem::FiniteElementSpace>& fespace,
                              const Eigen::Ref<const Eigen::VectorXd> vxyz_star_bar,
                              const Eigen::Ref<const Eigen::VectorXd> qlf_star_bar,
                              std::unique_ptr<mfem::GridFunction>& adjT_gf ) 
    {   
        MFEM_ASSERT(this->wxyz_star.size() == qlf_star_bar.size(), 
                    "wxyz_star.size() != qlf_star.size(), load_bar and weights pertaining to the load should have the same length");
        double lf_bar = 0.0;
        auto wxyz_star_bar = diffIntegrationWeights(this->vxyz_star, vxyz_star_bar, this->dim);

        mfem::IntegrationPoint ip;
        const mfem::FiniteElement* el = nullptr;
        
        mfem::Vector shape_vec, shape_bar;
        mfem::DenseMatrix dshape_dxi, dshape_dx;

        mfem::Vector xbar(this->dim); xbar = 0.0;
        mfem::Vector adjT_elem, b_elem;

        const int n_points = this->vxyz_star.Size() / this->dim;

        for (int j=0; j<n_points; ++j) {
            int elemNo = this->e_id[j];

            double p[4];
            p[0] = ref_loc( this->dim * j );
            p[1] = ref_loc( this->dim * j + 1 );
            p[2] = ref_loc( this->dim * j + 2 );
            p[3] = wxyz_star(j);
            ip.Set3w(p);

            // get xbar
            for (int d = 0; d < dim; ++d) {
                xbar(d) = vxyz_star_bar(d * n_points + j);
            }

            el = fespace->GetFE(elemNo);
            mfem::ElementTransformation* ElTr = fespace->GetElementTransformation(elemNo);
            ElTr->SetIntPoint(&ip);
            const mfem::DenseMatrix& iJac = ElTr->InverseJacobian();
            
            int ndof = el->GetDof();
            
            shape_vec.SetSize(ndof); shape_vec = 0.0; 
            shape_bar.SetSize(ndof); shape_bar = 0.0;

            dshape_dxi.SetSize(ndof, this->dim); dshape_dxi = 0.0;
            dshape_dx.SetSize(ndof, this->dim); dshape_dx = 0.0;

            el->CalcShape(ip, shape_vec);
            el->CalcDShape(ip, dshape_dxi);

            mfem::Mult(dshape_dxi, iJac, dshape_dx); // dN_dx = dN_dxi * iJac 
            dshape_dx.Mult(xbar, shape_bar);         // Nbar  = dN_dx * xbar

            adjT_elem.SetSize(ndof); adjT_elem = 0.0;
            adjT_gf->GetElementDofValues(elemNo, adjT_elem);

            b_elem.SetSize(ndof); b_elem = 0.0;

            b_elem.Add(wxyz_star_bar(j) * this->qlf_star(j), shape_vec);
            b_elem.Add(this->wxyz_star(j) * qlf_star_bar(j), shape_vec);
            b_elem.Add(this->wxyz_star(j) * this->qlf_star(j), shape_bar);
            
            lf_bar += adjT_elem * b_elem;
        }

        return lf_bar;
    }

};

#endif // SINGULAR_LOAD_INTEGRATORS
