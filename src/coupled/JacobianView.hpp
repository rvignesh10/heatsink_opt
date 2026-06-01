#ifndef JACOBIAN_VIEW_HPP
#define JACOBIAN_VIEW_HPP

#include <Eigen/Sparse>
#include <vector>
#include <cassert>

// Reuse or define a dynamic layout helper structure
struct DynamicLayoutJac
{
    const int ne;      // Number of elements (runtime)
    const int nf;      // Number of flow states (runtime)
    const int nxi;     // Number of flow points (runtime)
    const int nhs;     // Number of heat sink states (runtime)

    const int block_size;
    const int input_size;
    const int total_size;

    DynamicLayoutJac(int ne_, int nf_, int nxi_, int nhs_)
        : ne(ne_), nf(nf_), nxi(nxi_), nhs(nhs_),
          block_size(nf_ * nxi_),
          input_size(ne_ * nxi_),
          total_size(ne_ * block_size + nhs_)
    {}
};


struct JacobianView : private DynamicLayoutJac
{
    using Triplet = Eigen::Triplet<double>;
    using Layout = DynamicLayoutJac;
    
    // Inherit size members: ne, nf, nxi, nhs, fsize, block_size, total_size

    std::vector<Triplet>& global_triplets;

    // --- MODIFICATION 2: Accept dimensions in the constructor ---
    explicit JacobianView(std::vector<Triplet>& gtrips, int ne_, int nf_, int nxi_, int nhs_)
        : DynamicLayoutJac(ne_, nf_, nxi_, nhs_), // Initialize the dynamic layout
          global_triplets(gtrips)
    {}

    // --- index helpers ---
    int elem_offset(int e) const { return e * block_size; }
    int tail_offset() const      { return ne * block_size; }

    // ------------------------------------------------------
    // 1) element–element diagonal block: (block_size x block_size)
    // ------------------------------------------------------
    void add_block_elem(int e, const std::vector<Triplet>& local_trips)
    {
        assert(e >= 0 && e < ne); // Use runtime 'ne'

        const int base = elem_offset(e);

        for (const auto& t : local_trips) {
            const int lr = t.row();
            const int lc = t.col();
            assert(lr >= 0 && lr < block_size); // Use runtime 'block_size'
            assert(lc >= 0 && lc < block_size);
            global_triplets.emplace_back(base + lr, base + lc, t.value());
        }
    }

    // ------------------------------------------------------
    // 2) element–Ths block: (block_size x nhs)
    // ------------------------------------------------------
    void add_block_elem_Ths(int e, const std::vector<Triplet>& local_trips)
    {
        assert(e >= 0 && e < ne);

        const int ro = elem_offset(e);
        const int co = tail_offset();

        for (const auto& t : local_trips) {
            const int lr = t.row();
            const int lc = t.col();
            assert(lr >= 0 && lr < block_size);
            assert(lc >= 0 && lc < nhs); // Use runtime 'nhs'
            global_triplets.emplace_back(ro + lr, co + lc, t.value());
        }
    }

    // ------------------------------------------------------
    // 3) Ths–element block: (nhs x block_size)
    // ------------------------------------------------------
    void add_block_Ths_elem(int e, const std::vector<Triplet>& local_trips)
    {
        assert(e >= 0 && e < ne);

        const int ro = tail_offset();
        const int co = elem_offset(e);

        for (const auto& t : local_trips) {
            const int lr = t.row();
            const int lc = t.col();
            assert(lr >= 0 && lr < nhs);
            assert(lc >= 0 && lc < block_size);
            global_triplets.emplace_back(ro + lr, co + lc, t.value());
        }
    }

    void add_Ths_elems(const std::vector<Triplet>& trips) {
        const int ro = tail_offset();
        for (const auto& t : trips) {
            const int lr = t.row();
            const int lc = t.col();
            assert(lr >= 0 && lr < nhs);
            assert(lc >= 0 && lc < ne*block_size);
            global_triplets.emplace_back(ro + lr, lc, t.value());
        }
    }

    // ------------------------------------------------------
    // 4) Ths–Ths block: (nhs x nhs)
    // ------------------------------------------------------
    void add_block_Ths_Ths(const std::vector<Triplet>& local_trips)
    {
        const int base = tail_offset();

        for (const auto& t : local_trips) {
            const int lr = t.row();
            const int lc = t.col();
            assert(lr >= 0 && lr < nhs);
            assert(lc >= 0 && lc < nhs);
            global_triplets.emplace_back(base + lr, base + lc, t.value());
        }
    }
};

struct InputView : private DynamicLayoutJac {
    using Triplet = Eigen::Triplet<double>;
    using Layout = DynamicLayoutJac;
    
    // Inherit size members: ne, nf, nxi, nhs, fsize, block_size, total_size

    std::vector<Triplet>& global_triplets;

    explicit InputView(std::vector<Triplet>& gtrips, int ne_, int nf_, int nxi_, int nhs_)
        : DynamicLayoutJac(ne_, nf_, nxi_, nhs_), // Initialize the dynamic layout
          global_triplets(gtrips)
    {}

    int row_offset(int e) {return e * nxi;}
    int col_offset(int e) {return e * block_size;}

    /**
     * @brief sets the local input matrix of size [nxi, block_size] into the global input of size 
     * [ne*nxi, ne*(block_size)]
     * @param e evaporator index for which we set the block
     * @param local_trips a list of triplets that holds the local indexing for local input matrix
     */
    void set_input_jac(int e, const std::vector<Triplet>& local_trips) {
        assert(e >= 0 && e < ne);
        int roffset = row_offset(e);
        int coffset = col_offset(e);
        for (const auto& t : local_trips) {
            const int lr = t.row();
            const int lc = t.col();
            assert(lr >= 0 && lr < nxi);
            assert(lc >= 0 && lc < block_size);
            global_triplets.emplace_back(roffset + lr, coffset + lc, t.value());
        }
    }

};

#endif // JACOBIAN_VIEW_HPP