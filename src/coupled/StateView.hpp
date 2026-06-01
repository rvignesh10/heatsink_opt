// #ifndef STATE_VIEW_HPP
// #define STATE_VIEW_HPP

// #include <Eigen/Dense>
// #include <cassert>

// // --- MODIFICATION 1: Dynamic Layout Structure ---
// // No longer a template; holds dimensions as runtime members.
// struct DynamicLayout
// {
//     const int ne;      // Number of elements (runtime)
//     const int nf;      // Number of flow states (runtime)
//     const int nxi;     // Number of flow points (runtime)
//     const int nhs;     // Number of heat sink states (runtime)

//     const int block_size;
//     const int total_size;

//     DynamicLayout(int ne_, int nf_, int nxi_, int nhs_)
//         : ne(ne_), nf(nf_), nxi(nxi_), nhs(nhs_),
//           block_size(nf_ * nxi_),
//           total_size(ne_ * block_size + nhs_)
//     {}
// };

// // ====================== STATE VIEW (x) ======================
// // ===================== RESIDUAL VIEW (R) =====================

// struct StateView : private DynamicLayout
// {
//     using Layout = DynamicLayout;
//     // We inherit all size constants (ne, nf, nxi, etc.)

//     Eigen::VectorXd& state;    // x

//     // --- MODIFICATION 3: Accept dimensions in the constructor ---
//     StateView(Eigen::VectorXd& state_vec, int ne_, int nf_, int nxi_, int nhs_)
//         : DynamicLayout(ne_, nf_, nxi_, nhs_), // Initialize the layout
//           state(state_vec)
//     {
//         assert(state.size() == total_size);
//     }

//     StateView(const Eigen::VectorXd& state_vec, int ne_, int nf_, int nxi_, int nhs_)
//     : DynamicLayout(ne_, nf_, nxi_, nhs_), state(const_cast<Eigen::VectorXd&>(state_vec)) // requires state member to be non-const
//     {
//         assert(state.size() == total_size);
//     }

//     // fstate as nf x nxi matrix for element e
//     // Use Eigen::Dynamic to allow runtime sizes in Map
//     Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> fstate_mat(int e)
//     {
//         return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(
//             state.data() + e * block_size, nf, nxi // Pass nf and nxi (runtime sizes)
//         );
//     }

//     const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> fstate_mat(int e) const
//     {
//         return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(
//             state.data() + e * block_size, nf, nxi // Pass nf and nxi
//         );
//     }


//     // Ths_star tail (size nhs)
//     // Use Eigen::Dynamic for the row size
//     Eigen::Map<Eigen::VectorXd> Ths()
//     {
//         return Eigen::Map<Eigen::VectorXd>(
//             state.data() + ne * block_size, nhs
//         );
//     }

//     const Eigen::Map<const Eigen::VectorXd> Ths() const
//     {
//         return Eigen::Map<const Eigen::VectorXd>(
//             state.data() + ne * block_size, nhs
//         );
//     }
// };



// #endif // STATE_VIEW_HPP

#ifndef STATE_VIEW_HPP
#define STATE_VIEW_HPP

#include <Eigen/Dense>
#include <cassert>
#include <type_traits>

// Holds layout logic. 
// Kept separate so you can pass it around without the actual data if needed.
struct DynamicLayout
{
    const int ne;      // Number of elements
    const int nf;      // Number of flow states
    const int nxi;     // Number of flow points
    const int nhs;     // Number of heat sink states

    const int block_size; // Size of one element's flow chunk
    const int total_size; // Total size of vector
    const int input_size; 

    DynamicLayout(int ne_, int nf_, int nxi_, int nhs_)
        : ne(ne_), nf(nf_), nxi(nxi_), nhs(nhs_),
          block_size(nf_ * nxi_),
          total_size(ne_ * block_size + nhs_),
          input_size(ne_ * nxi_)
    {}
};

// Templated on VecType.
// Usage: 
//    StateView<VectorXd>       -> Mutable view
//    StateView<const VectorXd> -> Read-only view
template <typename VecType>
struct StateView : public DynamicLayout
{
    // Check that VecType is actually an Eigen Vector
    static_assert(std::is_base_of<Eigen::MatrixBase<std::decay_t<VecType>>, std::decay_t<VecType>>::value, 
                  "StateView must be initialized with an Eigen Vector type");

    VecType& state; // Can be "VectorXd&" OR "const VectorXd&"

    // Constructor automatically deduces types
    StateView(VecType& state_vec, int ne_, int nf_, int nxi_, int nhs_)
        : DynamicLayout(ne_, nf_, nxi_, nhs_), 
          state(state_vec)
    {   
        assert(state.size() == total_size && "Vector size does not match layout dimensions!");
    }

    // --- Accessors ---

    // 1. Flow State Matrix (nf x nxi) for element e
    auto fstate_mat(int e)
    {
        using Scalar = typename std::decay_t<VecType>::Scalar;
        
        // Check if the input vector is const
        constexpr bool IsConst = std::is_const<std::remove_reference_t<VecType>>::value;

        // 1. Determine the Pointer Type (double* vs const double*)
        using PtrType = std::conditional_t<IsConst, const Scalar*, Scalar*>;

        // 2. Determine the Map Target Type 
        //    If IsConst, we must map to "const Eigen::Matrix..."
        using MapTarget = std::conditional_t<IsConst,
                          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>;

        // 3. return the correct Map
        return Eigen::Map<MapTarget>(
            const_cast<PtrType>(state.data()) + e * block_size, nf, nxi
        );
    }

    // 2. Heat Sink State Vector (nhs)
    auto Ths()
    {
        using Scalar = typename std::decay_t<VecType>::Scalar;
        constexpr bool IsConst = std::is_const<std::remove_reference_t<VecType>>::value;

        using PtrType = std::conditional_t<IsConst, const Scalar*, Scalar*>;
        
        // Same logic here: Map to "const Matrix" if input is const
        using MapTarget = std::conditional_t<IsConst,
                          const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>,
                          Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>;

        return Eigen::Map<MapTarget>(
            const_cast<PtrType>(state.data()) + ne * block_size, nhs
        );
    }
};

// --- Helper for Type Deduction (Optional but handy) ---
// Allows you to write: auto view = make_state_view(vec, ...);
template <typename Derived>
auto make_state_view(Eigen::MatrixBase<Derived>& vec, int ne, int nf, int nxi, int nhs) {
    return StateView<Derived>(vec.derived(), ne, nf, nxi, nhs);
}

template <typename Derived>
auto make_state_view(const Eigen::MatrixBase<Derived>& vec, int ne, int nf, int nxi, int nhs) {
    return StateView<const Derived>(vec.derived(), ne, nf, nxi, nhs);
}


#endif // STATE_VIEW_HPP