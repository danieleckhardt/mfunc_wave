#ifndef DATA_TYPES_LEJA_APPROX_H_
#define DATA_TYPES_LEJA_APPROX_H_

#include <complex>
#include <limits>
#include <cmath>
#include <functional>
#include <variant>
#include <vector>

#include "../../types/Ellipse.h"
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

using namespace std::complex_literals;
using namespace dealii;


using BMatrixVariant = std::variant<
    std::function<int(FullMatrix<std::complex<double>>&)>,
    std::function<int(Vector<double>&, Vector<double>&)>,
    std::function<int(Vector<std::complex<double>>&, Vector<std::complex<double>>&)>
>;

namespace MavesS{
namespace TimeIntegration
{


template<int dim>
class Leja{

    public:

        static constexpr int max_degree_default = 1000;

        Leja(const int degree, Types::Ellipse<dim>& ellipse, double& time_step_size,
             const SparseMatrix<double>& A, const SparseMatrix<double>& M);


        void get_degree(int& degree_) const;

        void get_divided_differences(std::vector<std::complex<double>>& d_out) const;

        Types::Ellipse<dim>& ellipse;

        void get_count_mv(int& mv_count_out) const { mv_count_out = mv_count; }


        void matrix_run( BMatrixVariant Matrix_mult,
                          FullMatrix<std::complex<double>> &blockvector,
                          double tol = 1.,
                          int k_phi = 1 );

        void compute_divided_differences(unsigned int m, int k_phi);


        void prepare_coefficients(int k_phi);

        void update_geometry();

    private:

        double scalar_product(const Vector<std::complex<double>>& a,
                               const Vector<std::complex<double>>& b,
                               const SparseMatrix<double>& Matrix,
                               bool use_matrix = true);


        void compute_leja_geometry();


        void precompute_leja_points(unsigned int degree_max);


        void compute_taylor_phi_matrix(int idx, double tau_sub, unsigned int m,
                                        FullMatrix<std::complex<double>>& F) const;


        Vector<std::complex<double>> apply_lower_triangular(
                const FullMatrix<std::complex<double>>& F,
                const Vector<std::complex<double>>& w) const;


        static std::complex<double> phi_scalar(int k, std::complex<double> z);

        void apply_leja_real( BMatrixVariant Matrix_mult,
                               FullMatrix<std::complex<double>> &blockvector,
                               double tol,
                               int k_phi );

        void apply_leja_complex( BMatrixVariant Matrix_mult,
                                  FullMatrix<std::complex<double>> &blockvector,
                                  double tol,
                                  int k_phi );

        double calculate_residual(int k_phi);

        void refresh_geometry_if_needed();


        void ensure_divided_differences(int m_needed, int k_phi);

        static constexpr int dd_initial_block = 32;

        void set_degree(int degree_);

        // ---- Membervariablen ----

        int max_degree;         
                                 
        int degree = 0;         
        int k_phi_current = -1; 
                                
        int dd_degree = -1;     
        int mv_count = 0;

 
        double cached_radiusx = std::numeric_limits<double>::quiet_NaN();
        double cached_radiusy = std::numeric_limits<double>::quiet_NaN();
        double cached_center  = std::numeric_limits<double>::quiet_NaN();


        int points_case = -1;
        double& time_step_size; // tau

        double leja_center;        // c
        double leja_capacity;      // gamma = |ecc| / 2
        bool   use_complex_points; 


        std::vector<std::complex<double>> leja_points_ref;


        std::vector<std::complex<double>> divided_differences;


        FullMatrix<std::complex<double>> p_m, p_prev;
        FullMatrix<std::complex<double>> r_m, r_prev;
        FullMatrix<std::complex<double>> p_prime_m, p_prime_prev;
        FullMatrix<std::complex<double>> r_prime_m, r_prime_prev;


        FullMatrix<std::complex<double>> q_m, q_prime_m;

        FullMatrix<std::complex<double>> Matrix_x_vector, vector_primes;
        FullMatrix<std::complex<double>> res;

        const SparseMatrix<double>& A;
        const SparseMatrix<double>& M;
};

}/* namespace TimeIntegration*/
}/* namespace MavesS */

#endif // DATA_TYPES_LEJA_APPROX_H_