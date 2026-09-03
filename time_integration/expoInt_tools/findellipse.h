#ifndef FINDELLIPSE_H_
#define FINDELLIPSE_H_


#include <vector>
#include <complex>
#include <random>
#include <limits>
#include <cmath>

#include "chebyshev_approx.h"
#include "calc_eigen.h"
#include "arnoldikrylov.h"
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/dofs/dof_handler.h>



namespace MavesS //Multi-Waves-Scale
{

namespace TimeIntegration
{
    template <typename T>
    std::vector<T> linspace(T start, T end, T step) {
        std::vector<T> result;

        for (int i = 0; i * step < end - start; ++i) {
            result.push_back(start + i * step);
        }
        return result;
    }

    template<int dim>
    class FindEllipse {

        public:
            enum class EllipseMethod { Ritz, BoundingBox };

            enum class SpectralBounds { Eigen, Entrywise };

            void set_ellipse_method(EllipseMethod m){ ellipse_method = m; }
            void set_spectral_bounds(SpectralBounds b){ spectral_bounds = b; }
            EllipseMethod  get_ellipse_method()  const { return ellipse_method; }
            SpectralBounds get_spectral_bounds() const { return spectral_bounds; }
            // -------------------------------------------------------------

            FindEllipse(SparseMatrix<double>& mass_matrix, SparseMatrix<double>& homogeneous_matrix, SparseMatrix<double>& damping_matrix, DoFHandler<dim>&  dof_handler, bool store_info = false);
            int estimate_degree_max(std::function<std::complex<double>(std::complex<double>)> func,  double tol = 1e-2, int init_degee = 10);
            int estimate_degree_max(int phi_k,  double tol = 1e-2, int init_degee = 10);
            std::vector<int> prototype(double time_step_size );
            std::vector<int> prototype(double time_step_size, const FullMatrix<double>& A);

            std::vector<int> enclosure_ellipse(double time_step_size,
                                        double safety    = 4.0,
                                        double eps_shape = 0.0,
                                        double max_real  = std::numeric_limits<double>::infinity());
            void spectral_bounds_rectangle(double time_step_size,
                                           double safety, double eps_shape,
                                           double& alpha, double& nu, double& beta);
            //  mu and delta are independent of tau; invalidate after remeshing.
            void invalidate_spectral_cache(){
                mu_cached    = std::numeric_limits<double>::quiet_NaN();
                delta_cached = std::numeric_limits<double>::quiet_NaN();
            }

            int check_right_halfplane(double max_extent = 2.0);
            void chebyshev_arnoldi(int phi_k, Types::BlockMatrix<dim>& block_matrix,  std::vector<std::complex<double>>& detected_points, double tol_fix, double time_step_size);
            void get_ellipse(Types::Ellipse<dim>& ell_out){ell_out = this->ellipse;}
            void set_ellipse(Types::Ellipse<dim>& ell_in){this->ellipse = ell_in;}
            std::vector<std::vector<std::variant<
                int    // degree
                ,double // radius
                ,std::vector<double> //parameters
                ,std::vector<std::vector<std::variant<std::complex<double>,double,int>>> // points
            >>> info_steps;
            std::vector<std::complex<double>> eigenvalues_pub;
        private:
            std::vector<int> make_ellipse(std::vector<std::complex<double>>& pts){
                return (ellipse_method == EllipseMethod::BoundingBox)
                     ? ellipse.create_opt_ellipse_bounding_box(pts)
                     : ellipse.create_opt_ellipse(pts);
            }
            void   lumped_mass(std::vector<double>& m_lump) const;
            double gershgorin_lumped_bound(const SparseMatrix<double>& S,
                                           const std::vector<double>&  m_lump) const;

            double stewart_pencil_bound(const SparseMatrix<double>& S) const;
            double max_eig_pencil(const SparseMatrix<double>& S,
                                  double tol = 1e-4, int max_iter = 300);
            //  --------------------------------------------------------------
            void create_bd_point();
            bool almost_equal(const std::complex<double>& a, 
                  const std::complex<double>& b, 
                  double eps ); 
            bool estimate_degree(int& degree, double tol);
            std::complex<double> scalar_product(const Vector<std::complex<double>>& a,
                                                const Vector<std::complex<double>>& b,
                                                const SparseMatrix<double>& Matrix);
            void get_start_vectors(Vector<std::complex<double>>& start_vec_u, Vector<std::complex<double>>& start_vec_v, bool orthogonalize);
            void add_random_vector_and_orthogonalize( Vector<std::complex<double>>& vec_u, Vector<std::complex<double>>& vec_v);
            void extract_real_part( Vector<std::complex<double>>& vec_u, Vector<std::complex<double>>& vec_v);
            bool check_approximate_eigenpair(
                            Types::BlockMatrix<dim>& block_matrix,
                            const std::complex<double>& lambda,
                            const Vector<std::complex<double>>& eigvec_u,
                            const Vector<std::complex<double>>& eigvec_v,
                            double tol); 
            bool estimate_ellipse_chebyshev_arnoldi(
                    std::function<int(Vector<std::complex<double>>&, Vector<std::complex<double>>&)> cheb_n,
                    Types::BlockMatrix<dim>& block_matrix,
                    std::vector<std::complex<double>>& points,
                    double tol_fix, double& tol_variable,
                    int step);
            void build_reduced_matrix(Types::BlockMatrix<dim>& block_matrix,const int iterations_needed);
            void reconstruct_and_sort_eigenpairs(const int iterations_needed);
            EllipseMethod  ellipse_method  = EllipseMethod::Ritz;       // DEFAULT
            SpectralBounds spectral_bounds = SpectralBounds::Eigen;
            double mu_cached    = std::numeric_limits<double>::quiet_NaN();
            double delta_cached = std::numeric_limits<double>::quiet_NaN();
            int size;
            int degree;
            int degree_max;
            int max_iter_arnoldi;
            double radius_max;
            int max_steps;
            double time_step_size;
            bool degree_max_reached = false;
            bool store_info;
            double relative_residual;
            double global_res = 1.;
            std::complex<double>  bd_point;

            Vector<std::complex<double>> lambda_v;
            Vector<std::complex<double>> lambda_u;
            Vector<std::complex<double>> start_vec_krylov_u;
            Vector<std::complex<double>> start_vec_krylov_v;
            std::vector<std::complex<double>> sorted_eigenvalues;
            Vector<std::complex<double>> res_u;
            Vector<std::complex<double>> res_v;
            Vector<std::complex<double>>  tmp_vec;
            std::vector<Vector<std::complex<double>>> krylov_vectors_u; 
            std::vector<Vector<std::complex<double>>> krylov_vectors_v;
            std::vector<Vector<std::complex<double>>> approx_eigen_vectors_u;
            std::vector<Vector<std::complex<double>>> approx_eigen_vectors_v;
            std::vector<Vector<std::complex<double>>> sorted_eigenvectors_u;
            std::vector<Vector<std::complex<double>>> sorted_eigenvectors_v;
            std::vector<Vector<std::complex<double>>> block_vec_vec;
            std::vector<Vector<std::complex<double>>> tmp_vec_u;
            std::vector<Vector<std::complex<double>>> tmp_vec_v;
            std::vector<std::complex<double>> origin_points;
            std::vector<std::vector<Vector<std::complex<double>>>> detected_vectors;
            std::vector<std::variant<
                int    // degree
                ,double // radius
                ,std::vector<double> //parameters
                ,std::vector<std::vector<std::variant<std::complex<double>,double,int>>> // points
            >> info_single_step;
            std::vector<double> ellipse_parameters;
            std::vector<std::vector<std::variant<std::complex<double>,double,int>>> info_step_points;
            std::vector<std::variant<std::complex<double>,double,int>> info_step_single_point;


            std::vector<std::complex<double>> eigenvalues;
            FullMatrix<std::complex<double>> eigenvectors;
            FullMatrix<std::complex<double>> Hm_block_matrix;
            FullMatrix<std::complex<double>> test_block_matrix;

            FullMatrix<std::complex<double>> block_vec;

            SparseMatrix<double>& mass_matrix;
            Types::InverseMatrix<dim,double> M_inv;
            SparseMatrix<double> identity_matrix;
            SparseMatrix<double> test_matrix;
            SparseMatrix<double>& homogeneous_matrix; // this matrix will be created by calculate the micro solution
            SparseMatrix<double>& damping_matrix;  
            DoFHandler<dim>&  dof_handler;
            Types::Ellipse<dim> ellipse;

            Calc_eigen<dim> calc_eigen;
            Chebyshev<dim> cheb_approx;
            ArnoldiKrylov<dim, std::complex<double>> arnoldi_krylov;
            std::vector<int> ellipse_config;
            std::vector<int> ellipse_config_new;
        
    };

}
}
#endif /* FINDELLIPSE_H_ */