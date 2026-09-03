#include "findellipse.h"

namespace MavesS {
namespace TimeIntegration
{   

template<int dim>
FindEllipse<dim>::FindEllipse(SparseMatrix<double>& mass_matrix, SparseMatrix<double>& homogeneous_matrix, SparseMatrix<double>& damping_matrix,  DoFHandler<dim>&  dof_handler, bool store_info)
    :
    info_steps()
    ,max_steps(30)
    ,time_step_size(1.0) 
    ,store_info(store_info)
    ,info_single_step()
    ,ellipse_parameters()
    ,info_step_points()
    ,info_step_single_point()
    ,mass_matrix(mass_matrix)
    ,M_inv(mass_matrix,dof_handler)
    ,homogeneous_matrix(homogeneous_matrix)
    ,damping_matrix(damping_matrix)
    ,dof_handler(dof_handler)
    ,calc_eigen(eigenvalues, eigenvectors)
    ,cheb_approx(10, ellipse, time_step_size,homogeneous_matrix,mass_matrix)
    ,arnoldi_krylov(homogeneous_matrix, mass_matrix, krylov_vectors_u, krylov_vectors_v,time_step_size)
    {}

template<int dim> 
bool FindEllipse<dim>::almost_equal(const std::complex<double>& a, 
                  const std::complex<double>& b, 
                  double eps ) {
    return std::sqrt(std::pow(a.real() - b.real(), 2) + std::pow(a.imag() - b.imag(),2)) < eps;
}

template<int dim> 
void FindEllipse<dim>::create_bd_point() {
    double r_x, r_y, c;
    std::complex<double> foci;
    ellipse.get_radx(r_x);ellipse.get_rady(r_y); 
    ellipse.get_center(c);ellipse.get_ecc(foci);
    (void)foci;  
    if(r_x >= r_y){
        bd_point = c + r_x;
    }
    else{
        bd_point = c + r_y*1i;
    }
}

template<int dim>
void FindEllipse<dim>::get_start_vectors(
    Vector<std::complex<double>>& start_vec_u,
    Vector<std::complex<double>>& start_vec_v,
    bool orthogonalize)
{
    // --- Random number generator (kept local, but could be static for efficiency)
    // std::random_device rd;
    const unsigned int seed = 61; // Fixed seed for reproducibility
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    const unsigned int n = dof_handler.n_dofs();

    // --- Resize and initialize random vectors and store
    Vector<std::complex<double>> tmpu(n), tmpv(n);
    // tmpu = start_vec_krylov_u;
    // tmpv = start_vec_krylov_v;
    start_vec_u.reinit(n);
    start_vec_v.reinit(n);

    for (unsigned int i = 0; i < n; ++i)
    {
        start_vec_u[i] = std::complex<double>(dis(gen), dis(gen));
        start_vec_v[i] = std::complex<double>(dis(gen), dis(gen));
    }

    if (!orthogonalize)
        return;
    Vector<std::complex<double>> Au(n), Mv(n);

    for(unsigned int j= 0; j < detected_vectors.size(); j++){
        tmpu = detected_vectors[j][0];
        tmpv = detected_vectors[j][1];

        // --- Apply operators

        homogeneous_matrix.vmult(Au, tmpu);
        mass_matrix.vmult(Mv, tmpv);

        // --- Compute inner products (Hermitian)
        std::complex<double> num = 0.0;
        std::complex<double> den = 0.0;

        for (unsigned int i = 0; i < n; ++i)
        {
            num += std::conj(start_vec_u[i]) * Au[i];
            num += std::conj(start_vec_v[i]) * Mv[i];

            den += std::conj(tmpu[i]) * Au[i];
            den += std::conj(tmpv[i]) * Mv[i];
        }

        const double norm = std::abs(den);
        if (norm < 1e-14)
            continue;

        std::complex<double> alpha = std::conj(num) / den;

        for (unsigned int i = 0; i < n; ++i)
        {
            start_vec_u[i] -= alpha * tmpu[i];
            start_vec_v[i] -= alpha * tmpv[i];
        }

    }

}

template <int dim>
void FindEllipse<dim>::extract_real_part(
    Vector<std::complex<double>>& vec_u,
    Vector<std::complex<double>>& vec_v)
{
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i) {
        vec_u[i] = std::real(vec_u[i]);
        vec_v[i] = std::real(vec_v[i]);
    }
}

template<int dim>
bool FindEllipse<dim>::estimate_degree(int& degree, double tol){
    // estimate degree
    std::cout << "Estimated degree..." << std::endl;
    if (radius_max > (1.0+ tol*100.)){
        double damping = 1./ radius_max;
        degree = std::max(static_cast<int>(std::round(0.5*std::abs(std::log(tol)/std::log(damping)))), degree ) +10 ;
        std::cout << "  ...using that there is still a gap." << std::endl;
    }
    else{
        degree = static_cast<int>(std::round( degree* (1 + 0.1*std::abs(std::log(radius_max/(1.+tol) ) + 1 )))) ;
        std::cout << "  ...near to convergence." << std::endl;

    }
    if (degree < degree_max)
        return false;
    else{
        degree = degree_max;
        return true;
    }
}

template <int dim>
bool FindEllipse<dim>::check_approximate_eigenpair(
    Types::BlockMatrix<dim>& block_matrix,
    const std::complex<double>& lambda,
    const Vector<std::complex<double>>& eigvec_u,
    const Vector<std::complex<double>>& eigvec_v,
    double tol) 
{
    // Step 1: Apply A * v
    Vector<std::complex<double>> Au(eigvec_u.size());
    Vector<std::complex<double>> Av(eigvec_v.size());
    block_matrix.vmult_complex_split_weak_op(Au, Av, eigvec_u, eigvec_v);

    // Step 2: Form λ * v
    Vector<std::complex<double>> lambda_u = eigvec_u;
    Vector<std::complex<double>> lambda_v = eigvec_v;
    homogeneous_matrix.vmult(lambda_u,eigvec_u);
    mass_matrix.vmult(lambda_v,eigvec_v);
    lambda_u *= lambda;
    lambda_v *= lambda;

    // Step 3: Compute residual r = Av - λv
    Vector<std::complex<double>> res_u = Au;
    Vector<std::complex<double>> res_v = Av;
    res_u -= lambda_u;
    res_v -= lambda_v;

    // Step 4: Compute relative residual
    double res = std::sqrt(
        std::pow(res_u.l2_norm(), 2) + std::pow(res_v.l2_norm(), 2));
    double norm_Auv = std::sqrt(
        std::pow(Au.l2_norm(), 2) + std::pow(lambda_u.l2_norm(), 2) + std::pow(Av.l2_norm(), 2) + std::pow(lambda_v.l2_norm(), 2));
    // std::cout << "norm_Auv" << norm_Auv << std::endl;
    relative_residual = res / (norm_Auv + 1e-14);
    if (relative_residual  < global_res && ellipse.radius_p_alterative(lambda) >= ellipse.radius_p_alterative(sorted_eigenvalues[0])){
        // take as new startvector, eig_vec with rad(eig_val) is largest and res is smallest 
        start_vec_krylov_u = eigvec_u;
        start_vec_krylov_v = eigvec_v;
        global_res = relative_residual;
    }

    if (relative_residual < tol) {
        std::cout << "  Accepted λ=" << lambda
                    << " with residual " << relative_residual << " and Radius: " << ellipse.radius_p_alterative(lambda)/ellipse.radius_p_alterative(bd_point) << ". ";
        return true;
    } else {
        std::cout << "  Rejected λ=" << lambda
                    << " with residual " << relative_residual << " and Radius: " << ellipse.radius_p_alterative(lambda)/ellipse.radius_p_alterative(bd_point)<< std::endl;
        
        return false;
    }
}

template <int dim>
void FindEllipse<dim>::build_reduced_matrix(Types::BlockMatrix<dim>& block_matrix, const int iterations_needed){
    const unsigned int n = homogeneous_matrix.n();

    // Step 1: Compute A * V
    std::vector<Vector<std::complex<double>>> tmp_u(iterations_needed, Vector<std::complex<double>>(n));
    std::vector<Vector<std::complex<double>>> tmp_v(iterations_needed, Vector<std::complex<double>>(n));

    for (int i = 0; i < iterations_needed; ++i) {
        block_matrix.vmult_complex_split_weak_op(tmp_u[i], tmp_v[i], krylov_vectors_u[i], krylov_vectors_v[i]);
    }
    // Step 2: Project into Krylov basis -> fill Hm_block_matrix
    // Note: V_m is orthogonal w.r.t. [[A 0][0 M]]
    Hm_block_matrix.reinit(iterations_needed, iterations_needed);
    for (int i = 0; i < iterations_needed; ++i) {
        for (int j = 0; j < iterations_needed; ++j) {
            std::complex<double> val_ij = krylov_vectors_u[i] * tmp_u[j]
                                        + krylov_vectors_v[i] * tmp_v[j];
            Hm_block_matrix(i, j) = val_ij;
        }
    }
}

template <int dim>
void FindEllipse<dim>::reconstruct_and_sort_eigenpairs(const int iterations_needed){
    const unsigned int n = krylov_vectors_u[0].size();

    // Step 1: Reconstruct approximate eigenvectors in full space
    std::vector<Vector<std::complex<double>>> approx_u(iterations_needed, Vector<std::complex<double>>(n));
    std::vector<Vector<std::complex<double>>> approx_v(iterations_needed, Vector<std::complex<double>>(n));

    for (int i = 0; i < iterations_needed; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            std::complex<double> sum_u = 0.0;
            std::complex<double> sum_v = 0.0;
            for (int k = 0; k < iterations_needed; ++k) {
                sum_u += krylov_vectors_u[k][j] * eigenvectors[k][i];
                sum_v += krylov_vectors_v[k][j] * eigenvectors[k][i];
            }
            approx_u[i][j] = sum_u;
            approx_v[i][j] = sum_v;
        }
    }

    // Step 2: Sort indices by ellipse radius
    std::vector<int> indices(eigenvalues.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::stable_sort(indices.begin(), indices.end(), [this](int i1, int i2) {
        return ellipse.radius_p_alterative(eigenvalues[i1]) > ellipse.radius_p_alterative(eigenvalues[i2]);
    });

    // Step 3: Reorder eigenpairs
    sorted_eigenvalues.clear();
    sorted_eigenvectors_u.clear();
    sorted_eigenvectors_v.clear();

    for (int idx : indices) {
        sorted_eigenvalues.push_back(eigenvalues[idx]);
        sorted_eigenvectors_u.push_back(approx_u[idx]);
        sorted_eigenvectors_v.push_back(approx_v[idx]);
    }
    start_vec_krylov_u = sorted_eigenvectors_u[0];
    start_vec_krylov_v = sorted_eigenvectors_v[0];

}

template<int dim>
bool FindEllipse<dim>::estimate_ellipse_chebyshev_arnoldi( std::function<int(Vector<std::complex<double>>&, Vector<std::complex<double>>&)> cheb_n,
                                                            Types::BlockMatrix<dim>& block_matrix,
                                                            std::vector<std::complex<double>>& detected_points,
                                                            double tol_fix,double & tol_variable,
                                                            int steps){
    int found_count = 0;
    bool found_new_points = false;
    // const int n = homogeneous_matrix.n();
    double r_x,r_y,ce;
    std::cout << " "  << std::endl;
    std::cout << "steps: " << steps << std::endl;
    //
    if (store_info){
        info_single_step.clear();
        ellipse_parameters.clear();
        info_step_points.clear();
    }
    //
    
    // Run Arnoldi algorithm
    arnoldi_krylov.run_poly(cheb_n, nullptr, tol_variable);
    int iterations_needed = arnoldi_krylov.get_iterations();

    // Estimate spectral radius
    std::complex<double> maxeig = arnoldi_krylov.get_max_eigenvalue();
    radius_max = std::abs(std::pow(std::abs(maxeig), 1.0 / (degree)));
    //
    if (store_info){
        info_single_step.push_back(degree);
        info_single_step.push_back(radius_max);
    }
    //

    if (radius_max <= 1.0 + tol_fix){
        std::cout << "The maximum radius is " << radius_max << " <= "  << 1.0 + tol_fix << ". (degree =  " << degree << ")."  << std::endl;
        return false;
    } 
    else
        std::cout << "The maximum radius is " << radius_max << " > "  << 1.0 + tol_fix << ". Continue with tol = " << 1.0 + tol_fix  << " and degree =  " << degree << " . " << std::endl;
    
    // Compute (V^T , A @ V)_{H^1 x L^2} : 
    build_reduced_matrix(block_matrix,iterations_needed);
   
    // Compute eigenvalues and eigenvectors of the reduced H matrix
    calc_eigen.calc_all_eig(Hm_block_matrix, iterations_needed);

    // Reconstruct approximate eigenvectors
    reconstruct_and_sort_eigenpairs(iterations_needed);
    
    // Check approximate eigenpairs
    std::cout << "Checking approximate eigenpairs..." << std::endl;
    for (size_t count = 0; count < sorted_eigenvalues.size(); ++count) {
        const std::complex<double>& lambda = sorted_eigenvalues[count];
        double lambda_re = lambda.real();
        double lambda_imag = lambda.imag();
        
        if ( ellipse.radius_p_alterative(lambda) <= 1.0 || lambda.imag() < 0. ) {
            continue; // Skip eigenvalues with negative imaginary part or inside ellipse
        }
        //
        if (store_info){
            info_step_single_point.clear();
            info_step_single_point.push_back(lambda_re);
            info_step_single_point.push_back(lambda_imag);
        }
        //
        if (check_approximate_eigenpair(block_matrix, lambda,sorted_eigenvectors_u[count],sorted_eigenvectors_v[count],tol_fix)){
            auto it = std::find_if(detected_points.begin(), detected_points.end(), [&](const std::complex<double>& c){ return almost_equal(c, lambda, tol_fix);});
            if (it == detected_points.end()) {
                detected_points.push_back(lambda);
                detected_vectors.push_back({sorted_eigenvectors_u[count],
                                             sorted_eigenvectors_v[count]});
                std::cout << "New eigenpair!" << " (" << found_count  <<  ". in this loop)." <<  std::endl;
                found_count++;
                found_new_points = true;
                if(store_info){
                    info_step_single_point.push_back(relative_residual);
                    info_step_single_point.push_back(1);
                    info_step_points.push_back(info_step_single_point);
                }
            }
            else{
                std::cout << "Eigenpair already detected." << std::endl;
                if(store_info){
                    info_step_single_point.push_back(relative_residual);
                    info_step_single_point.push_back(-1);
                    info_step_points.push_back(info_step_single_point);
                }
            }
        }
        else{
            if (store_info){
                info_step_single_point.push_back(relative_residual);
                info_step_single_point.push_back(0);
                info_step_points.push_back(info_step_single_point);
            }
        }
    }
    global_res = 1.;
    // If no new ellipse, use smaller tol 
    ellipse_config = make_ellipse(detected_points);
    ellipse.get_radx(r_x);
    ellipse.get_rady(r_y);
    ellipse.get_center(ce);
    if (store_info){
        ellipse_parameters.push_back(r_x);
        ellipse_parameters.push_back(r_y);
        ellipse_parameters.push_back(ce);
        info_single_step.push_back(ellipse_parameters);
        info_single_step.push_back(info_step_points);
    }

    if (found_new_points){
        tol_variable = tol_fix;
    }
    else{
        if (!degree_max_reached){
            // degree += 10;
        }
        else{
            tol_variable *= 0.1;
            degree_max_reached = false;
            // degree_max += 5; 
            std::cout << " Maximal degree has been reached. Choose smaller tol=" << tol_variable << " for Arnoldi process. " << std::endl;
        }
        ellipse.get_rady(r_y);
        ellipse.set_rady_with_fix_center_and_ecc(r_y*0.95); // better convergence, keep foci and center fixed to converge against wanted eigenvalues
    }
    create_bd_point();
    return true;
}

////////////////////////////////
/*
    Public Methods
*/
///////////////////////////////

template<int dim>
int FindEllipse<dim>::estimate_degree_max(std::function<std::complex<double>(std::complex<double>)> func,  double tol, int init_degree){
    // estimate degree
    int estimated_degree = 0;
    cheb_approx.set_degree(init_degree);
    cheb_approx.estimate_degree_ellipse_by_error(func,tol);
    cheb_approx.get_degree(estimated_degree);
    std::cout << "  Set max degree : " << estimated_degree << std::endl;
    return estimated_degree  ; 
}

template<int dim>
int FindEllipse<dim>::estimate_degree_max(int phi_k, double tol, int init_degree){
    // estimate degree
    // int estimated_degree = 0;
    // cheb_approx.set_degree(init_degree);
    // cheb_approx.estimate_degree_ellipse_by_estimate(phi_k,tol);
    // cheb_approx.get_degree(estimated_degree);
    int degree = 1000;
    cheb_approx.set_degree(degree);
    std::cout << "  Set max degree : " << degree << std::endl;
    return degree  ; 
}

template< int dim> 
std::vector<int> FindEllipse<dim>::prototype(double time_step_size){
    // calculate eigenvalues
    size = dof_handler.n_dofs();
    FullMatrix<double> A_full_matrix;
    const IdentityMatrix id_matrix(size);
    FullMatrix<double> identity_m = id_matrix; 

    A_full_matrix.reinit(2*size,2*size);
    std::vector<int> first_index_set  = linspace(0,size,1);
    std::vector<int> second_index_set = linspace(size,2*size,1);

    identity_m.scatter_matrix_to(first_index_set,second_index_set,A_full_matrix);

    FullMatrix<double> homogeneous_matrix_full(size,size);
    homogeneous_matrix_full.copy_from(homogeneous_matrix);
    homogeneous_matrix_full *= -1.;
    M_inv.mmult(homogeneous_matrix_full,homogeneous_matrix_full,1e-10);
    homogeneous_matrix_full.scatter_matrix_to(second_index_set,first_index_set,A_full_matrix);

    FullMatrix<double> damping_matrix_full(size,size);
    damping_matrix_full.copy_from(damping_matrix);
    damping_matrix_full *= -1.;
    M_inv.mmult(damping_matrix_full,damping_matrix_full,1e-10);
    damping_matrix_full.scatter_matrix_to(second_index_set,second_index_set,A_full_matrix);
    A_full_matrix *= time_step_size; // scale matrix with time step size
    calc_eigen.calc_all_eig(A_full_matrix, 2*size);

    return make_ellipse(eigenvalues);
}

template< int dim> 
std::vector<int> FindEllipse<dim>::prototype(double time_step_size, const FullMatrix<double>& A){

    const int matrix_size = A.n();
    calc_eigen.calc_all_eig(A, matrix_size);
    for (unsigned long i = 0; i < eigenvalues.size(); ++i) {
        eigenvalues[i] *= time_step_size;
    }
    eigenvalues_pub = eigenvalues;

    return make_ellipse(eigenvalues);

}

////////////////////////////////
/*
    Rectangular spectral enclosure  ->  minimal-capacity ellipse (matrix-free)
*/
///////////////////////////////

template<int dim>
void FindEllipse<dim>::lumped_mass(std::vector<double>& m_lump) const
{
    const unsigned int n = mass_matrix.m();
    m_lump.assign(n, 0.0);
    for (unsigned int i = 0; i < n; ++i)
    {
        double s = 0.;
        for (auto it = mass_matrix.begin(i); it != mass_matrix.end(i); ++it)
            s += it->value();                    // row-sum lumping
        if (s <= 0.)                             // fallback for pathological rows
            s = mass_matrix.el(i, i);
        if (s <= 0.)
            throw std::runtime_error("FindEllipse::lumped_mass: non-positive lumped mass.");
        m_lump[i] = s;
    }
}


template<int dim>
double FindEllipse<dim>::gershgorin_lumped_bound(const SparseMatrix<double>& S,
                                                 const std::vector<double>&  m_lump) const
{
    double bound = 0.;
    for (unsigned int i = 0; i < S.m(); ++i)
    {
        double row = 0.;
        for (auto it = S.begin(i); it != S.end(i); ++it)
            row += std::abs(it->value()) / std::sqrt(m_lump[i] * m_lump[it->column()]);
        bound = std::max(bound, row);
    }
    return bound;
}


template<int dim>
double FindEllipse<dim>::stewart_pencil_bound(const SparseMatrix<double>& S) const
{
    double bound = 0.;
    for (unsigned int i = 0; i < S.m(); ++i)
    {
        double num = 0.;
        for (auto it = S.begin(i); it != S.end(i); ++it)
            num += std::abs(it->value());

        double den = 0.;
        for (auto it = mass_matrix.begin(i); it != mass_matrix.end(i); ++it)
            den += (it->column() == i) ? it->value() : -std::abs(it->value());

        if (den <= 0.)
            return std::numeric_limits<double>::infinity();
        bound = std::max(bound, num / den);
    }
    return bound;
}


template<int dim>
double FindEllipse<dim>::max_eig_pencil(const SparseMatrix<double>& S,
                                        double tol, int max_iter)
{
    const unsigned int n = S.m();
    if (S.frobenius_norm() < 1e-300)
        return 0.;                                  // e.g. D == 0

    Vector<double> x(n), y(n), z(n), Mx(n), rhs(n);
    std::mt19937 gen(12345);                        // deterministic on purpose
    std::uniform_real_distribution<> dis(-1., 1.);
    for (unsigned int i = 0; i < n; ++i) x[i] = dis(gen);


    auto solve_with_M = [&](Vector<double>& dst, const Vector<double>& b) -> bool
    {
        rhs = b;
        const double nb = rhs.l2_norm();
        if (nb < 1e-300) { dst = 0.; return true; }
        return M_inv.vmult(dst, rhs, tol * nb) != 0;      // 0 == solver failure
    };

    double theta = 0., theta_old = -1.;
    for (int it = 0; it < max_iter; ++it)
    {
        S.vmult(y, x);
        mass_matrix.vmult(Mx, x);
        const double denom = x * Mx;
        if (denom <= 0.) break;
        theta = (x * y) / denom;                    // Rayleigh quotient of (S,M)

        if (!solve_with_M(z, y))
        {
            std::cout << "WARNING max_eig_pencil: mass solve failed, returning "
                         "the current Rayleigh quotient (lower bound!)." << std::endl;
            return theta;
        }
        const double nz = z.l2_norm();
        if (nz < 1e-300) break;
        z /= nz;
        x = z;

        if (it > 2 && std::abs(theta - theta_old) <= tol * std::abs(theta)) break;
        theta_old = theta;
    }

    // a-posteriori inclusion radius  ||r||_{M^{-1}} / ||x||_M ,  r = S x - theta M x
    // S.vmult(y, x);
    // mass_matrix.vmult(Mx, x);
    // y.add(-theta, Mx);                              // y = r, stays intact
    // if (!solve_with_M(z, y))
    //     return theta;
    // const double res = std::sqrt(std::abs(y * z));  // sqrt( r^T M^{-1} r )
    // const double xM  = std::sqrt(std::abs(x * Mx));

    return theta;  //+ res / std::max(xM, 1e-300);
}

template<int dim>
void FindEllipse<dim>::spectral_bounds_rectangle(double time_step_size,
                                                 double safety, double eps_shape,
                                                 double& alpha, double& nu, double& beta)
{

    if (std::isnan(mu_cached) || std::isnan(delta_cached))
    {
        if (spectral_bounds == SpectralBounds::Eigen)
        {
            mu_cached    = max_eig_pencil(homogeneous_matrix);
            delta_cached = max_eig_pencil(damping_matrix);
            std::cout << "  spectral bounds: power iteration on the pencils (K,M),(D,M)"
                      << std::endl;
        }
        else
        {
            const double mu_st = stewart_pencil_bound(homogeneous_matrix);
            const double de_st = stewart_pencil_bound(damping_matrix);
            if (std::isfinite(mu_st) && std::isfinite(de_st))
            {
                mu_cached = mu_st;  delta_cached = de_st;
                std::cout << "  spectral bounds: Stewart (1975) pencil bound, rigorous"
                          << std::endl;
            }
            else
            {
                std::vector<double> m_lump;  lumped_mass(m_lump);
                mu_cached    = safety * gershgorin_lumped_bound(homogeneous_matrix, m_lump);
                delta_cached = safety * gershgorin_lumped_bound(damping_matrix,     m_lump);
                std::cout << "  spectral bounds: lumped-mass Gershgorin, safety = "
                          << safety << "  (M not strictly diagonally dominant, "
                          << "Stewart bound is infinite)" << std::endl;
            }
        }
        std::cout << "  mu = lambda_max(M^-1 K) = " << mu_cached
                  << " , delta = lambda_max(M^-1 D) = " << delta_cached << std::endl;
    }

    beta  =  time_step_size * std::sqrt(std::max(mu_cached, 0.));
    alpha = -time_step_size * std::max(delta_cached, 0.);
    nu    =  0.0;                                   // exact: W_G(S_H) = [-delta, 0]


    if (alpha > -eps_shape * beta)
        alpha = -eps_shape * beta;
}

template<int dim>
int FindEllipse<dim>::check_right_halfplane(double max_extent)
{
    const double extent = ellipse.right_halfplane_extent();
    if (extent <= 0.)
        return 1;
    const int s = std::max(1, (int)std::ceil(extent / std::max(max_extent, 1e-12)));
    if (extent > max_extent)
    {
        std::cout << "WARNING: ellipse extends to Re = " << extent
                  << " in the right half-plane. phi_k grows like exp(" << extent
                  << ") there, so roughly " << (int)std::ceil(extent/std::log(10.))
                  << " digits are lost to cancellation." << std::endl;
        std::cout << "         Suggested number of substeps: s = " << s
                  << " (the extent scales exactly like 1/s)." << std::endl;
    }
    return s;
}

template<int dim>
std::vector<int> FindEllipse<dim>::enclosure_ellipse(double time_step_size,
                                              double safety, double eps_shape,
                                              double max_real)
{
    double alpha, nu, beta;
    spectral_bounds_rectangle(time_step_size, safety, eps_shape, alpha, nu, beta);

    std::cout << "Enclosure rectangle: alpha = " << alpha
              << " , nu = " << nu << " , beta = " << beta << std::endl;

    const int n_def = ellipse.create_min_capacity_ellipse(alpha, nu, beta, max_real);
    create_bd_point();

    double r_x, r_y, ce, cap_e, cap_c;  std::complex<double> foci;
    ellipse.get_radx(r_x);   ellipse.get_rady(r_y);
    ellipse.get_center(ce);  ellipse.get_ecc(foci);
    ellipse.get_capacity_ellipse(cap_e);
    ellipse.get_capacity_confocal(cap_c);
    std::cout << "Enclosure ellipse:   radiusx = " << r_x << " radiusy = " << r_y
              << " center = " << ce << " foci = " << foci
              << " capacity_ellipse = " << cap_e
              << " capacity_confocal(gamma) = " << cap_c << std::endl;

    check_right_halfplane();

    this->time_step_size = time_step_size;
    ellipse_config = {n_def, -1, -1};
    return ellipse_config;
}

template<int dim>
void FindEllipse<dim>::chebyshev_arnoldi(int phi_k, Types::BlockMatrix<dim>& block_matrix,  std::vector<std::complex<double>>& detected_points, double tol_fix, double time_step_size) {
    double radius_tempx,radius_tempy,center;
    std::complex<double> foci;
    unsigned long points_size = detected_points.size();
    max_iter_arnoldi = std::min(200,2*block_matrix.get_size());
    arnoldi_krylov.set_time_step_size(time_step_size);
    arnoldi_krylov.set_max_iterations(max_iter_arnoldi);

    // Initialize Matrix-Vector operation
    BMatrixVariant Matrix_mult = [&block_matrix](FullMatrix<std::complex<double>>& block_vec) {
        return block_matrix.vmultBlock(block_vec, 1e-10);
    };

    get_start_vectors(krylov_vectors_u[0], krylov_vectors_v[0],false);
    double tol_variable = 1e-2;

    // Generate initial ellipse from eigenvalues
    ellipse_config = make_ellipse(detected_points);
    ellipse.get_ecc(foci);
    degree_max = estimate_degree_max(phi_k,tol_fix, 10) ;
    degree = std::min( 2*static_cast<int>(std::sqrt(std::abs(foci))) + 10 , degree_max); // initial degree
    cheb_approx.set_chebyshev_tn_basis(degree);
    create_bd_point();

    // matrix-vector operation using Chebyshev polynomial
    block_vec.reinit(dof_handler.n_dofs(),2);
    std::function<int(Vector<std::complex<double>>&, Vector<std::complex<double>>&)> cheb_n =
        [this, Matrix_mult](Vector<std::complex<double>>& b_1, Vector<std::complex<double>>& b_2) {
            for(unsigned int j = 0; j < b_1.size(); ++j){
                    this->block_vec(j,0) = b_1(j);
                    this->block_vec(j,1) = b_2(j);
            }
            cheb_approx.matrix_run(Matrix_mult, nullptr, this->block_vec, this->bd_point);
            for(unsigned int j = 0; j < b_1.size(); ++j){
                    b_1(j) = this->block_vec(j,0);
                    b_2(j) = this->block_vec(j,1);
            }
            return 0; // for compiling, has no sense at all
        };

    // Iterative refinement with step limit
    double r_x,r_y,ce;
    if (store_info){
        info_steps.clear();
        info_single_step.clear();
        ellipse_parameters.clear();
        ellipse.get_radx(r_x);
        ellipse.get_rady(r_y);
        ellipse.get_center(ce);
        ellipse_parameters.push_back(r_x);
        ellipse_parameters.push_back(r_y);
        ellipse_parameters.push_back(ce);
        //
        info_single_step.push_back(ellipse_parameters);
        info_steps.push_back(info_single_step);
    }
    for (int steps = 0; steps < max_steps; ++steps) {
        bool found_new_ellipse = estimate_ellipse_chebyshev_arnoldi(cheb_n, block_matrix, detected_points, tol_fix, tol_variable, steps );
        if (store_info){
            info_steps.push_back(info_single_step);
        }
        if (!found_new_ellipse || steps == max_steps-1) {
            make_ellipse(detected_points);
            ellipse.get_radx(radius_tempx);ellipse.get_rady(radius_tempy);ellipse.get_center(center);ellipse.get_ecc(foci);
            if (steps == max_steps-1) {
                std::cout << "Warning: Maximum steps reached without convergence." << std::endl;
            } else {
                std::cout << "Optimal ellipse has been found! Converged in " << steps << " steps. " << std::endl;
            }
            std::cout << "Final ellipse:  radiusx = "<< radius_tempx << " radiusy = "<< radius_tempy << " center = "<< center << " foci = "<< foci << std::endl;
            std::cout << " " << std::endl;
            std::cout << "Completed calculating the optimal ellipse." << std::endl;
            std::cout << " " << std::endl;
            break;  // Exit loop if no new ellipse is found
        }
        if( detected_points.size() > points_size ){
        // start again with random vector
            krylov_vectors_u[0] = start_vec_krylov_u;
            krylov_vectors_v[0] = start_vec_krylov_v;
            get_start_vectors(krylov_vectors_u[0], krylov_vectors_v[0],true);
            points_size = detected_points.size();
        }
        else{
        // use approximate eigenvector for restart
            krylov_vectors_u[0] = start_vec_krylov_u;
            krylov_vectors_v[0] = start_vec_krylov_v;
            extract_real_part(krylov_vectors_u[0],krylov_vectors_v[0]);
        }
        //
        degree_max_reached = estimate_degree(degree,tol_fix);
        cheb_approx.set_chebyshev_tn_basis(degree);
        ellipse.get_radx(radius_tempx);ellipse.get_rady(radius_tempy); ellipse.get_center(center); ellipse.get_ecc(foci);
        std::cout << "Ellipse found in this step:  radiusx = "<< radius_tempx << " radiusy = "<< radius_tempy << " center = "<< center << " foci = "<< foci 
                  <<  " bd_p = " << bd_point << std::endl; 
    }
}

template class FindEllipse<1>;
template class FindEllipse<2>;

}
}