#include "leja_approx.h"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>


namespace MavesS{
namespace TimeIntegration
{

template<int dim>
Leja<dim>::Leja(const int degree, Types::Ellipse<dim>& ellipse, double& time_step_size,
                 const SparseMatrix<double>& A, const SparseMatrix<double>& M)
    :
    ellipse(ellipse)
    ,max_degree(degree)
    ,time_step_size(time_step_size)
    ,A(A)
    ,M(M)
{

    if (max_degree < 1)
        throw std::invalid_argument(
            "Leja<dim>::Leja: degree (Iterationsrahmen) muss >= 1 sein.");
}

template<int dim>
void Leja<dim>::update_geometry()
{

    compute_leja_geometry();

    ellipse.get_radx(cached_radiusx);
    ellipse.get_rady(cached_radiusy);
    ellipse.get_center(cached_center);


    const int new_case = use_complex_points ? 1 : 0;
    if (new_case != points_case)
    {
        precompute_leja_points(static_cast<unsigned int>(max_degree));
        points_case = new_case;
    }

    divided_differences.clear();
    k_phi_current = -1;
    dd_degree     = -1;
    degree        = 0;
}

template<int dim>
void Leja<dim>::prepare_coefficients(int k_phi)
{

    if (k_phi < 0)
        throw std::invalid_argument("Leja<dim>::prepare_coefficients: k_phi muss >= 0 sein.");

    refresh_geometry_if_needed();
    compute_divided_differences(static_cast<unsigned int>(max_degree), k_phi);
}

template<int dim>
void Leja<dim>::refresh_geometry_if_needed()
{

    double radiusx, radiusy, center;
    ellipse.get_radx(radiusx);
    ellipse.get_rady(radiusy);
    ellipse.get_center(center);

    const bool unchanged = (radiusx == cached_radiusx)
                        && (radiusy == cached_radiusy)
                        && (center  == cached_center);

    if (!unchanged)
        update_geometry();
}

template<int dim>
void Leja<dim>::compute_leja_geometry()
{


    double radiusx, radiusy, center;
    ellipse.get_radx(radiusx);
    ellipse.get_rady(radiusy);
    ellipse.get_center(center);

    std::complex<double> ecc;
    ellipse.get_ecc(ecc);

    if (std::abs(ecc) < 1e-10)
        throw std::invalid_argument(
            "Leja<dim>::compute_leja_geometry: |ecc| ist (nahezu) null - "
            "die Ellipse entartet zu einem Kreis, das konfokale Intervall "
            "kollabiert auf einen Punkt (Kapazitaet 0). Leja-Interpolation "
            "ist in diesem Grenzfall nicht definiert.");

    leja_center         = center;
    leja_capacity        = std::abs(ecc) / 2.0;
    use_complex_points  = (radiusx < radiusy);
}

template<int dim>
void Leja<dim>::precompute_leja_points(unsigned int degree_max)
{


    const unsigned int n_candidates = 20000;
    leja_points_ref.assign(degree_max + 1, std::complex<double>(0., 0.));

    std::vector<std::complex<double>> candidates(n_candidates);
    for (unsigned int i = 0; i < n_candidates; ++i)
    {
        double t = -2.0 + 4.0 * static_cast<double>(i) / static_cast<double>(n_candidates - 1);
        candidates[i] = use_complex_points ? std::complex<double>(0., t)
                                            : std::complex<double>(t, 0.);
    }


    std::vector<double> log_prod_sum(n_candidates, 0.0);

    auto pick_argmax = [&]() -> unsigned int {
        unsigned int best_idx = 0;
        double best_val = -std::numeric_limits<double>::infinity();
        for (unsigned int i = 0; i < n_candidates; ++i)
        {
            if (log_prod_sum[i] > best_val)
            {
                best_val = log_prod_sum[i];
                best_idx = i;
            }
        }
        return best_idx;
    };

    auto update_log_prod_sum = [&](const std::complex<double>& xi_new) {
        for (unsigned int i = 0; i < n_candidates; ++i)
        {
            double dist = std::abs(candidates[i] - xi_new);

            log_prod_sum[i] += std::log(dist);
        }
    };

    if (!use_complex_points)
    {

        leja_points_ref[0] = std::complex<double>(2.0, 0.0);
        update_log_prod_sum(leja_points_ref[0]);

        for (unsigned int m = 1; m <= degree_max; ++m)
        {
            unsigned int idx = pick_argmax();
            leja_points_ref[m] = candidates[idx];
            update_log_prod_sum(leja_points_ref[m]);
        }
    }
    else
    {

        leja_points_ref[0] = std::complex<double>(0.0, 0.0);
        update_log_prod_sum(leja_points_ref[0]);

        unsigned int m = 1;
        while (m <= degree_max)
        {
            unsigned int idx = pick_argmax();
            leja_points_ref[m] = candidates[idx];
            update_log_prod_sum(leja_points_ref[m]);

            if (m + 1 <= degree_max)
            {
                leja_points_ref[m + 1] = -leja_points_ref[m];
                update_log_prod_sum(leja_points_ref[m + 1]);
            }
            m += 2;
        }
    }
}

template<int dim>
std::complex<double> Leja<dim>::phi_scalar(int k, std::complex<double> z)
{

    if (k == 0)
        return std::exp(z);

    if (std::abs(z) < 0.5)
    {
        std::complex<double> term = 1.0;
        for (int i = 1; i <= k; ++i)
            term /= static_cast<double>(i);          // term = 1/k!
        std::complex<double> sum = term;
        for (int n = 1; n <= 60; ++n)
        {
            term *= z / static_cast<double>(n + k);
            sum += term;
            if (std::abs(term) < 1e-17 * std::abs(sum))
                break;
        }
        return sum;
    }

    std::complex<double> phi_prev = std::exp(z);     // phi_0(z)
    double inv_factorial = 1.0;                      // 1/(j-1)! fuer j=1: 1/0!
    for (int j = 1; j <= k; ++j)
    {
        phi_prev = (phi_prev - inv_factorial) / z;
        inv_factorial /= static_cast<double>(j);
    }
    return phi_prev;
}

template<int dim>
void Leja<dim>::compute_taylor_phi_matrix(int idx, double tau_sub, unsigned int m,
                                           FullMatrix<std::complex<double>>& F) const
{

    const double S = 1e300;
    const double log_S = std::log(S);
    const double log_tau_gamma = std::log(tau_sub * leja_capacity);

    F.reinit(m + 1, m + 1);

    for (unsigned int j = 0; j <= m; ++j)
    {
        for (unsigned int i = j; i <= m; ++i)
        {
            const unsigned int D = i - j;
            const double log_val = log_S
                                    + static_cast<double>(D) * log_tau_gamma
                                    - std::lgamma(static_cast<double>(D + idx) + 1.0);
            const double val = std::exp(log_val);
            F(i, j) = std::complex<double>(val, 0.0);
            F(j, i) = std::complex<double>(val, 0.0);
        }
    }


    for (int l = 2; l <= 17; ++l)
    {
        for (unsigned int j = 0; j + 1 <= m; ++j)
        {
            const std::complex<double> z_j =
                leja_center + leja_capacity * leja_points_ref[j];
            F(j, j) = tau_sub * z_j * F(j, j)
                      / static_cast<double>(l + idx - 1);

            for (unsigned int i = j + 1; i <= m; ++i)
            {
                const std::complex<double> z_i =
                    leja_center + leja_capacity * leja_points_ref[i];
                F(j, i) = tau_sub * (z_i * F(j, i) + leja_capacity * F(j, i - 1))
                          / static_cast<double>(l + static_cast<int>(i - j) + idx - 1);
                F(i, j) = F(i, j) + F(j, i);
            }
        }
    }

    for (unsigned int i = 0; i <= m; ++i)
        for (unsigned int j = 0; j < i; ++j)
            F(i, j) /= S;

    for (unsigned int j = 0; j <= m; ++j)
    {
        const std::complex<double> z_j =
            leja_center + leja_capacity * leja_points_ref[j];
        F(j, j) = phi_scalar(idx, tau_sub * z_j);
    }

}

template<int dim>
Vector<std::complex<double>> Leja<dim>::apply_lower_triangular(
        const FullMatrix<std::complex<double>>& F,
        const Vector<std::complex<double>>& w) const
{
  
    const unsigned int n = w.size();
    Vector<std::complex<double>> result(n);
    for (unsigned int i = 0; i < n; ++i)
    {
        std::complex<double> sum = 0.;
        for (unsigned int j = 0; j <= i; ++j)
            sum += F(i, j) * w(j);
        result(i) = sum;
    }
    return result;
}

template<int dim>
void Leja<dim>::compute_divided_differences(unsigned int m, int k_phi)
{
    refresh_geometry_if_needed();

    if (leja_points_ref.size() <= m)
        throw std::invalid_argument(
            "Leja<dim>::compute_divided_differences: angeforderter Grad "
            "uebersteigt die Anzahl vorberechneter Referenz-Leja-Punkte "
            "(max_degree).");

    divided_differences.assign(m + 1, std::complex<double>(0., 0.));

    double z_max = 0.0;
    for (unsigned int i = 0; i <= m; ++i)
    {
        const std::complex<double> z_i =
            leja_center + leja_capacity * leja_points_ref[i];
        z_max = std::max(z_max, std::abs(z_i));
    }
    unsigned int J = static_cast<unsigned int>(std::ceil(z_max / 1.59));
    if (J == 0) J = 1;
    const double tau_sub = 1.0 / static_cast<double>(J);

    Vector<std::complex<double>> e1(m + 1);
    e1(0) = 1.0;

    FullMatrix<std::complex<double>> F0;
    compute_taylor_phi_matrix(0, tau_sub, m, F0);   // phi_0(tau_sub*H_m)

    if (k_phi == 0)
    {

        Vector<std::complex<double>> y = e1;
        for (unsigned int step = 0; step < J; ++step)
            y = apply_lower_triangular(F0, y);

        for (unsigned int i = 0; i <= m; ++i)
            divided_differences[i] = y(i);
    }
    else
    {

        std::vector<Vector<std::complex<double>>> source_terms(k_phi);
        {
            FullMatrix<std::complex<double>> F_tmp;
            for (int i = 0; i <= k_phi - 1; ++i)
            {
                compute_taylor_phi_matrix(i + 1, tau_sub, m, F_tmp);
                Vector<std::complex<double>> col0(m + 1);
                for (unsigned int r = 0; r <= m; ++r)
                    col0(r) = F_tmp(r, 0);
                source_terms[i] = col0;
            }
        }

        Vector<std::complex<double>> y(m + 1);   // y_0 = 0

        double t_j = 0.0;
        for (unsigned int step = 0; step < J; ++step)
        {
            Vector<std::complex<double>> y_new = apply_lower_triangular(F0, y);

            for (int i = 0; i <= k_phi - 1; ++i)
            {
                const int exponent = k_phi - 1 - i;
                double t_pow = (exponent == 0) ? 1.0 : std::pow(t_j, exponent);
                double inv_fact = 1.0;
                for (int q = 1; q <= exponent; ++q)
                    inv_fact /= static_cast<double>(q);
                const double scale = t_pow * inv_fact * std::pow(tau_sub, i + 1);

                for (unsigned int r = 0; r <= m; ++r)
                    y_new(r) += scale * source_terms[i](r);
            }

            y = y_new;
            t_j += tau_sub;
        }

        for (unsigned int i = 0; i <= m; ++i)
            divided_differences[i] = y(i);
    }

    dd_degree     = static_cast<int>(m);
    k_phi_current = k_phi;
}

template<int dim>
void Leja<dim>::ensure_divided_differences(int m_needed, int k_phi)
{
    if (k_phi != k_phi_current)
    {
        dd_degree = -1;          // Wechsel von k_phi macht alles ungueltig
    }

    if (dd_degree >= m_needed)
        return;

    int target = (dd_degree < 0) ? dd_initial_block : 2 * dd_degree;
    if (target < m_needed) target = m_needed;
    if (target > max_degree) target = max_degree;

    compute_divided_differences(static_cast<unsigned int>(target), k_phi);
}


template<int dim>
void Leja<dim>::get_degree(int& degree_) const
{
    degree_ = degree;
}

template<int dim>
void Leja<dim>::get_divided_differences(std::vector<std::complex<double>>& d_out) const
{
    d_out = divided_differences;
}

template<int dim>
void Leja<dim>::set_degree(int degree_){
    degree = degree_;
}

template <int dim>
double Leja<dim>::scalar_product(
    const Vector<std::complex<double>> &a,
    const Vector<std::complex<double>> &b,
    const SparseMatrix<double> &matrix,
    bool use_matrix)
{

    Vector<std::complex<double>> tmp_vec(a.size());

    if (use_matrix)
    {
        Vector<double> tmp_real(a.size());
        Vector<double> tmp_imag(a.size());
        Vector<double> a_real(a.size());
        Vector<double> a_imag(a.size());

        for (unsigned int i = 0; i < a.size(); ++i)
        {
            a_real[i] = a[i].real();
            a_imag[i] = a[i].imag();
        }
        if (a_real.l1_norm() > 1e-12){
            matrix.vmult(tmp_real, a_real);
            mv_count += 1;
        }
        if (a_imag.l1_norm() > 1e-12){
            matrix.vmult(tmp_imag, a_imag);
            mv_count += 1;
        }

        for (unsigned int i = 0; i < a.size(); ++i)
            tmp_vec[i] = {tmp_real[i], tmp_imag[i]};
    }
    else
        tmp_vec = a;

    std::complex<double> result = 0.0;
    for (unsigned int i = 0; i < b.size(); ++i)
        result += std::conj(b[i]) * tmp_vec[i];

    return result.real();
}

template <int dim>
double Leja<dim>::calculate_residual(int k_phi)
{
    res = Matrix_x_vector;
    res.add(-1.0, vector_primes);

    Vector<std::complex<double>> res_u(res.m());
    Vector<std::complex<double>> res_v(res.m());
    Vector<std::complex<double>> Ma_u(res.m());
    Vector<std::complex<double>> Ma_v(res.m());

    for (unsigned long i = 0; i < res.m(); ++i){
        res_u(i) = res(i,0);
        res_v(i) = res(i,1);
        Ma_u(i)  = Matrix_x_vector(i,0);
        Ma_v(i)  = Matrix_x_vector(i,1);
    }

    double l2_norm_tmp = std::sqrt(scalar_product(Ma_u,Ma_u,A) + scalar_product(Ma_v,Ma_v,M));
    l2_norm_tmp = std::min(l2_norm_tmp, 1.); // avoid too early stops due to very small norms

    double residual = std::sqrt(scalar_product(res_u,res_u,A) + scalar_product(res_v,res_v,M));

    if (k_phi > 0)
        residual /= static_cast<double>(k_phi);

    return residual / l2_norm_tmp;
}

template<int dim>
void Leja<dim>::apply_leja_real(
    BMatrixVariant Matrix_mult,
    FullMatrix<std::complex<double>> &blockvector,
    double tol,
    int k_phi)
{


    auto apply_matrix = [&](FullMatrix<std::complex<double>>& X) {
        mv_count += std::get<std::function<int(FullMatrix<std::complex<double>>&)>>(Matrix_mult)(X);
    };

    const double t   = time_step_size;
    const double gam = leja_capacity;
    const double c   = leja_center;

    if (std::abs(t) < 1e-14)
        throw std::invalid_argument(
            "Leja<dim>::apply_leja_real: time_step_size is near zero");

    ensure_divided_differences(1, k_phi);

    const FullMatrix<std::complex<double>> v = blockvector;  

    FullMatrix<std::complex<double>> Av = v;
    apply_matrix(Av);

    FullMatrix<std::complex<double>> p = v;
    p *= divided_differences[0];                       // p_0 = d_0 * v

    FullMatrix<std::complex<double>> r = Av;           // r_0 = B(v) - xi_0 * v
    r.add(-c, v);
    r *= 1.0 / gam;
    r.add(-leja_points_ref[0], v);

    FullMatrix<std::complex<double>> p_prime = v;
    p_prime *= 0.;                                     // p'_0 = 0

    FullMatrix<std::complex<double>> r_prime = Av;     // r'_0 = (1/gamma) * A * v
    r_prime *= 1.0 / (gam * t);

    FullMatrix<std::complex<double>> Ap_t = Av;
    Ap_t *= divided_differences[0];

    FullMatrix<std::complex<double>> source = v;
    if (k_phi == 0)
    {
        source *= 0.;
    }
    else
    {
        double inv_fact = 1.0;                          // 1/(k_phi-1)!
        for (int q = 1; q <= k_phi - 1; ++q)
            inv_fact /= static_cast<double>(q);
        source *= inv_fact;
    }

    FullMatrix<std::complex<double>> Ar, Arp, r_new, r_prime_new;

    bool converged = false;
    for (int m = 1; m <= max_degree; ++m)
    {
        ensure_divided_differences(m, k_phi);

        Ar  = r;        apply_matrix(Ar);
        Arp = r_prime;  apply_matrix(Arp);

        p.add(divided_differences[m], r);
        p_prime.add(divided_differences[m], r_prime);

        Ap_t.add(divided_differences[m], Ar);

        if (k_phi == 0)
        {
            Matrix_x_vector = Ap_t;
        }
        else
        {
            Matrix_x_vector = Ap_t;
            Matrix_x_vector.add(1.0, source);
            Matrix_x_vector.add(-static_cast<double>(k_phi), p);
        }
        vector_primes = p_prime;
        vector_primes *= t;

        const double residual = calculate_residual(k_phi);
        if (residual < tol)
        {
            std::cout << "  Leja (real) converged:"
                      << " degree: "   << m
                      << ", Residual: " << residual
                      << std::endl;
            set_degree(m);
            converged = true;
            break;
        }


        r_new = Ar;
        r_new.add(-c, r);
        r_new *= 1.0 / gam;
        r_new.add(-leja_points_ref[m], r);

        r_prime_new = Arp;
        r_prime_new.add(-c, r_prime);
        r_prime_new *= 1.0 / gam;
        r_prime_new.add(-leja_points_ref[m], r_prime);
        r_prime_new.add(1.0 / (gam * t), Ar);

        r       = std::move(r_new);
        r_prime = std::move(r_prime_new);
    }

    if (!converged)
    {
        set_degree(max_degree);
        throw std::runtime_error(
            "Leja<dim>::apply_leja_real: no convergence up to the maximum "
            "degree (max_degree). The most common cause is STAGNATION of the residual: "
            "with a large capacity of the ellipse, rounding errors limit the "
            "achievable accuracy; the residual then falls to a "
            "plateau and does not decrease further. Solution: Reduce the time step (and thus the "
            "capacity) or loosen the tolerance. Less common: max_degree is set too "
            "low.");
    }

    blockvector = p;
}

template<int dim>
void Leja<dim>::apply_leja_complex(
    BMatrixVariant Matrix_mult,
    FullMatrix<std::complex<double>> &blockvector,
    double tol,
    int k_phi)
{

    auto apply_matrix = [&](FullMatrix<std::complex<double>>& X) {
        mv_count += std::get<std::function<int(FullMatrix<std::complex<double>>&)>>(Matrix_mult)(X);
    };

    const double t   = time_step_size;
    const double gam = leja_capacity;
    const double c   = leja_center;

    if (std::abs(t) < 1e-14)
        throw std::invalid_argument(
            "Leja<dim>::apply_leja_complex: time_step_size near zero");

    ensure_divided_differences(2, k_phi);

    const FullMatrix<std::complex<double>> v = blockvector;

    FullMatrix<std::complex<double>> Av = v;
    apply_matrix(Av);

    // --- Initialisierung, Gl. (4d) und (19d) ---
    FullMatrix<std::complex<double>> p = v;
    p *= divided_differences[0];                       // p_0 = d_0 * v

    FullMatrix<std::complex<double>> r = Av;           // r_0 = (1/gamma)(tA - c) v
    r.add(-c, v);
    r *= 1.0 / gam;

    FullMatrix<std::complex<double>> p_prime = v;
    p_prime *= 0.;                                     // p'_0 = 0

    FullMatrix<std::complex<double>> r_prime = Av;     // r'_0 = (1/gamma) A v
    r_prime *= 1.0 / (gam * t);

    FullMatrix<std::complex<double>> Ap_t = Av;        // t*A*p_0
    Ap_t *= divided_differences[0];

    FullMatrix<std::complex<double>> source = v;
    if (k_phi == 0)
    {
        source *= 0.;
    }
    else
    {
        double inv_fact = 1.0;
        for (int q = 1; q <= k_phi - 1; ++q)
            inv_fact /= static_cast<double>(q);
        source *= inv_fact;
    }

    FullMatrix<std::complex<double>> Ar, Arp, Aq, Aqp, r_new, r_prime_new;

    bool converged = false;
    for (int m = 2; m <= max_degree; m += 2)
    {
        ensure_divided_differences(m, k_phi);

        const double b_im  = leja_points_ref[m-1].imag();
        const double b_im2 = b_im * b_im;               // Im(xi_{m-1})^2

        Ar  = r;        apply_matrix(Ar);
        Arp = r_prime;  apply_matrix(Arp);

        q_m = Ar;
        q_m.add(-c, r);
        q_m *= 1.0 / gam;

        q_prime_m = Arp;
        q_prime_m.add(-c, r_prime);
        q_prime_m *= 1.0 / gam;
        q_prime_m.add(1.0 / (gam * t), Ar);

        const double     re_dm1 = divided_differences[m-1].real();
        const std::complex<double> dm = divided_differences[m];

        p.add(re_dm1, r);
        p.add(dm, q_m);

        p_prime.add(re_dm1, r_prime);
        p_prime.add(dm, q_prime_m);

        Aq = q_m; apply_matrix(Aq);

        Ap_t.add(re_dm1, Ar);
        Ap_t.add(dm, Aq);

        if (k_phi == 0)
        {
            Matrix_x_vector = Ap_t;
        }
        else
        {
            Matrix_x_vector = Ap_t;
            Matrix_x_vector.add(1.0, source);
            Matrix_x_vector.add(-static_cast<double>(k_phi), p);
        }
        vector_primes = p_prime;
        vector_primes *= t;

        const double residual = calculate_residual(k_phi);
        if (residual < tol)
        {
            std::cout << "  Leja (complex conjugate) converged:"
                      << " degree: "   << m
                      << ", Residual: " << residual
                      << std::endl;
            set_degree(m);
            converged = true;
            break;
        }

        r_new = Aq;
        r_new.add(-c, q_m);
        r_new *= 1.0 / gam;
        r_new.add(b_im2, r);

 
        Aqp = q_prime_m; apply_matrix(Aqp);

        r_prime_new = Aqp;
        r_prime_new.add(-c, q_prime_m);
        r_prime_new *= 1.0 / gam;
        r_prime_new.add(1.0 / (gam * t), Aq);
        r_prime_new.add(b_im2, r_prime);

        r       = std::move(r_new);
        r_prime = std::move(r_prime_new);
    }

    if (!converged)
    {
        set_degree(max_degree - (max_degree % 2));
        throw std::runtime_error(
            "Leja<dim>::apply_leja_complex: no convergence up to the maximum "
            "degree (max_degree). The most common cause is STAGNATION of the residual: "
            "with a large capacity of the ellipse, rounding errors limit the "
            "achievable accuracy; the residual then falls to a "
            "plateau and does not decrease further. Solution: Reduce the time step (and thus the "
            "capacity) or loosen the tolerance. Less common: max_degree is set too "
            "low.");
    }

    blockvector = p;
}

template<int dim>
void Leja<dim>::matrix_run(
    BMatrixVariant Matrix_mult,
    FullMatrix<std::complex<double>> &blockvector,
    double tol,
    int k_phi)
{


    if (k_phi < 0)
        throw std::invalid_argument("Leja<dim>::matrix_run: k_phi muss >= 0 sein.");

    mv_count = 0;


    refresh_geometry_if_needed();



    if (use_complex_points)
        apply_leja_complex(Matrix_mult, blockvector, tol, k_phi);
    else
        apply_leja_real(Matrix_mult, blockvector, tol, k_phi);
}

template class Leja<1>;
template class Leja<2>;

}/* namespace TimeIntegration*/
}/* namespace MavesS */