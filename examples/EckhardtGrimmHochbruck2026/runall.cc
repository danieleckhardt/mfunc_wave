#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include <deal.II/base/utilities.h>

#include "../../algorithm/matrix_function.h"
#include "../../algorithm/global.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <chrono>
#include <vector>
#include <list>
#include <sstream>

namespace pt = boost::property_tree;

using namespace dealii;
using namespace MavesS;
using namespace TimeIntegration;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: split comma-separated string
// ─────────────────────────────────────────────────────────────────────────────
std::list<std::string> split(const std::string &input, char delimiter = ',')
{
    std::list<std::string> result;
    std::stringstream ss(input);
    std::string item;
    while (std::getline(ss, item, delimiter))
        result.push_back(item);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Comparison record (only filled for example 202)
// ─────────────────────────────────────────────────────────────────────────────
struct EllipseCompRow
{
    std::string beta;
    std::string ell_method;
    std::string mat_method;
    double radiusx, radiusy, center, capacity, extent, time_ell;
    int    degree, mv;
    double err;
};

// ─────────────────────────────────────────────────────────────────────────────
// Templated runner for a single example
// ─────────────────────────────────────────────────────────────────────────────
template <int dim>
void run_example(int example)
{
    std::cout << "\n------Start simulation (dim=" << dim
              << ", example=" << example << ")------\n";

    auto start_time = std::chrono::steady_clock::now();

    pt::ini_parser::read_ini(
        "../../../config/example" + std::to_string(example) + ".ini",
        global::config);

    // ── parse modes ──────────────────────────────────────────────────────────
    auto mode_list = split(
        global::config.get<std::string>("Calculation_Mode.method_matrix_function"));
    auto beta_list = split(
        global::config.get<std::string>("Data.beta"));

    // For example 202, read ellipse_method; fall back to "ritz" for all others.
    const bool is_202 = (example == 202);
    std::list<std::string> ell_list;
    if (is_202)
        ell_list = split(global::config.get<std::string>(
                             "Calculation_Mode.ellipse_method", "ritz"));
    else
        ell_list.push_back("ritz");

    std::cout << "methods        = { ";
    for (const auto &m : mode_list) std::cout << m << " ";
    std::cout << "}\n";
    std::cout << "betas          = { ";
    for (const auto &b : beta_list) std::cout << b << " ";
    std::cout << "}\n";
    if (is_202) {
        std::cout << "ellipse methods = { ";
        for (const auto &e : ell_list) std::cout << e << " ";
        std::cout << "}\n";
    }

    MatrixFunction<dim> mf;
    mf.setup_system();

    std::vector<EllipseCompRow> cmp_rows; // filled only for example 202
    std::vector<double>         err_vector;

    for (const std::string &b : beta_list)
    {
        std::cout << "\n=============================================\n";
        std::cout << "beta = " << b << "\n";
        std::cout << "=============================================\n";

        double beta_value = std::stod(b);
        mf.setup_matrices(beta_value);
        mf.expoint.setup_system_macro_space();

        if (global::config.get<bool>("Calculation_Mode.clc_ref_ellipse", false))
            mf.create_ref_ellipse();

        mf.create_ref_matrix_func_poly_krylov();

        for (const std::string &ell_name : ell_list)
        {
            // Set ellipse computation method
            if (is_202)
            {
                typename MatrixFunction<dim>::EllipseComputation ell_mode;
                if      (ell_name == "enclosure") ell_mode = MatrixFunction<dim>::enclosure;
                else if (ell_name == "ritz")      ell_mode = MatrixFunction<dim>::ritz;
                else
                    throw std::runtime_error("Unknown ellipse method: " + ell_name);
                mf.set_ellipse_method(ell_mode);

                if (is_202)
                {
                    std::cout << "\n--- ellipse method: " << ell_name << " ---\n";
                }
            }

            for (const std::string &mode_name : mode_list)
            {
                // Methods that don't use the ellipse: skip the second ell pass
                // to avoid running them twice.
                const bool ell_independent =
                    (mode_name == "poly_krylov" || mode_name == "rat_krylov");
                if (is_202 && ell_independent && ell_name != ell_list.front())
                    continue;

                typename MatrixFunction<dim>::MethodMatrixFunction mode;
                if      (mode_name == "poly_krylov") mode = MatrixFunction<dim>::poly_krylov;
                else if (mode_name == "rat_krylov")  mode = MatrixFunction<dim>::rat_krylov;
                else if (mode_name == "cheb")        mode = MatrixFunction<dim>::cheb;
                else if (mode_name == "leja")        mode = MatrixFunction<dim>::leja;
                else
                    throw std::runtime_error("Unknown method: " + mode_name);

                mf.set_method(mode);
                mf.run(b);

                double err, time_mf;
                int    mv;
                mf.get_results(err, time_mf, mv);
                err_vector.push_back(err);

                // For example 202, collect ellipse parameters for comparison table.
                if (is_202 && !ell_independent)
                {
                    EllipseCompRow row;
                    row.beta       = b;
                    row.ell_method = ell_name;
                    row.mat_method = mode_name;
                    row.mv         = mv;
                    row.err        = err;
                    mf.get_ellipse_results(row.radiusx, row.radiusy, row.center,
                                           row.capacity, row.extent,
                                           row.time_ell, row.degree);
                    cmp_rows.push_back(row);
                }
            }
        }
    }

    // ── comparison table (example 202 only) ──────────────────────────────────
    if (is_202 && !cmp_rows.empty())
    {
        std::cout << "\n";
        std::cout << "=========================================================="
                     "=====================================\n";
        std::cout << " Ellipse comparison (cheb/leja only; poly_krylov has no ellipse)\n";
        std::cout << "=========================================================="
                     "=====================================\n";
        std::cout << std::left
                  << std::setw(8)  << "beta"
                  << std::setw(12) << "ellipse"
                  << std::setw(8)  << "method"
                  << std::right
                  << std::setw(12) << "radiusx"
                  << std::setw(12) << "radiusy"
                  << std::setw(12) << "center"
                  << std::setw(12) << "gamma"
                  << std::setw(10) << "Re_max"
                  << std::setw(8)  << "degree"
                  << std::setw(10) << "t_ell/s"
                  << std::setw(12) << "err"
                  << std::setw(8)  << "mv"
                  << "\n";
        std::cout << std::string(116, '-') << "\n";
        for (const auto &r : cmp_rows)
        {
            std::cout << std::left
                      << std::setw(8)  << r.beta
                      << std::setw(12) << r.ell_method
                      << std::setw(8)  << r.mat_method
                      << std::right << std::scientific << std::setprecision(3)
                      << std::setw(12) << r.radiusx
                      << std::setw(12) << r.radiusy
                      << std::setw(12) << r.center
                      << std::setw(12) << r.capacity
                      << std::setw(10) << r.extent
                      << std::defaultfloat
                      << std::setw(8)  << r.degree
                      << std::scientific << std::setprecision(2)
                      << std::setw(10) << r.time_ell
                      << std::setw(12) << r.err
                      << std::defaultfloat
                      << std::setw(8)  << r.mv
                      << "\n";
        }
        std::cout << std::string(116, '-') << "\n";
        std::cout << " gamma  = capacity of the confocal interval |ecc|/2 "
                     "(the gamma of the Leja recursion)\n";
        std::cout << " Re_max = center + radiusx; phi_k grows ~exp(Re_max) there\n";
        std::cout << "=========================================================="
                     "=====================================\n";
    }

    // ── summary ──────────────────────────────────────────────────────────────
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    int size;
    mf.get_size(size);

    std::cout << "\n==============================================\n";
    std::cout << "  Example:        "
              << global::config.get<std::string>("Calculation_Mode.example") << "\n";
    std::cout << "  Method:         "
              << global::config.get<std::string>("Calculation_Mode.method_matrix_function") << "\n";
    if (is_202)
        std::cout << "  Ellipse method: "
                  << global::config.get<std::string>("Calculation_Mode.ellipse_method") << "\n";
    std::cout << "  Grid width:     2^-"
              << global::config.get<std::string>("Space_Discretization.initial_refinement") << "\n";
    std::cout << "  FE degree:      "
              << global::config.get<std::string>("Space_Discretization.fe_degree") << "\n";
    std::cout << "  Matrix size:    " << size << "\n";
    std::cout << "  Epsilon:        2^"
              << global::config.get<std::string>("Data.epsilon") << "\n";
    std::cout << "  Beta:           "
              << global::config.get<std::string>("Data.beta") << "\n";
    std::cout << "  Total Time:     " << elapsed.count() << " s\n";
    std::cout << "==============================================\n";
    std::cout << "-------End simulation------\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char **argv)
{
    try
    {
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        // Add or remove examples here. Dim is determined by the number:
        //   1xx -> dim 1,  2xx/3xx -> dim 2.
        std::vector<int> examples = {201, 202, 301};

        for (int example : examples)
        {
            if (example < 200)
                run_example<1>(example);
            else
                run_example<2>(example);
        }
    }
    catch (std::exception &exc)
    {
        std::cerr << "\nException:\n" << exc.what() << "\n";
        return 1;
    }
    catch (...)
    {
        std::cerr << "\nUnknown exception!\n";
        return 1;
    }

    return 0;
}