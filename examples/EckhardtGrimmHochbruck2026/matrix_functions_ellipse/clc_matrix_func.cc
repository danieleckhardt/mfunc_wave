#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/foreach.hpp>
#include <deal.II/base/utilities.h>



#include "../../../algorithm/matrix_function.h"
#include "../../../algorithm/global.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include <list>
#include <chrono> // Walltime

namespace pt = boost::property_tree;
int main(int argc, char **argv)
{
// SlepcInitialize(&argc, &argv, NULL, NULL);
try
    {
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        auto start_time = std::chrono::steady_clock::now();
        using namespace dealii;
        using namespace MavesS;
        using namespace TimeIntegration;


        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///////

        const int example = 202; // choose the Example
        const unsigned int dim = 2; // if the example number = 1x, than set dim = 1, if example number = 2x, than  dim = 2

        ///////// Usually you do not have to change anything above the line
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        std::cout<< "------Start simulation------" << std::endl;
        pt::ini_parser::read_ini("../../../../config/example"+ std::to_string(example) + ".ini", MavesS::global::config);
        std::vector<double> err_vector;

        MatrixFunction<dim>::MethodMatrixFunction mode;
        MatrixFunction<dim>::EllipseComputation   ellipse_mode;
        std::string mode_string    = global::config.get<std::string>("Calculation_Mode.method_matrix_function");
        std::string beta_string    = global::config.get<std::string>("Data.beta");
        std::string ellipse_string = global::config.get<std::string>("Calculation_Mode.ellipse_method");

        std::string delimiter = ",";

        size_t pos = 0;
        std::list<std::string> mode_list;
        std::list<std::string> beta_list;
        std::list<std::string> ellipse_list;
        while ((pos = mode_string.find(delimiter)) != std::string::npos) {
            mode_list.push_back(mode_string.substr(0, pos));
            mode_string.erase(0, pos + delimiter.length());
        }
        pos = 0;
        while ((pos = beta_string.find(delimiter)) != std::string::npos) {
            beta_list.push_back(beta_string.substr(0, pos));
            beta_string.erase(0, pos + delimiter.length());
        }
        pos = 0;
        while ((pos = ellipse_string.find(delimiter)) != std::string::npos) {
            ellipse_list.push_back(ellipse_string.substr(0, pos));
            ellipse_string.erase(0, pos + delimiter.length());
        }
        mode_list.push_back(mode_string.substr(0, pos));
        beta_list.push_back(beta_string.substr(0, pos));
        ellipse_list.push_back(ellipse_string.substr(0, pos));
            std::cout << "methods = { ";
        for (std::string n : mode_list)
            std::cout << n << ", ";
        std::cout << "};\n";
            std::cout << "betas = { ";
        for (std::string n : beta_list)
            std::cout << n << ", ";
        std::cout << "};\n";
            std::cout << "ellipse methods = { ";
        for (std::string n : ellipse_list)
            std::cout << n << ", ";
        std::cout << "};\n";

        // collected for the final comparison
        std::vector<std::string> cmp_beta, cmp_ell, cmp_method;
        std::vector<double>      cmp_radiusx, cmp_radiusy, cmp_center;
        std::vector<double>      cmp_capacity, cmp_extent, cmp_time_ell, cmp_err;
        std::vector<int>         cmp_degree, cmp_mv;

        MatrixFunction<dim> matrix_function_test;
        err_vector.resize(beta_list.size()*mode_list.size()*ellipse_list.size());
        matrix_function_test.setup_system();

        for( std::string b : beta_list )
        {
          std::cout << std::endl;
          std::cout << "#############################################" << std::endl;
          std::cout << "#############################################" << std::endl;
          std::cout << "Compute reference quantities for beta = " << b<< std::endl;
          std::cout << "#############################################" << std::endl;
          std::cout << "#############################################" << std::endl;
          std::cout << std::endl;

          double beta_value = std::stod(b);
          matrix_function_test.setup_matrices(beta_value);
          matrix_function_test.expoint.setup_system_macro_space();
          if(global::config.get<bool>("Calculation_Mode.clc_ref_ellipse")){
            matrix_function_test.create_ref_ellipse();
          }
          matrix_function_test.create_ref_matrix_func_poly_krylov();

          for(std::string ellipse_mode_string : ellipse_list)
          {
            if(ellipse_mode_string == "ritz")
              ellipse_mode = MatrixFunction<dim>::ritz;
            else if(ellipse_mode_string == "enclosure")
              ellipse_mode = MatrixFunction<dim>::enclosure;
            else
            {
              Assert(false, ExcNotImplemented());
              ellipse_mode = MatrixFunction<dim>::ritz;
            }
            matrix_function_test.set_ellipse_method(ellipse_mode);

            std::cout << std::endl;
            std::cout << "---------------------------------------------" << std::endl;
            std::cout << "Ellipse computation: " << ellipse_mode_string << std::endl;
            std::cout << "---------------------------------------------" << std::endl;

            for(std::string mode_string : mode_list)
            {
                double err_temp;
                double computation_time_temp;
                int mv_temp;
                if(mode_string == "poly_krylov")
                  mode = MatrixFunction<dim>::poly_krylov;
                else if(mode_string == "rat_krylov")
                  mode = MatrixFunction<dim>::rat_krylov;
                else if(mode_string == "cheb")
                  mode = MatrixFunction<dim>::cheb;
                else if(mode_string == "leja")
                  mode = MatrixFunction<dim>::leja;
                else
                {
                Assert(false, ExcNotImplemented());
                  mode = MatrixFunction<dim>::poly_krylov;
                }
                matrix_function_test.set_method(mode);
                matrix_function_test.run(b);
                matrix_function_test.get_results(err_temp, computation_time_temp, mv_temp);
                err_vector.push_back(err_temp);

                double e_radiusx, e_radiusy, e_center, e_capacity, e_extent, e_time_ell;
                int e_degree;
                matrix_function_test.get_ellipse_results(e_radiusx, e_radiusy, e_center,
                                                         e_capacity, e_extent, e_time_ell, e_degree);
                cmp_beta.push_back(b);
                cmp_ell.push_back(ellipse_mode_string);
                cmp_method.push_back(mode_string);
                cmp_radiusx.push_back(e_radiusx);
                cmp_radiusy.push_back(e_radiusy);
                cmp_center.push_back(e_center);
                cmp_capacity.push_back(e_capacity);
                cmp_extent.push_back(e_extent);
                cmp_time_ell.push_back(e_time_ell);
                cmp_degree.push_back(e_degree);
                cmp_err.push_back(err_temp);
                cmp_mv.push_back(mv_temp);
            }
          }
        }

    //
    // Comparison of the two ellipse computations
    //
    std::cout << std::endl;
    std::cout << "=====================================================================================" << std::endl;
    std::cout << " Comparison of the ellipse computations" << std::endl;
    std::cout << "=====================================================================================" << std::endl;
    std::cout << std::left
              << std::setw(9)  << "beta"
              << std::setw(12) << "ellipse"
              << std::setw(8)  << "method"
              << std::right
              << std::setw(13) << "radiusx"
              << std::setw(13) << "radiusy"
              << std::setw(13) << "center"
              << std::setw(13) << "gamma"
              << std::setw(11) << "Re_max"
              << std::setw(9)  << "degree"
              << std::setw(11) << "t_ell[s]"
              << std::setw(13) << "err(last)"
              << std::setw(9)  << "mv"
              << std::endl;
    for (unsigned long i = 0; i < cmp_beta.size(); ++i)
    {
        std::cout << std::left
                  << std::setw(9)  << cmp_beta[i]
                  << std::setw(12) << cmp_ell[i]
                  << std::setw(8)  << cmp_method[i]
                  << std::right << std::scientific << std::setprecision(4)
                  << std::setw(13) << cmp_radiusx[i]
                  << std::setw(13) << cmp_radiusy[i]
                  << std::setw(13) << cmp_center[i]
                  << std::setw(13) << cmp_capacity[i]
                  << std::setw(11) << cmp_extent[i]
                  << std::defaultfloat
                  << std::setw(9)  << cmp_degree[i]
                  << std::scientific << std::setprecision(2)
                  << std::setw(11) << cmp_time_ell[i]
                  << std::setw(13) << cmp_err[i]
                  << std::defaultfloat
                  << std::setw(9)  << cmp_mv[i]
                  << std::endl;
    }
    std::cout << "-------------------------------------------------------------------------------------" << std::endl;
    std::cout << " gamma  = capacity of the confocal interval, |ecc|/2 (the gamma of the Leja recursion)" << std::endl;
    std::cout << " Re_max = center + radiusx, rightmost point. phi_k grows like exp(Re_max) there," << std::endl;
    std::cout << "          so the attainable relative accuracy is about exp(Re_max)*eps_mach."      << std::endl;
    std::cout << " err/mv = value of the last (smallest) tolerance of the sweep."                    << std::endl;
    std::cout << "=====================================================================================" << std::endl;

    std::cout<< "-------End simulation------" << std::endl;

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    int size;
    matrix_function_test.get_size(size);
    std::cout
      << "=============================================="
      << std::endl
      << "   Example:                      " << global::config.get<std::string>("Calculation_Mode.example")
      << std::endl
      << "   Method:                       " << global::config.get<std::string>("Calculation_Mode.method_matrix_function")
      << std::endl
      << "   Ellipse method:               " << global::config.get<std::string>("Calculation_Mode.ellipse_method")
      << std::endl
      << "   Grid width:                   2^-" << global::config.get<std::string>("Space_Discretization.initial_refinement")
      << std::endl
      << "   FE degree:                    " << global::config.get<std::string>("Space_Discretization.fe_degree")
      << std::endl
      << "   Matrix size:                  " << size
      << std::endl
      << "   Epsilon:                      2^" << global::config.get<std::string>("Data.epsilon")
      << std::endl
      << "   Beta:                         " << global::config.get<std::string>("Data.beta")
      << std::endl
      << "   Total Time:                   " << elapsed_seconds.count() << " s"
      << std::endl
      << "=============================================="
      << std::endl;
    std::cout << "Path to Logfile:" << "../output/LastRun.txt" << std::endl;
  }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      // SlepcFinalize();
      return 1;

    }

  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      // SlepcFinalize();
      return 1;
    }
  // SlepcFinalize();
  return 0;
}