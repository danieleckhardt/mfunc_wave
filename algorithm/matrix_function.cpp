#include "matrix_function.h"

namespace MavesS{
    using namespace dealii;
    using namespace TimeIntegration;

    template< typename VectorType>
    void generateRandomVector(VectorType& vec, double min_val, double max_val) {
        std::random_device rd;  // Seed
        std::mt19937 gen(rd()); // Mersenne Twister engine
        std::uniform_real_distribution<> dis(min_val, max_val);

        for (unsigned int i = 0; i < vec.size(); ++i) {
            vec[i] = dis(gen);  // Assign random value
        }
    }

    template <typename VectorType>
    void generateDiscreteSine(VectorType &vec, double frequency = 1.0)
    {
        const unsigned int N = vec.size();
        for (unsigned int i = 0; i < N; ++i)
        {
            double x = static_cast<double>(i ) / (N-1 ); // normalized grid point in (0, 1)
            vec[i] = std::sin(frequency * numbers::PI * x*x);
        }
    }
    template <int dim>
    class GenerateVector2D : public Function<dim, double>
    {
    public:
    GenerateVector2D()
        : Function<dim, double>(1)
    {}

    virtual double value(const Point<dim> &p,
                        const unsigned int /*component*/ = 0) const override
    {
        return std::sin(numbers::PI * p[0] * p[0]) *
            std::sin(numbers::PI * p[1] * p[1]);
    }
    };

    // explicit instantiation
    template class GenerateVector2D<2>;

    template<int dim>
    MatrixFunction<dim>::MatrixFunction()
    :
          dof_handler(triangulation)
        , fe(global::config.get<int>("Space_Discretization.fe_degree"))
        , tau(global::config.get<double>("Data.tau"))
        , expoint(fe,dof_handler,constraints,homogeneous_matrix
            ,damping_matrix,mass_matrix,tau,0,global::config.get<bool>("Output.info_ellipse") )
        , epsilon(pow(2,global::config.get<int>("Data.epsilon")))
        , tol_vec()
        , err_vec()
        , computation_time_mf_vec()
        , phi_k(global::config.get<int>("Data.phik"))
        , calc_eigen(eigenvalues,eigenvectors)
        , computing_timer(std::cout, TimerOutput::never, TimerOutput::wall_times)
        , genHomCoeff(1.,1.,true,0,1,1, dof_handler, fe, constraints, computing_timer) // first values just for compiling


    {
        while(!(currentPath.filename().string() == "build") && !currentPath.empty() )
        {
            currentPath = currentPath.parent_path();
        }
        currentPath = currentPath.parent_path();
    }

    template<int dim>
    double MatrixFunction<dim>::scalar_product(const Vector<double>& a,
                                                const Vector<double>& b,
                                                const SparseMatrix<double>& matrix,
                                                bool use_matrix ) 
    {
        Vector<double> tmp_vec(a.size());
        if(use_matrix){
            matrix.vmult(tmp_vec, a);
        }
        else{
            tmp_vec = a;
        }

        double result = 0;
        for (unsigned int i = 0; i < b.size(); ++i)
        {
            result += b[i] * tmp_vec[i];
        }

        return result;
    }

    template<int dim>
    std::string MatrixFunction<dim>::ellipse_method_string() const
    {
        switch (ellipse_method)
        {
            case enclosure: return "enclosure";
            case ritz:      return "ritz";
            default:        return "ritz";
        }
    }

    // Largest normalized radius of the reference eigenvalues w.r.t. the current
    // ellipse.  <= 1 means the ellipse covers the spectrum.
    // NOTE: uses the member 'eigenvalues', which error_calc_ell() fills via
    // calc_all_eig(ref_matrix_operator, .) -- call it AFTER error_calc_ell().
    // ref_matrix_operator is NOT scaled by tau, the ellipse is, hence the factor.
    template<int dim>
    double MatrixFunction<dim>::ellipse_coverage()
    {
        double max_rad = 0.;
        for (const auto& lambda : eigenvalues)
            max_rad = std::max(max_rad,
                               expoint.ellipse_opt.radius_p_alterative(tau * lambda));
        return max_rad;
    }

    template<int dim>
    std::string MatrixFunction<dim>::ellipse_cache_filename() const
    {
        const std::string example_id = global::config.get<std::string>("Calculation_Mode.example");
        std::string p = currentPath.string() + "/output/errors/Example" + example_id + "/ellipse_cache";
        p += "/h"    + global::config.get<std::string>("Space_Discretization.initial_refinement");
        p += "_deg"  + global::config.get<std::string>("Space_Discretization.fe_degree");
        p += "_tau"  + global::config.get<std::string>("Data.tau");
        p += "_beta" + beta_current;
        std::ostringstream ss;
        ss << "_tolell" << tol_ell;
        p += ss.str();
        p += "_ell" + ellipse_method_string();
        p += ".points";
        return p;
    }

    template<int dim>
    void MatrixFunction<dim>::save_ellipse_points(const std::string& filename) const
    {
        std::filesystem::create_directories(std::filesystem::path(filename).parent_path());

        std::ofstream out(filename);
        if (!out) {
            std::cout << "  [ellipse cache] Can not write: " << filename << std::endl;
            return;   
        }
        out << std::setprecision(17);
        out << tau << " " << tol_ell << " "
            << global::config.get<bool>("Calculation_Mode.strong_damping") << " "
            << ellipse_points.size() << "\n";
        for (const auto& z : ellipse_points)
            out << z.real() << " " << z.imag() << "\n";
        out.close();
        std::cout << "  [ellipse cache] Saved (" << ellipse_points.size()
                  << " points): " << filename << std::endl;
    }

    template<int dim>
    bool MatrixFunction<dim>::load_ellipse_points(const std::string& filename)
    {
        std::ifstream in(filename);
        if (!in)
            return false;

        double file_tau = 0., file_tol_ell = 0.;
        int file_strong_damping = 0;
        std::size_t count = 0;
        if (!(in >> file_tau >> file_tol_ell >> file_strong_damping >> count))
            return false;

        const bool strong_damping = global::config.get<bool>("Calculation_Mode.strong_damping");
        const double eps = 1e-12;
        if (std::abs(file_tau - tau) > eps * std::max(1.0, std::abs(tau))
            || std::abs(file_tol_ell - tol_ell) > eps * std::max(1.0, std::abs(tol_ell))
            || file_strong_damping != static_cast<int>(strong_damping))
        {
            std::cout << "  [ellipse cache] File does not match the configuration"
                         "(tau/tol_ell/strong_damping) - will be recalculated." << std::endl;
            return false;
        }
        if (count < 2) {
            std::cout << "  [ellipse cache] Not enough points saved - "
                         "will be recalculated." << std::endl;
            return false;
        }

        std::vector<std::complex<double>> pts;
        pts.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            double re = 0., im = 0.;
            if (!(in >> re >> im)) {
                std::cout << "  [ellipse cache] File is incomplete - "
                             "will be recalculated." << std::endl;
                return false;
            }
            pts.emplace_back(re, im);
        }

        ellipse_points = std::move(pts);
        return true;
    }

    template<int dim>
    void MatrixFunction<dim>::set_func_lambda()
    {
        if (phi_k == 0)
            func_lambda = [this](std::complex<double> x){ return func_matrix.expo(x); };
        else if (phi_k == 1)
            func_lambda = [this](std::complex<double> x){ return func_matrix.phi1(x); };
        else
            throw std::invalid_argument(
                "MatrixFunction::set_func_lambda:  phi_k = 0 or 1 is supported "
                "(Data.phik)");
    }

    template<int dim>
    void MatrixFunction<dim>::setup_system() {
        //
        const unsigned int initial_refinement = global::config.get<int>("Space_Discretization.initial_refinement");
        GridGenerator::hyper_cube(triangulation,0,1);
        triangulation.refine_global(initial_refinement);
        dof_handler.distribute_dofs(fe);
        IndexSet boundary_dofs(dof_handler.n_dofs());

        boundary_dofs = DoFTools::extract_boundary_dofs(dof_handler);
       
        constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        VectorTools::interpolate_boundary_values(
            dof_handler,               
            0,                         
            Functions::ZeroFunction<dim>(),        
            constraints);       
        VectorTools::interpolate_boundary_values(
            dof_handler,               
            1,                         
            Functions::ZeroFunction<dim>(),        
            constraints);       
        constraints.close();
        //
        size = dof_handler.n_dofs();
        DynamicSparsityPattern dsp(size);
        DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
        sparsity_pattern.copy_from(dsp);
        laplace_matrix.reinit(sparsity_pattern);
        homogeneous_matrix.reinit(sparsity_pattern);
        damping_matrix.reinit(sparsity_pattern);
        mass_matrix.reinit(sparsity_pattern);

        // For Matrix Function
        b_vec1.reinit(size);
        b_vec2.reinit(size);
        if (global::config.get<bool>("Calculation_Mode.random_vectors")){
            generateRandomVector(b_vec1,0.,1.);
            generateRandomVector(b_vec2,0.,1.);
        }
        else{
            if (dim == 1){
                generateDiscreteSine(b_vec1,1.0);
                generateDiscreteSine(b_vec2,1.0);
            }
            else if (dim == 2) {
                GenerateVector2D<dim> initial_function;
                VectorTools::interpolate(dof_handler, initial_function, b_vec1);
                VectorTools::interpolate(dof_handler, initial_function, b_vec2);
            }
            
        }

        ref_vec_in.reinit(2*size);
        for ( int i = 0; i < size; ++i){
            ref_vec_in[i] = b_vec1[i];
            ref_vec_in[i + size ] =  b_vec2[i];
        }

        ref_vec_out.reinit(2*size);
    }

    template<int dim>
    void MatrixFunction<dim>::setup_matrices(const double coef_damping){
        //
        MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(fe.degree + 1), mass_matrix);
        MatrixCreator::create_laplace_matrix(dof_handler, QGauss<dim>(fe.degree + 1), laplace_matrix);
        constraints.condense(laplace_matrix);
        constraints.condense(mass_matrix);
        coef_a = 0.5;
        homogeneous_matrix.copy_from(laplace_matrix);
        homogeneous_matrix *= coef_a;
        if(global::config.get<bool>("Calculation_Mode.strong_damping"))
            damping_matrix.copy_from(laplace_matrix);
        else
            genHomCoeff.assemble_damping_grad(damping_matrix);

        damping_matrix *= coef_damping;
        //
        if(global::config.get<bool>("Calculation_Mode.clc_ref_ellipse")){
            const IdentityMatrix id_matrix(size);
            identity_m = id_matrix; 
            ref_matrix_func.reinit(2*size,2*size);
            first_index_set = linspace(0,size,1);
            second_index_set = linspace(size,2*size,1);

            identity_m.scatter_matrix_to(first_index_set,second_index_set,ref_matrix_func);
            FullMatrix<double> mass_matrix_full;
            mass_matrix_full.reinit(size,size);
            mass_matrix_full.copy_from(mass_matrix);
            mass_matrix_full.gauss_jordan();

            FullMatrix<double> homogeneous_matrix_full(size,size);
            FullMatrix<double> damping_matrix_full(size,size);
            FullMatrix<double> M_inv_homogeneous_matrix_full(size,size);
            homogeneous_matrix_full.copy_from(homogeneous_matrix);
            damping_matrix_full.copy_from(damping_matrix);
            homogeneous_matrix_full *= -1.;
            damping_matrix_full *= -1.;
            //M_inv_homogeneous_matrix_full = homogeneous_matrix_full;
            mass_matrix_full.mmult(M_inv_homogeneous_matrix_full,homogeneous_matrix_full);
            M_inv_homogeneous_matrix_full.scatter_matrix_to(second_index_set,first_index_set,ref_matrix_func);
            
            FullMatrix<double> M_inv_damping_matrix_full(size,size);
            mass_matrix_full.mmult(M_inv_damping_matrix_full,damping_matrix_full);

            M_inv_damping_matrix_full.scatter_matrix_to(second_index_set,second_index_set,ref_matrix_func);

            ref_matrix_operator = ref_matrix_func;
            // for (int i = 0; i < homogeneous_matrix_full.m(); ++i){
            //     for (int j = 0; j < homogeneous_matrix_full.m(); ++j ){
            //         // std::cout << homogeneous_matrix_full(i,j) << " ";
            //     }
            //     std::cout << std::endl;
            // }
        }
    }

    template<int dim>   
    double MatrixFunction<dim>::error_calc_ell(double& ref_radiusx, double& ref_radiusy, double& ref_center, std::complex<double>& ref_eccentricity)
    {

        double err_radiusx,err_radiusy, err_center;
        std::complex<double> err_eccentricity;
        calc_eigen.calc_all_eig(ref_matrix_operator,2*size);   
        // double max_rad = 0.;
        double ref_radius_ellipse;
        expoint.ellipse_opt.get_radius_ellipse(ref_radius_ellipse);
        expoint.ellipse_opt.get_parameters(err_radiusx,err_radiusy,err_eccentricity,err_center);
        //
        // std::complex<double> bd_point;
        // if(err_eccentricity.imag() == 0){
        //     bd_point = std::complex<double>(err_radiusx + err_center);
        // }
        // else{
        //     bd_point = std::complex<double>(err_radiusy*1i + err_center);
        // }
        // for (int i = 0; i < eigenvalues.size(); ++i) {
        //     eigenvalues[i] *= tau;
        //     // std::cout << "Eigenvalue " << i << ": " << eigenvalues[i] << " Radius: " << expoint.ellipse_opt.radius_p(eigenvalues[i]) << "; " ;
        //     if (max_rad < expoint.ellipse_opt.radius_p(eigenvalues[i])* 1./expoint.ellipse_opt.radius_p(bd_point ) ){
        //         max_rad = expoint.ellipse_opt.radius_p(eigenvalues[i])* 1./expoint.ellipse_opt.radius_p(bd_point ) ;
        //     }
        // }
        // std::cout << "max_rad:" << max_rad << ", ell_rad" << ref_radius_ellipse << std::endl;
        // std::cout<< " xr " <<ref_radiusx << " " <<  err_radiusx<< std::endl;
        // std::cout<< " yr " <<ref_radiusy << " " <<  err_radiusy<< std::endl;
        // std::cout<< " cr " <<ref_center << " " <<  err_center<< std::endl;
        // std::cout<< " er " <<ref_eccentricity << " " <<  err_eccentricity<< std::endl;
        //
        err_radiusx     -= ref_radiusx;
        err_radiusy     -= ref_radiusy;
        err_center      -= ref_center;
        err_eccentricity -= ref_eccentricity;


        double err_total = std::pow(std::abs(err_radiusx),2) + std::pow(std::abs(err_radiusy),2) + std::pow(std::abs(err_center),2) + std::pow(std::abs(err_eccentricity),2);
        double scale     = std::pow(std::abs(ref_radiusx),2) + std::pow(std::abs(ref_radiusy),2) + std::pow(std::abs(ref_center),2) + std::pow(std::abs(ref_eccentricity),2);
        return std::sqrt(err_total/scale);
            
    }

    template<int dim>   
    double MatrixFunction<dim>::error_calc_matrix_function(const Vector<double>& ref_vec_long, Vector<double>& calc_vec1, Vector<double>& calc_vec2)
    {
        Vector<double> ref_vec1;
        Vector<double> ref_vec2;

        ref_vec1.reinit(size);
        ref_vec2.reinit(size);

        for (int i = 0; i < size; ++i){
            ref_vec1[i] = ref_vec_long[i];
            ref_vec2[i] = ref_vec_long[i+ size];
        }

        Vector<double> err_vec1 =  ref_vec1;
        err_vec1 -= calc_vec1;
        Vector<double> err_vec2 =  ref_vec2;
        err_vec2 -= calc_vec2;
        // for ( int i = 0; i < size; ++i){
        //     std::cout << "ref1 " << ref_vec1[i] << " calc1 " << calc_vec1[i] << std::endl;
        //     std::cout << "ref2 " << ref_vec2[i] << " calc2 " << calc_vec2[i] << std::endl;
        //     std::cout << "err1 " << err_vec1[i] <<std::endl;
        //     std::cout << "err2 " << err_vec2[i] <<std::endl;
        // }

        double err_total = std::sqrt(scalar_product(err_vec1, err_vec1, homogeneous_matrix) + scalar_product(err_vec2, err_vec2, mass_matrix));
        double scale =  1 ; //std::sqrt(std::pow(ref_vec1.l2_norm(),2) + std::pow(ref_vec2.l2_norm(),2));
        return err_total/scale;
            
    }

    template<int dim>   
    void MatrixFunction<dim>::create_ref_matrix_func()
    {
        // temp variable 
        std::cout << "Compute reference matrix function via full diagonalization..." << std::endl;
        set_func_lambda();          
        int two_size = 2*size; //2*size;
        const int lda = two_size;
        const int ldvl = two_size;
        const int ldvr = two_size;
        int info(0);
        std::vector<double> A_temp(two_size * two_size);        
        std::vector<double> wr(two_size), wi(two_size), vl(two_size*two_size), vr(two_size*two_size);        
        //
        eigenvectors_ref.reinit(two_size,two_size);
        eigenvectors_inverse_ref.reinit(two_size,two_size);
        eigenvalues.resize(two_size);
        for ( int i = 0; i < two_size; ++i) {
            for ( int j = 0; j < two_size; ++j) {
                A_temp[i * two_size + j] = ref_matrix_func(i, j); 
            }
        }

        info = LAPACKE_dgeev(LAPACK_ROW_MAJOR,'N', 'V',two_size, A_temp.data() ,lda,  wr.data(), wi.data(), vl.data(), ldvl ,vr.data() ,ldvr);
        Assert (info >=0, ExcInternalError());
        if (info != 0)
            std::cerr << "LAPACK error in dgeev" << std::endl;

        // std::cout << "2 " ;
        for ( int i = 0; i < two_size; ++i){
            eigenvalues[i] = std::complex<double>( wr[i], wi[i] );
        }

        int col = 0;
        while (col < two_size) {
            if (wi[col] == 0.0) {  // Real eigenvalue
                for (int j = 0; j < two_size; ++j) {
                    eigenvectors_ref(j, col) = {vr[j * two_size + col], 0.0}; // dgeev saves eigenvectors row-wise
                }
                col++;
            } else {  // Complex pair (col and col+1)
                for (int j = 0; j < two_size; ++j) {
                    // Eigenvector for wr[col] + i·wi[col]
                    eigenvectors_ref(j, col) = {
                        vr[j * two_size + col],
                        vr[j * two_size + (col + 1)]  // dgeev saves eigenvectors row-wise
                    };
                    // Eigenvector for wr[col] - i·wi[col] (conjugate)
                    eigenvectors_ref(j, col + 1) = {
                        vr[j * two_size + col ],
                        -vr[j * two_size + (col + 1)] // dgeev saves eigenvectors row-wise
                    };
                }
                col += 2;  // Skip the next column (already handled)
            }
        }

        eigenvectors_inverse_ref.copy_from(eigenvectors_ref);
        eigenvectors_inverse_ref.gauss_jordan();

        std::complex<double> sum;
        for (int i=0; i<two_size; ++i)   {
            for (int j=0; j<two_size; ++j){
                sum = 0.;
                for (int l=0; l<two_size; ++l){

                    sum +=  eigenvectors_ref(i,l)*func_lambda( tau*eigenvalues[l] )* eigenvectors_inverse_ref(l,j); 
                }
                ref_matrix_func(i,j) = sum.real();
            }
        }   

        for (int i=0; i<two_size; ++i)   {
            ref_vec_out[i] = 0; // needed if one uses more betas
            for (int j=0; j<two_size; ++j){
                ref_vec_out[i] += ref_matrix_func(i,j)*ref_vec_in[j];
            }
        }
    }

    template<int dim>   
    void MatrixFunction<dim>::create_ref_matrix_func_poly_krylov(){
        // Types::BlockMatrix<dim> blockmatrix_ref(homogeneous_matrix,damping_matrix,mass_matrix,dof_handler, tau, false);
        std::cout << "Compute reference matrix function via poly_krylov..." << std::endl;

        expoint.set_max_iteration_krylov(2*size);
        std::unique_ptr<Types::BlockMatrix<dim>> bmatrix_ref =  std::make_unique<Types::BlockMatrix<dim>>(homogeneous_matrix, damping_matrix, mass_matrix, dof_handler, tau,false,global::config.get<bool>("Calculation_Mode.strong_damping"));
        BMatrixVariant bmatrix_ref_vmult = [&bmatrix_ref](Vector<double>& a ,Vector<double> & b) { return bmatrix_ref->vmult(a,b,1e-10); };
        set_func_lambda();   
        Vector<double> clc_vec1 = b_vec1;
        Vector<double> clc_vec2 = b_vec2;
        double tol = 1e-10;
        expoint.matrix_func_poly_krylov( bmatrix_ref_vmult,func_lambda,clc_vec1,clc_vec2,tol);
        for ( int i = 0; i < size; ++i){
            ref_vec_out[i] = clc_vec1[i];
            ref_vec_out[i + size ] =  clc_vec2[i];
        }

    }

    template<int dim>   
    void MatrixFunction<dim>::create_ref_ellipse()
    {
        std::cout <<    "Compute reference ellipse..." << std::endl;
        std::vector<int> config = expoint.find_opt_ellipse_prototype(ref_matrix_operator);   
        std::cout << "Config" << config[0] << " " << config[1] << " " << config[2] << " ";
        expoint.ellipse_opt.get_parameters(ref_radiusx,ref_radiusy,ref_eccentricity,ref_center); 
        
    }


    template<int dim>
    void MatrixFunction<dim>::run( std::string beta )
    {
        degree                                   = 1;
        tol_vec                                  = {5e-1,1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5,5e-6,1e-6};
        tol_ell                                  = 1e-4;
        rat_shift                                = .05;
        ell_safety                               = global::config.get<double>("Calculation_Mode.ellipse_safety",    4.0);
        ell_eps_shape                            = global::config.get<double>("Calculation_Mode.ellipse_eps_shape", 0.0);
        ell_max_real                             = global::config.get<double>("Calculation_Mode.ellipse_max_real",
                                                        std::numeric_limits<double>::infinity());
        beta_current                             = beta;   

        // expoint.setup_system_macro_space();
        Types::BlockMatrix<dim> blockmatrix(homogeneous_matrix,damping_matrix,mass_matrix,dof_handler, tau, false,global::config.get<bool>("Calculation_Mode.strong_damping"));
        auto start = std::chrono::steady_clock::now();
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        std::string clc_method;

        switch (method)
        {
            case poly_krylov:
                clc_method = "poly_krylov";
                break;
            case rat_krylov:
                clc_method = "rat_krylov";
                break;
            case cheb:
                clc_method = "cheb";
                break;
            case leja:                     
                clc_method = "leja";
                break;
            default:
                break;
        }
        std::cout << std::endl;
        std::cout << "#############################################################" << std::endl;
        std::cout << "Start matrix function calculation using " << clc_method << " method..." << std::endl;
        std::cout << "#############################################################" << std::endl;
        std::cout << std::endl;

        switch (method){
            case cheb: case leja:           
            {
                // create_ref_ellipse();
                std::cout << "## Start ellipse calculation ("
                          << ellipse_method_string() << ")...##" << std::endl;
                std::cout << std::endl;
                start = std::chrono::steady_clock::now();

                ellipse_from_cache = false;
                if (ellipse_method == enclosure)
                {
                    expoint.set_spectral_bounds(FindEllipse<dim>::SpectralBounds::Eigen);
                    expoint.find_opt_ellipse_enclosure(ell_safety, ell_eps_shape, ell_max_real);
                    ellipse_points.clear();
                }
                else
                {
                    const std::string ell_cache = ellipse_cache_filename();
                    const bool reuse_ellipse =
                        global::config.get<bool>("Calculation_Mode.reuse_ellipse", true);

                    if (reuse_ellipse && load_ellipse_points(ell_cache))
                    {
                        expoint.ellipse_opt.create_opt_ellipse(ellipse_points);
                        ellipse_from_cache = true;
                        std::cout << "  [ellipse cache] Reused ("
                                  << ellipse_points.size() << " points): "
                                  << ell_cache << std::endl;
                    }
                    else
                    {
                        ellipse_points = expoint.find_opt_ellipse(tol_ell, &blockmatrix);
                        if (reuse_ellipse)
                            save_ellipse_points(ell_cache);
                    }
                }

                end = std::chrono::steady_clock::now();
                duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

                if (method == cheb)
                    degree = expoint.estimate_degree_phi_k(phi_k,1e-8,degree);
                computation_time_ell = duration.count();
                std::cout << "Time for calculate ellipse: " << computation_time_ell << " s."
                          << (ellipse_from_cache ? "  (from cache)" : "") << std::endl;
                {
                    double e_radiusx, e_radiusy, e_center, e_capacity;
                    std::complex<double> e_ecc;
                    expoint.ellipse_opt.get_parameters(e_radiusx,e_radiusy,e_ecc,e_center);
                    expoint.ellipse_opt.get_capacity_confocal(e_capacity);
                    const double e_extent = expoint.ellipse_opt.right_halfplane_extent();
                    std::cout << "  Ellipse [" << ellipse_method_string() << "]:"
                              << " radiusx = " << e_radiusx
                              << ", radiusy = " << e_radiusy
                              << ", center = "  << e_center
                              << ", foci = "    << e_ecc     << std::endl;
                    std::cout << "             capacity(gamma) = " << e_capacity
                              << ", rightmost point = " << e_extent;
                    if (e_extent > 0.)
                        std::cout << "  -> phi_k hump ~ exp(" << e_extent << ") = "
                                  << std::exp(std::min(e_extent, 700.));
                    std::cout << std::endl;
                }
                if (global::config.get<bool>("Calculation_Mode.clc_ref_ellipse")){
                    err_ell = this->error_calc_ell(ref_radiusx,ref_radiusy,ref_center,ref_eccentricity);
                    std::cout << "Relativ error in ellipse parameters: " << err_ell << std::endl;
                    const double coverage = this->ellipse_coverage();
                    std::cout << "Spectral coverage max_i radius_p(tau*lambda_i): " << coverage
                              << (coverage <= 1.0 ? "   (covered)" : "   (NOT covered!)")
                              << std::endl;
                    std::cout << std::endl;
                }

                set_func_lambda();
                start = std::chrono::steady_clock::now();
                if (method == cheb)
                    expoint.prepare_cheb(func_lambda);
                else
                    expoint.prepare_leja(phi_k);
                end = std::chrono::steady_clock::now();
                computation_time_prep =
                    std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
                std::cout << "Time for coefficient precomputation (excluded from timings): "
                          << computation_time_prep << " s." << std::endl;

                std::cout << "## Start calculating matrix function... ## " << std::endl;
                break;
            }
            case poly_krylov: case rat_krylov: 
                // std::cout << size << std::endl;
                expoint.set_max_iteration_krylov(2*size);
                break;
            default:
                std::cout << "choose poly_krylov, rat_krylov, cheb, leja" << std::endl;
                break;
        }

        bmatrix =  std::make_unique<Types::BlockMatrix<dim>>(homogeneous_matrix, damping_matrix, mass_matrix, dof_handler, tau,false,global::config.get<bool>("Calculation_Mode.strong_damping"),true);
        set_func_lambda();   
        Vector<double> clc_vec1 = b_vec1;
        Vector<double> clc_vec2 = b_vec2;
        err_vec.clear();
        computation_time_mf_vec.clear();
        mv_count_total_vec.clear();
        degree_used_vec.clear();
        
        for(auto& tol : tol_vec ){
            double tol_mult = 1e-2*tol;   
            if(method == cheb || method == leja){  
                bmatrix_vmult = [this,tol_mult](FullMatrix<std::complex<double>> & Ma) { return this->bmatrix->vmultBlock(Ma,tol_mult); };
            }else if (method == poly_krylov){
                bmatrix_vmult = [this,tol_mult](Vector<double>& a ,Vector<double> & b) { return this->bmatrix->vmult(a,b,tol_mult); };
            }else if (method == rat_krylov){
                bmatrix->set_rat_shift(rat_shift);
                bmatrix->set_factor(tau);
                bmatrix_vmult = [this,tol_mult](Vector<double>& a ,Vector<double> & b) { return this->bmatrix->vmult_shift(a,b,tol_mult); };
                bmatrix_vmult_inv = [this,tol_mult](Vector<double>& a ,Vector<double> & b) { return  this->bmatrix->vmult_rat(a,b,tol_mult); };
            }
            else{
                std::cerr << "choose poly_krylov, rat_krylov, cheb, leja" << std::endl;
            }
            std::cout << "Using Tolerance: " << tol << std::endl;

            start = std::chrono::steady_clock::now();
            bool converged = true;   
            switch (method){
                case cheb:
                expoint.matrix_func_cheb(bmatrix_vmult,func_lambda,clc_vec1,clc_vec2,tol,phi_k);
                break;
                case leja:               
                try {
                    expoint.matrix_func_leja(bmatrix_vmult,func_lambda,clc_vec1,clc_vec2,tol,phi_k);
                } catch (const std::runtime_error& e) {
                    converged = false;
                    std::cout << "  [leja] Tolerance not reached: "
                              << e.what() << std::endl;
                }
                break;
                case poly_krylov: 
                expoint.matrix_func_poly_krylov(bmatrix_vmult,func_lambda,clc_vec1,clc_vec2,tol);
                break;
                case rat_krylov: 
                expoint.matrix_func_rat_krylov(bmatrix_vmult,bmatrix_vmult_inv,func_lambda,clc_vec1,clc_vec2,tol,rat_shift);
                break;
                default:
                std::cout << "choose poly_krylov, rat_krylov, cheb, leja" << std::endl;
                break;
            }
            
            end = std::chrono::steady_clock::now();
            duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            
            err = error_calc_matrix_function(ref_vec_out,clc_vec1,clc_vec2);
            expoint.get_mv_count_per_time_step(mv_count_total);
            computation_time_mf = duration.count();

            // Polynomial degree that was actually used.  Only cheb and leja build a
            // polynomial; 0 marks "no degree available" (all other methods).
            // Without convergence both classes report the maximum degree reached.
            degree_used = 0;
            if (method == cheb)
                expoint.get_degree_cheb(degree_used);
            else if (method == leja)
                expoint.get_degree_leja(degree_used);

            std::cout << "  Error in matrix function: " << err
                      << (converged ? "" : "   (not converged)") << std::endl;
            std::cout << "  Matrix-vector multiplications: " << mv_count_total << std::endl;
            if (method == cheb || method == leja)
                std::cout << "  Polynomial degree: " << degree_used << std::endl;
            std::cout << "  Time for matrix function computation: " << computation_time_mf << " s."  << std::endl;
            
            computation_time_mf_vec.push_back(computation_time_mf);
            err_vec.push_back(err);
            mv_count_total_vec.push_back(mv_count_total);
            degree_used_vec.push_back(degree_used);
            clc_vec1 = b_vec1;
            clc_vec2 = b_vec2;
        }
        std::cout << std::endl;

        //
        // Store results
        //
        ConvergenceTable table_total{};
        for (unsigned long i= 0; i < tol_vec.size(); ++i){
            table_total.add_value("error",err_vec[i]);
            table_total.add_value("Matrix-vector multiplications",mv_count_total_vec[i]);
            table_total.add_value("Time in s",computation_time_mf_vec[i]);
            // 4th column only for the polynomial methods; the files of
            // poly_krylov / rat_krylov keep their previous three columns.
            if (method == cheb || method == leja)
                table_total.add_value("Polynomial degree",degree_used_vec[i]);
        }
        table_total.set_precision("error", 6);
        std::string example_id = global::config.get<std::string>("Calculation_Mode.example");

        const std::string base_path = currentPath.string() + "/output/errors/Example" + example_id + "/" + clc_method;
        std::string filename = base_path + "/h" + global::config.get<std::string>("Space_Discretization.initial_refinement");
        filename += "_deg" + global::config.get<std::string>("Space_Discretization.fe_degree");
        filename += "_tau" + global::config.get<std::string>("Data.tau");
        filename += "_beta" + beta;
        if(clc_method == "cheb" || clc_method == "leja")
            filename += "_ell" + ellipse_method_string();

        if(clc_method == "cheb" || clc_method == "leja"){  
            std::string filename_ell = filename;
            filename_ell += "_ell_info";
            if(global::config.get<bool>("Output.info_ellipse")){            
                expoint.store_info_ellipse(filename_ell);
            }
            else{
                std::ofstream file(filename_ell);
                double radiusx,radiusy,center;
                std::complex<double> eccentricity;
                expoint.ellipse_opt.get_parameters(radiusx,radiusy,eccentricity,center);
                file << radiusx << ";" << radiusy << ";" << eccentricity << ";" << center  << std::endl;
                file << computation_time_ell;
                file << ";" << computation_time_prep;  
                file.close();

            }
        }
        
        table_total.set_scientific("error", true);
        table_total.set_scientific("Matrix-vector multiplications", true);
        table_total.set_scientific("Time in s", true);
        std::ofstream out_file(filename);
        table_total.write_text(out_file);
        std::cout << "Path to file: " << filename << std::endl;
        std::cout << "Finish calculation for beta = " << beta << " with " << clc_method  << ". " << std::endl;
        std::cout << std::endl; 


    }


    template class MatrixFunction<1>;
    template class MatrixFunction<2>;
}