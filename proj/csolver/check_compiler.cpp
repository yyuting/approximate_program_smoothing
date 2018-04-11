
#include <omp.h>
#include "problems.h"
#include "compiler_problem.h"
#include <cnpy.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>

using namespace cv;

#define THREAD_PARALLEL         1                       /* Use thread parallel render */
#define LAPLACIAN_DENOISE       1

#define COEF_L1                 0.28209479177387814       
#define COEF_L2                 0.07957747154594767
#define COEF_L3                 0.02244839026564582   

Problem *create_compiler_problem(int ndims, int nsamples, CompilerProblem **compiler_problem) {
    CompilerProblem *cp = new CompilerProblem(ndims);
    if (compiler_problem) { *compiler_problem = cp; }
    Problem *ans = cp;
    if (nsamples) {
        ans = new SampledProblem(ans, nsamples);
    }
    return ans;
}

Problem *create_geometry_problem(int ndims, int nsamples, int nfeatures, const string geometry_name, int height, int width, double min_time, double max_time, int camera_path, CompilerProblem **compiler_problem) {
    CompilerProblem *cp = new CompilerProblem(nfeatures);
    if (compiler_problem) {
        *compiler_problem = cp;
    }
    Problem *shader = cp;
    Problem *ans;
    if (geometry_name == string("plane")) {
        ans = new PlaneShader(ndims, shader, height, width, min_time, max_time, camera_path);
    } else if (geometry_name == string("hyperboloid1")) {
        ans = new HyperbolicShader(ndims, shader, height, width, min_time, max_time, camera_path);
    } else if (geometry_name == string("sphere")) {
        ans = new SphereShader(ndims, shader, height, width, min_time, max_time, camera_path);
    } else {
        throw runtime_error("Could not create problem: unknown geometry name");
    }
    if (nsamples) {
        ans = new SampledProblem(ans, nsamples);
    }
    return ans;
}

#define CREATE_PROBLEM() \
        if (shader_only) { \
            problem = create_geometry_problem(ndims, nsamples, nfeatures, geometry_name, height, width, min_time, max_time, camera_path, &compiler_problem); \
        } else { \
            problem = create_compiler_problem(ndims, nsamples, &compiler_problem); \
        }

#define DELETE_PROBLEM() \
    delete compiler_problem; \
    delete problem;

void check_compiler(map<string, string> named_args, const vector<int> &ndimsL, int repeats, int check_g, int nerror, int nground, int seed, int nsamples, int print_rho, const vector<int> &render_viewport, const vector<double> &render_sigma, const vector<double> &render_t, int g_samples, const string &outdir, bool do_f, const string gname, int render_index, bool skip_save, bool shader_only, const string geometry_name, int nfeatures, int camera_path, double min_time, double max_time, bool spatially_varying, bool gradient, bool adjust_var, double timeout, const string &ground_truth_filename, bool denoise, int patch_size, int search_size, double denoise_param, const string &load_adjust_var, const string &grounddir, bool is_gt, bool allow_var_error) {
    string fname = "f";
    bool stop_on_nan = false;
    bool is_render = (render_viewport.size() >= 2) && render_viewport[0] > 0;
    if (g_samples <= 0) { g_samples = 1; }
    if (!named_args.count("ndims")) { fprintf(stderr, "--ndims argument is required for check_compiler\n"); exit(1); }
    printf("render_t: %s\n", to_string(render_t).c_str());
    int width = is_render ? render_viewport[0]: 480;
    int height = is_render ? render_viewport[1]: 320;

    init_gaussian_rng_samples(seed);
    
    int nsamples_error = (nerror >= 1) ? nerror: DEFAULT_ERROR_SAMPLES;
    int nsamples_ground = (nground >= 1) ? nground: DEFAULT_GROUND_SAMPLES;

    for (int ndims: ndimsL) {
        for (int repeat = 0; repeat < repeats; repeat++) {
            Problem *problem;
            CompilerProblem *compiler_problem;
            
            CREATE_PROBLEM();
            
            /*
            if (is_render) {
                if (ndims != 3) { fprintf(stderr, "error: rendering but dims != 3, exiting\n"); exit(1); }
                if (render_t.size() == 0) { fprintf(stderr, "error: rendering but no render_t, exiting\n"); exit(1); }
                problem->bound_lo[0] = 0.0; problem->bound_hi[0] = render_viewport[0]-1.0;
                problem->bound_lo[1] = 0.0; problem->bound_hi[1] = render_viewport[1]-1.0;
                problem->bound_lo[2] = render_t[0]; problem->bound_hi[2] = render_t[render_t.size()-1]+1e-8;
            }
            */
            if (print_rho) {
                problem->print_info();
            }
            
            if (repeat == 0) {

                double f_time, g_time;
                problem->time(f_time, g_time, timeout);
                printf("time_f: %e\n", f_time);
                double f_geom_time, g_geom_time;
                if (shader_only) {
                    problem->geometry_time(f_geom_time, g_geom_time, timeout);
                    printf("time_f_geom: %e\n", f_geom_time);
                }

                if (nsamples < 1000) {
                    if (!denoise) {
                        printf("time_g: %e\n", g_time);
                    }
                    if (shader_only) {
                        printf("time_g_geom: %e\n", g_geom_time);
                    }
                }

                VectorXd ground_truth_cache;
                int initial_ground_truth_size = ground_truth_cache.size();
                if (ground_truth_filename.size()) {
                    FILE *g_file = fopen(ground_truth_filename.c_str(), "rb");
                    if (g_file) {
                        fclose(g_file);
                        printf("loading ground truth samples\n");
                        cnpy::NpyArray arr = cnpy::npy_load(ground_truth_filename.c_str());
                        double *arr_data = reinterpret_cast<double *>(arr.data);
                        if (arr.shape.size() != 1) { fprintf(stderr, "wrong size for loaded ground truth samples\n"); }
                        ground_truth_cache.resize(arr.shape[0]);
                        for (int i = 0; i < ground_truth_cache.size(); i++) { ground_truth_cache[i] = arr_data[i]; }
                        initial_ground_truth_size = ground_truth_cache.size();
                    }
                }
                init_gaussian_rng_samples(seed);
                for (int jrepeat = 0; jrepeat < 1; jrepeat++) {
                    if (nerror >= -1) {
                        seed_all_rngs(seed);
                        double err_f = problem->error(nsamples_error, nsamples_ground, seed, 2.0, false, is_render ? &render_sigma: NULL, &ground_truth_cache);
                        printf("error_f: %e\n", err_f);
                        if (!denoise) {
                            seed_all_rngs(seed);
                            //printf("error: %d %d %d %d %d %d %p\n", nsamples_error, nsamples_ground, seed, int(is_render), int(spatially_varying), int(gradient), ground_truth_cache);
                            double err_g = problem->error(nsamples_error, nsamples_ground, seed, 2.0, true, is_render ? &render_sigma: NULL, &ground_truth_cache);
                            printf("error_g: %e\n", err_g);
                        }
                    }
                }
                if (ground_truth_cache.size() != initial_ground_truth_size && ground_truth_filename.size()) {
                    printf("saving ground truth samples\n");
                    const unsigned int shape_gt[] = {ground_truth_cache.size()};
                    cnpy::npy_save(ground_truth_filename.c_str(), &ground_truth_cache[0], shape_gt, 1, "w");
                }
            }
            
            if (is_render && (!skip_save || denoise)) {
                int channels = 3;
                bool is_color;
                if (problem->vec_output.size() == channels) {
                    is_color = true;
                } else {
                    is_color = false;
                }
                int is_f_max(1);
                if (skip_save) { is_f_max = 0; }
                for (int is_f = 0; is_f <= is_f_max; is_f++) {
                    if (!do_f && is_f) { continue; }
                    //Problem *problem = create_compiler_problem(ndims, nsamples);

                    VectorXd I(height * width * channels);
                    VectorXd sigma = problem->make_vector();
                    for (int i = 0; i < render_sigma.size(); i++) {
                        if (i < sigma.size()) {
                            sigma[i] = render_sigma[i];
                        }
                    }
                    
                    for (int t_index = 0; t_index < render_t.size(); t_index++) {

#if LAPLACIAN_DENOISE
                        Mat gau0, gau1, gau2;
                        Mat lap0, lap1, lap2;
                        Mat temp0, temp1, temp2;
                        Mat lap0_8u, lap1_8u, lap2_8u;
                        Mat gau0_8u, gau1_8u, gau2_8u;
                        int data_type, u_data_type;
                        Mat image, out;
                        Mat dimage;
                        if (is_color) {
                            data_type = CV_32FC3;
                            u_data_type = CV_8UC3;
                        } else {
                            data_type = CV_32FC1;
                            u_data_type = CV_8UC1;
                        }
                        
                        lap0 = Mat::zeros(height, width, data_type);
                        lap1 = Mat::zeros(height/2, width/2, data_type);
                        lap2 = Mat::zeros(height/4, width/4, data_type);
                        
                        lap0_8u = Mat::zeros(height, width, u_data_type);
                        lap1_8u = Mat::zeros(height/2, width/2, u_data_type);
                        lap2_8u = Mat::zeros(height/4, width/4, u_data_type);
                        
                        temp0 = Mat::zeros(height, width, data_type);
                        temp1 = Mat::zeros(height/2, width/2, data_type);
                        temp2 = Mat::zeros(height/4, width/4, data_type);
                        
                        gau0 = Mat::zeros(height/2, width/2, data_type);
                        gau1 = Mat::zeros(height/4, width/4, data_type);
                        gau2 = Mat::zeros(height/8, width/8, data_type);
                        
                        gau0_8u = Mat::zeros(height/2, width/2, u_data_type);
                        gau1_8u = Mat::zeros(height/4, width/4, u_data_type);
                        gau2_8u = Mat::zeros(height/8, width/8, u_data_type);
                        
                        out = Mat::zeros(height, width, CV_32FC3);
                        
                        dimage = Mat::zeros(height, width, data_type);
                        
                        if (denoise) {
                            setNumThreads(1);
                        }
#endif
                        for (int i = 0; i < I.size(); i++) {
                            I[i] = 0.0;
                        }
                        bool denoising_g = (denoise & !is_f && nsamples < 1000);
                        double T0_render = wall_time();
//                        printf("begin iteration %d, render_r=%f\n", t_index, render_t[t_index]);
//                        int I_index = 0;
                        if (denoising_g) {
                            omp_set_num_threads(1);
                        }
#if THREAD_PARALLEL
#pragma omp parallel private(problem, compiler_problem) shared(I)
{
                        init_gaussian_rng_samples(t_index);
                        if (omp_get_thread_num() == 0) {
                            printf("using %d threads\n", omp_get_num_threads());
                        }
                        CREATE_PROBLEM();
                        VectorXd p = problem->make_vector();
#pragma omp for
#else
                        init_gaussian_rng_samples(t_index);
                        VectorXd p = problem->make_vector();
#endif
                        for (int y = 0; y < height; y++) {
                            for (int x = 0; x < width; x++) {
                                bool stop_early = false;
                                int I_index = ((y * width) + x) * channels;
                                for (int g_sample = 0; g_sample < g_samples; g_sample++) {
    //                            printf("y=%d\n", y);
    //                                printf("x=%d\n", x); fflush(stdout);
                                    p[0] = x;
                                    p[1] = y;
                                    p[2] = render_t[t_index];
                                    double val = is_f ? problem->f(p): problem->g(p, sigma);
                                    if (problem->vec_output.size() == channels) {
                                        for (int c = 0; c < channels; c++) {
                                            val = problem->vec_output[c];
                                            if (stop_on_nan && std::isnan(val)) { fprintf(stderr, "nan at %d, %d, stopping\n", x, y); exit(1); }
                                            I[I_index+c] += val;
                                        }
                                    } else {
                                        for (int c = 0; c < channels; c++) {
                                            if (stop_on_nan && std::isnan(val)) { fprintf(stderr, "nan at %d, %d, stopping\n", x, y); exit(1); }
                                            I[I_index+c] += val;
                                        }
                                    }
                                    
                                    if (g_sample == 0 && !problem->probability_select_used) {
                                        stop_early = true;
                                        break;
                                    }
                                }
                                if (g_samples != 1 && !stop_early) {
                                    for (int c = 0; c < channels; c++) {
                                        I[I_index+c] /= g_samples;
                                    }
                                }
                            }
                        }
#if THREAD_PARALLEL
                        DELETE_PROBLEM();
}
#endif
                        //Mat image, out;
#if LAPLACIAN_DENOISE                        
                        
                        double T1_render = wall_time();
                        if (denoise && !is_f) {
                             for (int i = 0; i < I.size(); i++) {
                                if (std::isnan(I[i])){
                                    I[i] = 0.0;
                                } else if (I[i] > 1.0) {
                                    I[i] = 1.0;
                                } else if (I[i] < 0.0) {
                                    I[i] = 0.0;
                                }
                            }
                            image = Mat(height, width, CV_64FC3, I.data());
                            //image *= 256.0;
                            if (!is_color) {
                                image.convertTo(out, CV_32FC3, 256.0);
                                cvtColor(out, dimage, CV_RGB2GRAY);
                            } else {
                                image.convertTo(dimage, CV_32FC3, 256.0);
                            }
                            pyrDown(dimage, gau0);
                            pyrDown(gau0, gau1);
                            pyrDown(gau1, gau2);
                            
                            pyrUp(gau2, temp2);
                            lap2 = gau1 - temp2;
                            pyrUp(gau1, temp1);
                            lap1 = gau0 - temp1;
                            pyrUp(gau0, temp0);
                            lap0 = dimage - temp0;
                            
                            gau2.convertTo(gau2_8u, u_data_type);
                            
                            lap0.convertTo(lap0_8u, u_data_type, 0.5, 128.0);
                            lap1.convertTo(lap1_8u, u_data_type, 0.5, 128.0);
                            lap2.convertTo(lap2_8u, u_data_type, 0.5, 128.0);

                            fastNlMeansDenoising(gau2_8u, gau2_8u, 10.0, 5, 10);
                            fastNlMeansDenoising(lap2_8u, lap2_8u, 10.0, 5, 10);
                            fastNlMeansDenoising(lap1_8u, lap1_8u, 10.0, 5, 10);
                            fastNlMeansDenoising(lap0_8u, lap0_8u, denoise_param, 5, 10);
                            
                            lap2_8u.convertTo(lap2, data_type, 2.0, -256.0);
                            lap1_8u.convertTo(lap1, data_type, 2.0, -256.0);
                            lap0_8u.convertTo(lap0, data_type, 2.0, -256.0);
                            
                            gau2_8u.convertTo(gau2, data_type);
                            pyrUp(gau2, temp2);
                            gau1 = temp2 + lap2;
                            pyrUp(gau1, temp1);
                            gau0 = temp1 + lap1;
                            pyrUp(gau0, temp0);
                            dimage = temp0 + lap0;
                        }
#endif                        
#if !LAPLACIAN_DENOISE
                        if (denoising_g) {
                            image = Mat(height, width, CV_64FC3, I.data());
                            image.convertTo(image, CV_8UC3, 256);
                            if (is_color) {
                                cvtColor(image, image, CV_RGB2BGR);
                            }
                            fastNlMeansDenoising(image, out, denoise_param, patch_size, search_size);
                        }
#endif
                        double T2_render = wall_time();
                        printf("rendered in %f secs\n", T2_render-T0_render);
//                        printf("done loop, saving image\n"); fflush(stdout);
                        char filename[32768];
                        char lz4_filename[32768];
                        const char *sel_name = is_f ? fname.c_str(): gname.c_str();
                        sprintf(filename, is_f ? "%s/%s%05d": "%s/%s%05d", outdir.c_str(), sel_name, render_index+t_index);

                        if (!is_f && denoise && is_color && nsamples < 1000) {
                            cvtColor(dimage, dimage, CV_RGB2BGR);
                        }
                        
                        if (is_f || !denoise || nsamples >= 1000 || is_gt) {
                            save_image(I, string(filename), width, height, channels);
                        } else if (!skip_save) {
#if !LAPLACIAN_DENOISE
                            if (!is_color) {
                                cvtColor(out, out, CV_RGB2GRAY);
                            }
                            imwrite(string(filename) + string(".png"), out);
#else
                            //imwrite(string(filename) + string(".png"), image);
                            imwrite(string(filename) + string(".png"), dimage);
#endif
                        }
                        
                        if (!is_f && denoise && nsamples < 1000) {
                            
                            double g_time = (T2_render - T0_render) / (width * height);
                            double g_nlm_time = (T2_render - T1_render) / (width * height);

                            printf("time_g: %e\n", g_time);
                            printf("time_g_nlm: %e\n", g_nlm_time);
                            
                            if (nerror >= -1) {
                                char ground_filename[32768];
                                sprintf(ground_filename, "%s/%s%05d.png", grounddir.c_str(), string("ground").c_str(), render_index+t_index);
                                image = imread(ground_filename);
#if !LAPLACIAN_DENOISE
                                if (!is_color) {
                                    if (image.type() != 0) {
                                        cvtColor(image, image, CV_RGB2GRAY);
                                    }
                                    if (out.type() != 0) {
                                        cvtColor(out, out, CV_RGB2GRAY);
                                    }
                                }
                                if (is_color) {
                                    image.convertTo(image, CV_64FC3, 1.0 / 256.0);
                                    out.convertTo(out, CV_64FC3, 1.0 / 256.0);
                                } else {
                                    image.convertTo(image, CV_64FC1, 1.0 / 256.0);
                                    out.convertTo(out, CV_64FC1, 1.0 / 256.0);
                                }
                                out -= image;
                                double g_norm = norm(out, NORM_L2);
#else
                                if (!is_color) {
                                    if (image.channels() != 1) {
                                        cvtColor(image, image, CV_RGB2GRAY);
                                    }
                                }
                                image.convertTo(image, data_type);
                                //image -= dimage;
                                //image /= 256.0;
                                double g_norm = norm(image, dimage, NORM_L2);
                                g_norm /= 256.0;
#endif
                                double pixel_size = is_color ? I.size() : (I.size() / channels);
                                double sqrt_pixel = sqrt(pixel_size);
                                double g_error = g_norm / sqrt_pixel;
                                printf("error_g: %e\n", g_error);
                            }
                        }
#if SAVE_NPY
                        sprintf(filename, is_f ? "%s/%s%05d.npy": "%s/%s%05d.npy", outdir.c_str(), sel_name, render_index+t_index);
                        const unsigned int shape_I[] = {height, width, channels};
                        cnpy::npy_save(filename, &I[0], shape_I, 3, "w");

#endif
//                        printf("image filename: %s\n", filename); fflush(stdout);
//                        printf("done iteration\n");
                    }
                }
            }

            delete problem;
        }
        printf("Checking compiler_problem %d: ", ndims);
    }
}
