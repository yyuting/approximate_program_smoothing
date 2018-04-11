
#include <vector>
#include "util.h"
#include "problems.h"
#include "lodepng.h"

#define CLIP_SAMPLES                    0            /* Whether to clip samples to valid domain. */

/* Whether to precompute samples in SampledProblem */
#define PRECOMPUTE_SAMPLES 1

#define CHECK_F_G_EPSILON  5e-2                         /* How closely f, g should match at sigma=0. */

#define DEFAULT_LOG_RHO_TARGET_F_CALLS 10000            /* Default number of calls to f for log_rho */

using namespace std;

/* -----------------------------------------------------------------------------------
   Problem
   ----------------------------------------------------------------------------------- */

Problem::Problem() :probability_select_used(false) {
}

Problem::~Problem() {
}

int Problem::max_ndims() {
    return 1000*1000*1000;
}

VectorXd Problem::make_vector(double v) {
    return VectorXd::Constant(ndims, v);
}

/** A random sample of variables from the valid bounding box (bound_lo, bound_hi) */
void Problem::sample_vars(VectorXd &x) {
    for (int i = 0; i < ndims; i++) {
        x[i] = rand_uniform(bound_lo[i], bound_hi[i]);
    }
}

VectorXd Problem::sample_vars() {
    VectorXd ans(ndims);
    sample_vars(ans);
    return ans;
}

double time_func(function<double()> f, double min_time=0.5, int iteration_count=-1) {
    int times = 1;
    double ans = 0.0;
    double T = 0.0;
    while (true) {
        double start_time = wall_time();
        if (iteration_count > 0) { times = iteration_count; }
        for (int i = 0; i < times; i++) {
            ans += f();
        }
        T = wall_time() - start_time;
        if (iteration_count > 0) { break; }
        if (T >= min_time) { break; }
        times *= 2;
    }
    return T / times + (ans ? 1e-50: 0.0);
}

void Problem::time(double &f_time, double &g_time, double timeout) {
    f_time = 0.0;
    g_time = 0.0;
    auto p = sample_vars();
    VectorXd grad(make_vector());
    auto w = make_vector(1e-8);
    w[2] = 0.0;
    
    f_time = time_func([&]() { return f(p); }, timeout);
    g_time = time_func([&]() { return g(p, w); }, timeout);
}

double Problem::error(int nsamples_error, int nsamples_ground, int seed, double pnorm, bool is_g, const vector<double> *render_sigma, VectorXd *ground_truth_cache) {
    auto p_center = make_vector();
    auto p = make_vector();
    VectorXd range(bound_hi - bound_lo);

    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);
        
    double ans = 0.0;
    double T0 = wall_time();
    MatrixXd samples(nsamples_ground, ndims);
    for (int j = 0; j < nsamples_ground; j++) {
        for (int k = 0; k < ndims; k++) {
            samples(j, k) = distribution(generator);
        }
    }
    int channels;
    
    if (vec_output.size() == 0) {
        channels = 1;
    } else {
        channels = vec_output.size();
    }

    bool cache_init = ground_truth_cache ? (ground_truth_cache->size() == nsamples_error*channels): false;

    if (ground_truth_cache && ground_truth_cache->size() && !cache_init) {
        printf("warning: ground truth cache is wrong size, resizing\n");
    }
    if (!cache_init && ground_truth_cache) {
        ground_truth_cache->resize(nsamples_error*channels);
    }
    int cache_index = 0;
    double T1 = wall_time();
    double weight_sum = 0.0;
    for (int i = 0; i < nsamples_error; i++) {
        sample_vars(p_center);
        VectorXd w;
        if (render_sigma) {
            w = this->make_vector();
            for (int i = 0; i < render_sigma->size(); i++) {
                w[i] = (*render_sigma)[i];
            }
        } else {
            w = range * (MAX_SIGMA_RATIO * randf());
        }
        
        VectorXd g_ground(VectorXd::Constant(channels, 0.0));
        if (ground_truth_cache && cache_init) {
            for (int c = 0; c < channels; c++) {
                g_ground[c] = (*ground_truth_cache)[cache_index++];
            }
        } else {
            for (int c = 0; c < channels; c++) {
                g_ground[c] = 0.0;
            }
            for (int j = 0; j < nsamples_ground; j++) {
                for (int k = 0; k < ndims; k++) {
                    p[k] = p_center[k] + samples(j, k) * w[k];
                    if (CLIP_SAMPLES || render_sigma) {
                        if (p[k] < bound_lo[k]) { p[k] = bound_lo[k]; }
                        else if (p[k] > bound_hi[k]) { p[k] = bound_hi[k]; }
                    }
                }
                double val = f(p);
                accum_output(g_ground, val, channels);
            }
            g_ground /= nsamples_ground;
            if (ground_truth_cache) {
                for (int c = 0; c < channels; c++) {
                    (*ground_truth_cache)[cache_index++] = g_ground[c];
                }
            }
        }
        VectorXd g_est(VectorXd::Constant(channels, 0.0));
        for (int c = 0; c < channels; c++) {
            g_est[c] = 0.0;
        }
        double g_val = is_g ? g(p_center, w): f(p_center);
        accum_output(g_est, g_val, channels);
        
        if (render_sigma) {
            for (int c = 0; c < channels; c++) {
                if (g_est[c] < 0) { g_est[c] = 0.0; }
                if (g_est[c] > 1) { g_est[c] = 1.0; }
                if (std::isnan(g_est[c])) { g_est[c] = 0.0; }
            }
        }
        double weight = 1.0;
        double g_error;
        double f_val = f(p_center);
        for (int c = 0; c < channels; c++) {
            if (render_sigma) {
                if (g_ground[c] < 0) { g_ground[c] = 0.0; }
                if (g_ground[c] > 1) { g_ground[c] = 1.0; }
                if (std::isnan(g_ground[c])) { g_ground[c] = 0.0; }
            }
            g_error = abs(g_ground[c] - g_est[c]);
            ans += weight * pow(g_error, pnorm);
            weight_sum += weight;
        }
    }
    ans /= weight_sum;
    
    return pow(ans, 1.0/pnorm);
}

void Problem::accum_output(VectorXd &out, double val, int nchannel) {
    if (vec_output.size() == nchannel) {
        for (int c = 0; c < nchannel; c++) {
            out[c] += vec_output[c];
        }
    } else {
        for (int c = 0; c < nchannel; c++) {
            out[c] += val;
        }
    }
}

void Problem::print_info() {
}

void Problem::geometry_time(double &f_time, double &g_time, double timeout) {
}

/* -----------------------------------------------------------------------------------
   ProblemLogRho
   ----------------------------------------------------------------------------------- */

ProblemLogRho::ProblemLogRho() {
    n_samples = 0;
    target_f_calls = DEFAULT_LOG_RHO_TARGET_F_CALLS;
}

void ProblemLogRho::log_rho_reserve(int n) {
    n_samples = 0;
    a_accum.resize(n);
    b_accum.resize(n);
    a2_accum.resize(n);
    b2_accum.resize(n);
    ab_accum.resize(n);
}

void ProblemLogRho::log_rho_reset() {
    a_accum = VectorXd::Constant(a_accum.size(), 0.0);
    b_accum = VectorXd::Constant(b_accum.size(), 0.0);
    a2_accum = VectorXd::Constant(a2_accum.size(), 0.0);
    b2_accum = VectorXd::Constant(b2_accum.size(), 0.0);
    ab_accum = VectorXd::Constant(ab_accum.size(), 0.0);
}

void ProblemLogRho::log_rho(int index, double a, double b) {
    a_accum[index] += a;
    b_accum[index] += b;
    a2_accum[index] += a*a;
    b2_accum[index] += b*b;
    ab_accum[index] += a*b;
    n_samples++;
}

double ProblemLogRho::log_rho_plus(int index, double a, double b) {
    log_rho(index, a, b);
    return a + b;
}

double ProblemLogRho::log_rho_minus(int index, double a, double b) {
    log_rho(index, a, b);
    return a - b;
}

double ProblemLogRho::log_rho_times(int index, double a, double b) {
    log_rho(index, a, b);
    return a * b;
}

double ProblemLogRho::log_rho_divide(int index, double a, double b) {
    log_rho(index, a, b);
    return a / b;
}

void ProblemLogRho::print_info() {
    VectorXd x = make_vector();
    log_rho_reset();
    for (int i = 0; i < target_f_calls; i++) {
        sample_vars(x);
        f(x);
    }
    
    for (int i = 0; i < a_accum.size(); i++) {
        int n = n_samples;
        double a_denom = sqrt(max(n * a2_accum[i] - a_accum[i]*a_accum[i], 0.0));
        double b_denom = sqrt(max(n * b2_accum[i] - b_accum[i]*b_accum[i], 0.0));
        double r = 0.0;
        if (a_denom && b_denom) {
            r = (n * ab_accum[i] - a_accum[i] * b_accum[i]) / (a_denom * b_denom);
        }
        if (std::isnan(r)) {
            r = 0.0;
        }
        printf("log_rho_%d: %f\n", i, r);
//        printf("log_rho_%d: %f (%f, %f, %f, %f, %f, %d)\n", i, r, a_accum[i], b_accum[i], a2_accum[i], b2_accum[i], ab_accum[i], n_samples);
    }
}

/* -----------------------------------------------------------------------------------
   SampledProblem
   ----------------------------------------------------------------------------------- */

SampledProblem::SampledProblem(Problem *problem_, int nsamples_) {
    problem = problem_;
    nsamples = nsamples_;
    ndims = problem->ndims;
    bound_lo = problem->bound_lo;
    bound_hi = problem->bound_hi;
    if (problem->vec_output.size() != 0) {
        vec_output.resize(problem->vec_output.size());
    }
}

double SampledProblem::f(const VectorXd &x) {
    //return problem->f(x);
    double ans = problem->f(x);
    if (problem->vec_output.size() == 0) {
        return ans;
    } else {
        for (int i = 0; i < vec_output.size(); i++) {
            vec_output[i] = problem->vec_output[i];
        }
    }
    return 0.0;
}

void SampledProblem::draw_sample(VectorXd &x_sample, int sample_index, const VectorXd &x, const VectorXd &w) {
#if PRECOMPUTE_SAMPLES
    /*
    static MatrixXd sample_arr(problem->ndims, nsamples);
    static bool sample_arr_init = false;
    if (!sample_arr_init) {
        sample_arr_init = true;
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, 1.0);
        for (int i = 0; i < problem->ndims; i++) {
            for (int j = 0; j < nsamples; j++) {
                double num = distribution(generator);
                sample_arr(i, j) = num;
            }
        }
    }
    */
    
    //auto sample_array_col = sample_arr.col(sample_index);
    for (int i = 0; i < x.size(); i++) {
        x_sample[i] = Gaussian_RNG::get(i) * w[i] + x[i];
    }
    Gaussian_RNG::advance(x.size());
#endif
    //VectorXd sample_array_col2(sample_arr.col(sample_index));
    //printf("sample_array before return: %s\n", to_string(sample_array_col2).c_str());
    //printf("x before return: %s\n", to_string(x).c_str());
    //printf("w before return: %s\n", to_string(w).c_str());
    //printf("x_sample before return: %s\n", to_string(x_sample).c_str());
    
#if(!PRECOMPUTE_SAMPLES)
    static std::default_random_engine generator;
    static std::normal_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < problem->ndims; i++) {
        x_sample[i] = distribution(generator) * w[i] + x[i];
    }
#endif

#if CLIP_SAMPLES
    for (int i = 0; i < problem->ndims; i++) {
        if (x_sample[i] < bound_lo[i]) {
            x_sample[i] = bound_lo[i];
        } else if (x_sample[i] > bound_hi[i]) {
            x_sample[i] = bound_hi[i];
        }
    }
#endif
}

double SampledProblem::g(const VectorXd &x, const VectorXd &w) {
    VectorXd x_sample(ndims);
    double ans = 0.0;
    
    if (problem->vec_output.size() == 0) {
        for (int j = 0; j < nsamples; j++) {
            draw_sample(x_sample, j, x, w);
            //printf("x: %s, j: %d, w: %s, x_sample: %s\n", to_string(x).c_str(), j, to_string(w).c_str(), to_string(x_sample).c_str());
            ans += problem->f(x_sample);
        }
        ans /= nsamples;
    } else {
        vec_output.resize(problem->vec_output.size());
        for (int i = 0; i < vec_output.size(); i++) {
            vec_output[i] = 0.0;
        }
        for (int j = 0; j < nsamples; j++) {
            draw_sample(x_sample, j, x, w);
            //printf("x: %s, j: %d, w: %s, x_sample: %s\n", to_string(x).c_str(), j, to_string(w).c_str(), to_string(x_sample).c_str());
            problem->f(x_sample);
            vec_output += problem->vec_output;
        }
        vec_output /= nsamples;
    }
    return ans;
}

void SampledProblem::sample_vars(VectorXd &x) {
    problem->sample_vars(x);
}

VectorXd SampledProblem::sample_vars() {
    return problem->sample_vars();
}

void SampledProblem::geometry_time(double &f_time, double &g_time, double timeout) {
    f_time = 0.0;
    g_time = 0.0;
    auto p = sample_vars();
    auto w = make_vector(1e-8);
    w[2] = 0.0;
    
    problem->geometry_only = true;
    
    f_time = time_func([&]() { return f(p); }, timeout);
    g_time = time_func([&]() { return g(p, w); }, timeout);
    
    problem->geometry_only = false;
}

/* ShaderProblem */
ShaderProblem::ShaderProblem(int ndims_, Problem *shader_, int height, int width, double min_time, double max_time, int path) {
    shader = shader_;
    ndims = ndims_;
    camera_path = path;
    assert(shader->ndims <= NUM_FEATURES);
    if (ndims != 3) { fprintf(stderr, "Error: ShaderProblem should be 3D\n"); exit(1); }
    bound_lo = VectorXd::Constant(ndims, 0.000000);
    bound_hi = VectorXd::Constant(ndims, 480.000000);
    bound_hi[0] = width;
    bound_hi[1] = height;
    bound_lo[2] = min_time;
    bound_hi[2] = max_time;
    vec_output.resize(shader->vec_output.size());
    local_features.resize(NUM_FEATURES);
    local_features_sigma.resize(NUM_FEATURES);
    geometry_only = false;
}

void ShaderProblem::get_features_sigma(const VectorXd &x, const Vector22d &features, const VectorXd &w, Vector22d &features_sigma) {
    static VectorXd x_temp;
    if (x_temp.size() != x.size()) {
        x_temp.resize(x.size());
    }
    for (int i = 0; i < ndims; i++) {
        x_temp[i] = x[i];
    }
    double h = 1e-8;
    
    Vector22d features2;
    Vector22d features_variance;
    
    for (int i = 0; i < shader->ndims; i++) {
        features_variance[i] = 0.0;
    }
    
#if !PYTHON_GEOMETRY_OPT
    for (int i = 0; i < shader->ndims; i++) {
        features_sigma[i] = 0.0;
    }
#endif

#if PASS_IN_CONSTANT_FEATURES
    for (int j = 0; j < ndims; j++) {
        if (w[j] > 0) {
            double x_orig = x_temp[j];
            x_temp[j] += h;
            get_features(x_temp, features2);
            x_temp[j] = x_orig;
            for (int i = 0; i < nonconstant_indices; i++) {
                features_variance[i] += square((features2[i] - features[i]) / h) * w[j] * w[j];
            }
        }
    }
    for (int i = 0; i < nonconstant_indices; i++) {
        features_sigma[i] = sqrt(features_variance[i]);
        //printf("%f ", features_sigma[i]);
    }
#else
    for (int j = 0; j < ndims; j++) {
        if (w[j] > 0) {
            double x_orig = x_temp[j];
            x_temp[j] += h;
            get_features(x_temp, features2);
            x_temp[j] = x_orig;
            for (int i = 0; i < shader->ndims; i++) {
                features_variance[i] += square((features2[i] - features[i]) / h) * w[j] * w[j];
            }
        }
    } 
    //printf("features_sigma:");
    for (int i = 0; i < shader->ndims; i++) {
        features_sigma[i] = sqrt(features_variance[i]);
        //printf("%f ", features_sigma[i]);
    }
#endif
    //printf("\n");
}

void ShaderProblem::get_feature_and_sigma(const VectorXd &x, Vector22d &features, const VectorXd &w, Vector22d &features_sigma) {
    get_features(x, features);
    get_features_sigma(x, features, w, features_sigma);
}

void ShaderProblem::geometry_time(double &f_time, double &g_time, double timeout) {
    f_time = 0.0;
    g_time = 0.0;
    auto p = sample_vars();
    auto w = make_vector(1e-8);
    w[2] = 0.0;
    
    geometry_only = true;
    
    f_time = time_func([&]() { return f(p); }, timeout);
    g_time = time_func([&]() { return g(p, w); }, timeout);
    
    geometry_only = false;
    
    //Vector22d features;
    //Vector22d features_sigma;
    
    //f_time = time_func([&]() { get_features(p, features);
    //                           return 0.0; }, 
    //                   timeout);
    //g_time = time_func([&]() { get_feature_and_sigma(p, features, w, features_sigma);
    //                           return 0.0; }, 
    //                   timeout);
}

double ShaderProblem::f(const VectorXd &x) {
    Vector22d features;
    get_features(x, features);
    if (geometry_only) {
        for (int i = 0; i < vec_output.size(); i++) {
            vec_output[i] = 0.0;
        }
        return 0.0;
    }
    double ans = shader->f(features);
    for (int i = 0; i < vec_output.size(); i++) {
        vec_output[i] = shader->vec_output[i];
    }
    return ans;
}

thread_local VectorXd ShaderProblem::local_features;
thread_local VectorXd ShaderProblem::local_features_sigma;

double ShaderProblem::g(const VectorXd &x, const VectorXd &w) {
    //double T0 = wall_time();

    Vector22d features;
    Vector22d features_sigma;
    //double T_mid = wall_time();
    //double T1 = wall_time();
    //T_vars += T1 - T0;
    //get_features(x, features);
    //get_features_sigma(x, features, w, features_sigma);
    get_feature_and_sigma(x, features, w, features_sigma);
    //double T2 = wall_time();
    for (int i = 0; i < shader->ndims; i++) {
        local_features[i] = features[i];
    }
    for (int i = 0; i < shader->ndims; i++) {
        local_features_sigma[i] = features_sigma[i];
    }
    
    if (geometry_only) {
        for (int i = 0; i < vec_output.size(); i++) {
            vec_output[i] = 0.0;
        }
        return 0.0;
    }
    
    double ans = shader->g(local_features, local_features_sigma);
    for (int i = 0; i < vec_output.size(); i++) {
        vec_output[i] = shader->vec_output[i];
    }
    //double T3 = wall_time();
    //double T4 = wall_time();
    
    //T_call += T4 - T2;
    //T_g += T3 - T0;
    return ans;
}

/* PlaneShader */
void PlaneShader::get_features(const VectorXd &x, Vector22d &features) {
    Vector3d ray_dir(x[0] - bound_hi[0] / 2.0, x[1] + 1.0, bound_hi[0] / 2.0);
    
    Vector3d ray_origin(0.0, 0.0, 50.0);
    
    double theta;
    double camera_radius = 300.0;
    
    double cos_theta, sin_theta;
    if (camera_path == 1) {
        theta = x[2];
        cos_theta = cos(theta);
        sin_theta = sin(theta);
    } else if (camera_path == 2) {
        theta = -x[2];
        cos_theta = cos(theta);
        sin_theta = sin(theta);
        ray_origin[0] = camera_radius * cos_theta;
        ray_origin[1] = -camera_radius * sin_theta;
        ray_origin[2] = 50.0 + x[2] * 50.0;
    } else if (camera_path == 3) {
        theta = 0.0;
        cos_theta = 1.0;
        sin_theta = 0.0;
    } else {
        throw runtime_error("Could not create camera path: unknown camera path number");
    }
    
    Vector3d ray_dir_p(-(ray_dir[0] * cos_theta - ray_dir[2] * sin_theta), -(ray_dir[0] * sin_theta + ray_dir[2] * cos_theta), -ray_dir[1]);
    ray_dir_p.normalize();
    
    Vector3d N(0.0, 0.0, 1.0);
    
    //Vector3d light_dir(0.3, 0.8, 1.0);
    //light_dir.normalize();
    Vector3d light_dir(0.22808577638091165, 0.60822873701576452, 0.76028592126970562);
    
    double t_ray = -ray_origin[2] / ray_dir_p[2]; //-N.dot(ray_origin) / N.dot(ray_dir_p);
    
    Vector3d intersect_pos;
    intersect_pos = ray_origin + t_ray * ray_dir_p;
    
    features[FEATURE_TIME] = x[2];
    features[FEATURE_POSITION_X] = intersect_pos[0];
    features[FEATURE_POSITION_Y] = intersect_pos[1];
    features[FEATURE_POSITION_Z] = intersect_pos[2];
    features[FEATURE_RAY_DIR_X] = -ray_dir_p[0];
    features[FEATURE_RAY_DIR_Y] = -ray_dir_p[1];
    features[FEATURE_RAY_DIR_Z] = -ray_dir_p[2];
    
#if !PYTHON_GEOMETRY_OPT
    features[FEATURE_TEXTURE_S] = intersect_pos[0];
    features[FEATURE_TEXTURE_T] = intersect_pos[1];
    features[FEATURE_NORMAL_X] = N[0];
    features[FEATURE_NORMAL_Y] = N[1];
    features[FEATURE_NORMAL_Z] = N[2];
    
    //fill_features_light_dir_time(features, light_dir, -ray_dir_p, x[2]);
    
    features[FEATURE_TANGENT_X] = 1.0;
    features[FEATURE_TANGENT_Y] = 0.0;
    features[FEATURE_TANGENT_Z] = 0.0;
    features[FEATURE_BINORMAL_X] = 0.0;
    features[FEATURE_BINORMAL_Y] = 1.0;
    features[FEATURE_BINORMAL_Z] = 0.0;
    features[FEATURE_IS_VALID] = 1.0;
    features[FEATURE_LIGHT_DIR_X] = light_dir[0];
    features[FEATURE_LIGHT_DIR_Y] = light_dir[1];
    features[FEATURE_LIGHT_DIR_Z] = light_dir[2];
#endif
}

void PlaneShader::get_feature_and_sigma(const VectorXd &x, Vector22d &features, const VectorXd &w, Vector22d &features_sigma) {
    //double T0 = wall_time();
    Vector3d ray_dir(x[0] - bound_hi[0] / 2.0, x[1] + 1.0, bound_hi[0] / 2.0);
    Vector3d ray_origin(0.0, 0.0, 50.0);
    
    double theta;
    double camera_radius = 300.0;
    
    double cos_theta, sin_theta;
    
    if (camera_path == 1) {
        theta = x[2];
        cos_theta = cos(theta);
        sin_theta = sin(theta);
    } else if (camera_path == 2) {
        theta = -x[2];
        cos_theta = cos(theta);
        sin_theta = sin(theta);
        ray_origin[0] = camera_radius * cos_theta;
        ray_origin[1] = -camera_radius * sin_theta;
        ray_origin[2] = 50.0 + x[2] * 50.0;
    } else if (camera_path == 3) {
        theta = 0.0;
        cos_theta = 1.0;
        sin_theta = 0.0;
    } else {
        throw runtime_error("Could not create camera path: unknown camera path number");
    }
    
    Vector3d ray_dir_p_unnorm(-(ray_dir[0] * cos_theta - ray_dir[2] * sin_theta), -(ray_dir[0] * sin_theta + ray_dir[2] * cos_theta), -ray_dir[1]);
    Vector3d ray_dir_p(ray_dir_p_unnorm);
    ray_dir_p.normalize();
    
    Vector3d N(0.0, 0.0, 1.0);
    
    //Vector3d light_dir(0.3, 0.8, 1.0);
    //light_dir.normalize();
    Vector3d light_dir(0.22808577638091165, 0.60822873701576452, 0.76028592126970562);
    
    double t_ray = -ray_origin[2] / ray_dir_p[2]; //-N.dot(ray_origin) / N.dot(ray_dir_p);
    
    Vector3d intersect_pos;
    intersect_pos = ray_origin + t_ray * ray_dir_p;
    
    features[FEATURE_TIME] = x[2];
    features[FEATURE_POSITION_X] = intersect_pos[0];
    features[FEATURE_POSITION_Y] = intersect_pos[1];
    features[FEATURE_POSITION_Z] = intersect_pos[2];
    features[FEATURE_RAY_DIR_X] = -ray_dir_p[0];
    features[FEATURE_RAY_DIR_Y] = -ray_dir_p[1];
    features[FEATURE_RAY_DIR_Z] = -ray_dir_p[2];

    features_sigma[FEATURE_TIME] = w[2];
    Vector22d features_variance;
    for (int i = 1; i < shader->ndims; i++) {
        features_variance[i] = 0.0;
    }
    
    double h = 1e-8;
    double ray_dir_cache;
    
    // compute values for x+h
    if (w[0] > 0) {
        ray_dir_cache = ray_dir[0];
        ray_dir[0] = x[0] + h - bound_hi[0] / 2.0;
        ray_dir_p[0] = -(ray_dir[0] * cos_theta - ray_dir[2] * sin_theta);
        ray_dir_p[1] = -(ray_dir[0] * sin_theta + ray_dir[2] * cos_theta);
        ray_dir_p[2] = ray_dir_p_unnorm[2];
        ray_dir_p.normalize();
        t_ray = -ray_origin[2] / ray_dir_p[2];
        intersect_pos = ray_origin + t_ray * ray_dir_p;
        accum_features_variance(features, intersect_pos, ray_dir_p, features_variance, w[0]*w[0], h);
        ray_dir[0] = ray_dir_cache;
    }
    // compute values for y+h
    if (w[1] > 0) {
        ray_dir_cache = ray_dir[1];
        ray_dir[1] = x[1] + h + 1.0;
        ray_dir_p[0] = ray_dir_p_unnorm[0];
        ray_dir_p[1] = ray_dir_p_unnorm[1];
        ray_dir_p[2] = -ray_dir[1];
        ray_dir_p.normalize();
        t_ray = -ray_origin[2] / ray_dir_p[2];
        intersect_pos = ray_origin + t_ray * ray_dir_p;
        accum_features_variance(features, intersect_pos, ray_dir_p, features_variance, w[1]*w[1], h);
        ray_dir[1] = ray_dir_cache;
    }
    // compute values for t+h
    if (w[2] > 0 && (camera_path == 1 || camera_path == 2)) {
        if (camera_path == 1) {
            theta = x[2] + h;
            cos_theta = cos(theta);
            sin_theta = sin(theta);
        } else {
            theta = -(x[2] + h);
            cos_theta = cos(theta);
            sin_theta = sin(theta);
            ray_origin[0] = camera_radius * cos_theta;
            ray_origin[1] = -camera_radius * sin_theta;
            ray_origin[2] = 50.0 + (x[2] + h) * 50.0;
        }
        ray_dir_p[0] = -(ray_dir[0] * cos_theta - ray_dir[2] * sin_theta);
        ray_dir_p[1] = -(ray_dir[0] * sin_theta + ray_dir[2] * cos_theta);
        ray_dir_p[2] = ray_dir_p_unnorm[2];
        ray_dir_p.normalize();
        t_ray = -ray_origin[2] / ray_dir_p[2];
        intersect_pos = ray_origin + t_ray * ray_dir_p;
        accum_features_variance(features, intersect_pos, ray_dir_p, features_variance, w[2]*w[2], h);
    }
    for (int i = 1; i < shader->ndims; i++) {
        features_sigma[i] = sqrt(features_variance[i]);
    }
    //double T_final = wall_time();
    //T_feature_and_sigma += T_final - T0;
}

inline void PlaneShader::accum_features_variance(const Vector22d &features, const Vector3d &intersect_pos, const Vector3d &ray_dir_p, Vector22d &features_variance, double var, double h) {
    features_variance[FEATURE_POSITION_X] += square((intersect_pos[0] - features[FEATURE_POSITION_X]) / h) * var;
    features_variance[FEATURE_POSITION_Y] += square((intersect_pos[1] - features[FEATURE_POSITION_Y]) / h) * var;
    features_variance[FEATURE_POSITION_Z] += square((intersect_pos[2] - features[FEATURE_POSITION_Z]) / h) * var;
    features_variance[FEATURE_RAY_DIR_X] += square((-ray_dir_p[0] - features[FEATURE_RAY_DIR_X]) / h) * var;
    features_variance[FEATURE_RAY_DIR_Y] += square((-ray_dir_p[1] - features[FEATURE_RAY_DIR_Y]) / h) * var;
    features_variance[FEATURE_RAY_DIR_Z] += square((-ray_dir_p[2] - features[FEATURE_RAY_DIR_Z]) / h) * var;
}

/* HyperbolicShader */
void HyperbolicShader::get_features(const VectorXd &x, Vector22d &features) {
    
    Vector3d C(0.0, 0.0, 0.0);
    
    Vector3d light_dir(-0.25916052767440806, -0.4319342127906801, 0.8638684255813602);
    
    double R = 30.0;
    double camera_radius = 600.0;
    double theta = -x[2] / 2.0 + M_PI / 4.0;
    double alpha, phi;
    double sin_theta, cos_theta, sin_alpha, cos_alpha, sin_theta_phi, cos_theta_phi;
    
    Vector3d ray_dir(x[0] - bound_hi[0] / 2.0, x[1] - bound_hi[1] / 4.0, -bound_hi[0] / 2.0);
    
    if (camera_path == 1) {
        alpha = x[2] / 4.0 + M_PI / 3.0;
        ray_dir[1] = x[1] - bound_hi[1] / 3.0;
        phi = 0.0;
    } else if (camera_path == 2) {
        alpha = x[2] / 4.0 + M_PI / 3.5;
        ray_dir[1] = x[1] - bound_hi[1] / 4.0;
        phi = M_PI / 6.0;
    }
    
    sin_theta = sin(theta);
    cos_theta = cos(theta);
    sin_alpha = sin(alpha);
    cos_alpha = cos(alpha);
    sin_theta_phi = sin(theta + phi);
    cos_theta_phi = cos(theta + phi);
    
    ray_dir.normalize();
    
    Vector3d ray_origin(C);
    ray_origin[0] += camera_radius * sin_theta;
    ray_origin[1] -= camera_radius * tan(alpha);
    ray_origin[2] += camera_radius * cos_theta;
    
    //Vector3d ray_dir1(rotate_x(ray_dir, alpha));
    //Vector3d ray_dir_p(rotate_y(ray_dir1, theta+phi));
    Vector3d ray_dir_p;
    ray_dir_p[0] = cos_theta_phi * ray_dir[0] + \
                   sin_alpha * sin_theta_phi * ray_dir[1] + \
                   cos_alpha * sin_theta_phi * ray_dir[2];
    ray_dir_p[1] = cos_alpha * ray_dir[1] - sin_alpha * ray_dir[2];
    ray_dir_p[2] = -sin_theta_phi * ray_dir[0] + \
                   sin_alpha * cos_theta_phi * ray_dir[1] + \
                   cos_alpha * cos_theta_phi * ray_dir[2];
    //Vector3d ray_dir_p(rotate_y(ray_dir2, phi));
    
    double quadric [10] = {1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -2.0*C[0], R*R, 2.0*C[2], C[0]*C[0]-C[2]*C[2]-C[1]*R*R};
    solve_quadric(quadric, ray_origin, ray_dir_p, features);
    
    double features_scale = 10.0;
    
    features[FEATURE_TEXTURE_S] = (features[FEATURE_POSITION_X] - C[0]) / features_scale;
    features[FEATURE_TEXTURE_T] = (features[FEATURE_POSITION_Z] - C[2]) / features_scale;
    
    features[FEATURE_TANGENT_X] = 1.0;
    features[FEATURE_TANGENT_Y] = -2.0 * features_scale * features[FEATURE_TEXTURE_S] / (R * R);
    features[FEATURE_TANGENT_Z] = 0.0;
    features[FEATURE_BINORMAL_X] = 0.0;
    features[FEATURE_BINORMAL_Y] = 2.0 * features_scale * features[FEATURE_TEXTURE_T] / (R * R);
    features[FEATURE_BINORMAL_Z] = 1.0;
    
#if PYTHON_GEOMETRY_OPT
    fill_features_dir_time(features, -ray_dir_p, x[2]);
#else
    fill_features_light_dir_time(features, light_dir, -ray_dir_p, x[2]);
#endif
}

void HyperbolicShader::get_feature_and_sigma(const VectorXd &x, Vector22d &features, const VectorXd &w, Vector22d &features_sigma) {
    Vector3d C(0.0, 0.0, 0.0);
    
    Vector3d light_dir(-0.25916052767440806, -0.4319342127906801, 0.8638684255813602);
    
    double R = 30.0;
    double camera_radius = 600.0;
    double theta = -x[2] / 2.0 + M_PI / 4.0;
    double alpha, phi;
    double sin_theta, cos_theta, sin_alpha, cos_alpha, sin_theta_phi, cos_theta_phi;
    
    Vector3d ray_dir_unnorm(x[0] - bound_hi[0] / 2.0, 0.0, -bound_hi[0] / 2.0);
    
    if (camera_path == 1) {
        alpha = x[2] / 4.0 + M_PI / 3.0;
        ray_dir_unnorm[1] = x[1] - bound_hi[1] / 3.0;
        phi = 0.0;
    } else if (camera_path == 2) {
        alpha = x[2] / 4.0 + M_PI / 3.5;
        ray_dir_unnorm[1] = x[1] - bound_hi[1] / 4.0;
        phi = M_PI / 6.0;
    }
    
    sin_theta = sin(theta);
    cos_theta = cos(theta);
    sin_alpha = sin(alpha);
    cos_alpha = cos(alpha);
    sin_theta_phi = sin(theta + phi);
    cos_theta_phi = cos(theta + phi);
    
    Vector3d ray_origin(C);
    ray_origin[0] += camera_radius * sin_theta;
    ray_origin[1] -= camera_radius * sin_alpha / cos_alpha;
    ray_origin[2] += camera_radius * cos_theta;
    
    double sin_sin, cos_sin, sin_cos, cos_cos;
    sin_sin = sin_alpha * sin_theta_phi;
    cos_sin = cos_alpha * sin_theta_phi;
    sin_cos = sin_alpha * cos_theta_phi;
    cos_cos = cos_alpha * cos_theta_phi;
    
    Vector3d ray_dir_p;
    ray_dir_p[0] = cos_theta_phi * ray_dir_unnorm[0] + \
                   sin_sin * ray_dir_unnorm[1] + \
                   cos_sin * ray_dir_unnorm[2];
    ray_dir_p[1] = cos_alpha * ray_dir_unnorm[1] - sin_alpha * ray_dir_unnorm[2];
    ray_dir_p[2] = -sin_theta_phi * ray_dir_unnorm[0] + \
                   sin_cos * ray_dir_unnorm[1] + \
                   cos_cos * ray_dir_unnorm[2];
    ray_dir_p.normalize();
    
    //Vector3d ray_dir1(rotate_x(ray_dir, alpha));
    //Vector3d ray_dir_p(rotate_y(ray_dir1, theta+phi));
    //Vector3d ray_dir_p(rotate_y(ray_dir2, phi));
    
    double quadric [10] = {1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -2.0*C[0], R*R, 2.0*C[2], C[0]*C[0]-C[2]*C[2]-C[1]*R*R};
    solve_quadric(quadric, ray_origin, ray_dir_p, features);
    
    double features_scale = 10.0;
    double scale_by_R = 2.0 * features_scale / (R * R);
    
    features[FEATURE_TEXTURE_S] = (features[FEATURE_POSITION_X] - C[0]) / features_scale;
    features[FEATURE_TEXTURE_T] = (features[FEATURE_POSITION_Z] - C[2]) / features_scale;
    
    features[FEATURE_TANGENT_X] = 1.0;
    features[FEATURE_TANGENT_Y] = -scale_by_R * features[FEATURE_TEXTURE_S];
    features[FEATURE_TANGENT_Z] = 0.0;
    features[FEATURE_BINORMAL_X] = 0.0;
    features[FEATURE_BINORMAL_Y] = scale_by_R * features[FEATURE_TEXTURE_T];
    features[FEATURE_BINORMAL_Z] = 1.0;
    
    fill_features_dir_time(features, -ray_dir_p, x[2]);
    
    features_sigma[FEATURE_TIME] = w[2];
    Vector22d features_variance;
    for (int i = 1; i < shader->ndims; i++) {
        features_variance[i] = 0.0;
    }
    
    double h = 1e-8;
    double ray_dir_cache;
    Vector22d features_temp;
    // compute values for x+h
    if (w[0] > 0) {
        ray_dir_cache = ray_dir_unnorm[0];
        ray_dir_unnorm[0] = x[0] + h - bound_hi[0] / 2.0;
        ray_dir_p[0] = cos_theta_phi * ray_dir_unnorm[0] + \
                       sin_sin * ray_dir_unnorm[1] + \
                       cos_sin * ray_dir_unnorm[2];
        ray_dir_p[1] = cos_alpha * ray_dir_unnorm[1] - sin_alpha * ray_dir_unnorm[2];
        ray_dir_p[2] = -sin_theta_phi * ray_dir_unnorm[0] + \
                       sin_cos * ray_dir_unnorm[1] + \
                       cos_cos * ray_dir_unnorm[2];
        ray_dir_p.normalize();
        solve_quadric(quadric, ray_origin, ray_dir_p, features_temp);
        accum_features_variance(features, features_temp, ray_dir_p, features_variance, w[0]*w[0], h);
        ray_dir_unnorm[0] = ray_dir_cache;
    }

    // compute values for y + h
    if (w[1] > 0) {
        ray_dir_cache = ray_dir_unnorm[1];
        ray_dir_unnorm[1] += h;
        ray_dir_p[0] = cos_theta_phi * ray_dir_unnorm[0] + \
                       sin_sin * ray_dir_unnorm[1] + \
                       cos_sin * ray_dir_unnorm[2];
        ray_dir_p[1] = cos_alpha * ray_dir_unnorm[1] - sin_alpha * ray_dir_unnorm[2];
        ray_dir_p[2] = -sin_theta_phi * ray_dir_unnorm[0] + \
                       sin_cos * ray_dir_unnorm[1] + \
                       cos_cos * ray_dir_unnorm[2];
        ray_dir_p.normalize();
        solve_quadric(quadric, ray_origin, ray_dir_p, features_temp);
        accum_features_variance(features, features_temp, ray_dir_p, features_variance, w[1]*w[1], h);
        ray_dir_unnorm[1] = ray_dir_cache;
    }
    // compute values for t + h
    if (w[2] > 0) {
        theta = -(x[2] + h) / 2.0 + M_PI / 4.0;
        if (camera_path == 1) {
            alpha = (x[2] + h) / 4.0 + M_PI / 3.0;
            ray_dir_unnorm[1] = x[1] - bound_hi[1] / 3.0;
            phi = 0.0;
        } else if (camera_path == 2) {
            alpha = (x[2] + h) / 4.0 + M_PI / 3.5;
            ray_dir_unnorm[1] = x[1] - bound_hi[1] / 4.0;
            phi = M_PI / 6.0;
        }
        
        sin_theta = sin(theta);
        cos_theta = cos(theta);
        sin_alpha = sin(alpha);
        cos_alpha = cos(alpha);
        sin_theta_phi = sin(theta + phi);
        cos_theta_phi = cos(theta + phi);
        
        ray_origin[0] = C[0] + camera_radius * sin_theta;
        ray_origin[1] = C[1] - camera_radius * sin_alpha / cos_alpha;
        ray_origin[2] = C[2] + camera_radius * cos_theta;
        
        sin_sin = sin_alpha * sin_theta_phi;
        cos_sin = cos_alpha * sin_theta_phi;
        sin_cos = sin_alpha * cos_theta_phi;
        cos_cos = cos_alpha * cos_theta_phi;
        
        ray_dir_p[0] = cos_theta_phi * ray_dir_unnorm[0] + \
                       sin_sin * ray_dir_unnorm[1] + \
                       cos_sin * ray_dir_unnorm[2];
        ray_dir_p[1] = cos_alpha * ray_dir_unnorm[1] - sin_alpha * ray_dir_unnorm[2];
        ray_dir_p[2] = -sin_theta_phi * ray_dir_unnorm[0] + \
                       sin_cos * ray_dir_unnorm[1] + \
                       cos_cos * ray_dir_unnorm[2];
        ray_dir_p.normalize();
        solve_quadric(quadric, ray_origin, ray_dir_p, features_temp);
        accum_features_variance(features, features_temp, ray_dir_p, features_variance, w[2]*w[2], h);
    }
    
    features_sigma[FEATURE_POSITION_X] = sqrt(features_variance[FEATURE_POSITION_X]);
    features_sigma[FEATURE_POSITION_Y] = sqrt(features_variance[FEATURE_POSITION_Y]);
    features_sigma[FEATURE_POSITION_Z] = sqrt(features_variance[FEATURE_POSITION_Z]);
    features_sigma[FEATURE_RAY_DIR_X] = sqrt(features_variance[FEATURE_RAY_DIR_X]);
    features_sigma[FEATURE_RAY_DIR_Y] = sqrt(features_variance[FEATURE_RAY_DIR_Y]);
    features_sigma[FEATURE_RAY_DIR_Z] = sqrt(features_variance[FEATURE_RAY_DIR_Z]);
    features_sigma[FEATURE_NORMAL_X] = sqrt(features_variance[FEATURE_NORMAL_X]);
    features_sigma[FEATURE_NORMAL_Y] = sqrt(features_variance[FEATURE_NORMAL_Y]);
    features_sigma[FEATURE_NORMAL_Z] = sqrt(features_variance[FEATURE_NORMAL_Z]);
    features_sigma[FEATURE_IS_VALID] = sqrt(features_variance[FEATURE_IS_VALID]);
    
    features_sigma[FEATURE_TEXTURE_S] = features_sigma[FEATURE_POSITION_X] / features_scale;
    features_sigma[FEATURE_TEXTURE_T] = features_sigma[FEATURE_POSITION_Z] / features_scale;
    
    features_sigma[FEATURE_TANGENT_X] = 0.0;
    features_sigma[FEATURE_TANGENT_Y] = -scale_by_R * features_sigma[FEATURE_TEXTURE_S];
    features_sigma[FEATURE_TANGENT_Z] = 0.0;
    
    features_sigma[FEATURE_BINORMAL_X] = 0.0;
    features_sigma[FEATURE_BINORMAL_Y] = scale_by_R * features_sigma[FEATURE_TEXTURE_T];
    features_sigma[FEATURE_BINORMAL_Z] = 0.0;
}

inline void HyperbolicShader::accum_features_variance(const Vector22d &features, const Vector22d &features_temp, const Vector3d &ray_dir, Vector22d &features_variance, double var, double h) {
    features_variance[FEATURE_POSITION_X] += square((features_temp[FEATURE_POSITION_X] - features[FEATURE_POSITION_X]) / h) * var;
    features_variance[FEATURE_POSITION_Y] += square((features_temp[FEATURE_POSITION_Y] - features[FEATURE_POSITION_Y]) / h) * var;
    features_variance[FEATURE_POSITION_Z] += square((features_temp[FEATURE_POSITION_Z] - features[FEATURE_POSITION_Z]) / h) * var;
    features_variance[FEATURE_RAY_DIR_X] += square((-ray_dir[0] - features[FEATURE_RAY_DIR_X]) / h) * var;
    features_variance[FEATURE_RAY_DIR_Y] += square((-ray_dir[1] - features[FEATURE_RAY_DIR_Y]) / h) * var;
    features_variance[FEATURE_RAY_DIR_Z] += square((-ray_dir[2] - features[FEATURE_RAY_DIR_Z]) / h) * var;
    features_variance[FEATURE_NORMAL_X] += square((features_temp[FEATURE_NORMAL_X] - features[FEATURE_NORMAL_X]) / h) * var;
    features_variance[FEATURE_NORMAL_Y] += square((features_temp[FEATURE_NORMAL_Y] - features[FEATURE_NORMAL_Y]) / h) * var;
    features_variance[FEATURE_NORMAL_Z] += square((features_temp[FEATURE_NORMAL_Z] - features[FEATURE_NORMAL_Z]) / h) * var;
    features_variance[FEATURE_IS_VALID] += square((features_temp[FEATURE_IS_VALID] - features[FEATURE_IS_VALID]) / h) * var;
}

void SphereShader::get_features(const VectorXd &x, Vector22d &features) {
    Vector3d C(0.0, 0.0, 0.0);
    double R = 175.0;
    
    Vector3d light_dir(-0.25916052767440806, -0.4319342127906801, 0.8638684255813602);
    
    double camera_radius = 300.0;
    double alpha, theta;
    
    if (camera_path == 1) {
        alpha = M_PI / 6.0;
        theta = -x[2];
    } else if (camera_path == 2) {
        alpha = -M_PI / 4.0;
        theta = x[2];
    } else {
        throw runtime_error("Could not create camera path: unknown camera path number");
    }
    
    double sin_theta, cos_theta, sin_alpha, cos_alpha;
    sin_theta = sin(theta);
    cos_theta = cos(theta);
    sin_alpha = sin(alpha);
    cos_alpha = cos(alpha);
    
    Vector3d ray_origin(C);
    ray_origin[0] += camera_radius * sin_theta;
    ray_origin[1] -= camera_radius * sin_alpha / cos_alpha;
    ray_origin[2] += camera_radius * cos_theta;
    
    Vector3d ray_dir(x[0] - bound_hi[0] / 2.0, x[1] - bound_hi[1] / 2.0, -bound_hi[0] / 2.0);
    ray_dir.normalize();
    
    Vector3d ray_dir_p;
    ray_dir_p[0] = cos_theta * ray_dir[0] + \
                   sin_alpha * sin_theta * ray_dir[1] + \
                   cos_alpha * sin_theta * ray_dir[2];
    ray_dir_p[1] = cos_alpha * ray_dir[1] - sin_alpha * ray_dir[2];
    ray_dir_p[2] = -sin_theta * ray_dir[0] + \
                   sin_alpha * cos_theta * ray_dir[1] + \
                   cos_alpha * cos_theta * ray_dir[2];
    
    double quadric[10] = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -2.0*C[0], -2.0*C[1], -2.0*C[2], C[0]*C[0]+C[1]*C[1]+C[2]*C[2]-R*R};
    solve_quadric(quadric, ray_origin, ray_dir_p, features);
    
    double features_scale = 320.0 / M_PI;
    double u = atan2(features[FEATURE_NORMAL_X], features[FEATURE_NORMAL_Z]);
    double v = acos(features[FEATURE_NORMAL_Y]);
    features[FEATURE_TEXTURE_S] = u * features_scale;
    features[FEATURE_TEXTURE_T] = v * features_scale;
    
    double cos_u = cos(u);
    double sin_u = sin(u);
    features[FEATURE_TANGENT_X] = cos_u;
    features[FEATURE_TANGENT_Y] = 0.0;
    features[FEATURE_TANGENT_Z] = -sin_u;
    
    double cos_v = cos(v);
    features[FEATURE_BINORMAL_X] = -sin_u * cos_v;
    features[FEATURE_BINORMAL_Y] = sin(v);
    features[FEATURE_BINORMAL_Z] = -cos_u * cos_v;
    
#if PYTHON_GEOMETRY_OPT
    fill_features_dir_time(features, -ray_dir_p, x[2]);
#else
    fill_features_light_dir_time(features, light_dir, -ray_dir_p, x[2]);
#endif
}

void SphereShader::get_feature_and_sigma(const VectorXd &x, Vector22d &features, const VectorXd &w, Vector22d &features_sigma) {
    Vector3d C(0.0, 0.0, 0.0);
    double R = 175.0;
    
    Vector3d light_dir(-0.25916052767440806, -0.4319342127906801, 0.8638684255813602);
    
    double camera_radius = 300.0;
    double alpha, theta;
    
    if (camera_path == 1) {
        alpha = M_PI / 6.0;
        theta = -x[2];
    } else if (camera_path == 2) {
        alpha = -M_PI / 4.0;
        theta = x[2];
    } else {
        throw runtime_error("Could not create camera path: unknown camera path number");
    }
    
    double sin_theta, cos_theta, sin_alpha, cos_alpha;
    sin_theta = sin(theta);
    cos_theta = cos(theta);
    sin_alpha = sin(alpha);
    cos_alpha = cos(alpha);
    
    Vector3d ray_origin(C);
    ray_origin[0] += camera_radius * sin_theta;
    ray_origin[1] -= camera_radius * sin_alpha / cos_alpha;
    ray_origin[2] += camera_radius * cos_theta;
    
    Vector3d ray_dir(x[0] - bound_hi[0] / 2.0, x[1] - bound_hi[1] / 2.0, -bound_hi[0] / 2.0);
    Vector3d ray_dir_p;
    ray_dir_p[0] = cos_theta * ray_dir[0] + \
                   sin_alpha * sin_theta * ray_dir[1] + \
                   cos_alpha * sin_theta * ray_dir[2];
    ray_dir_p[1] = cos_alpha * ray_dir[1] - sin_alpha * ray_dir[2];
    ray_dir_p[2] = -sin_theta * ray_dir[0] + \
                   sin_alpha * cos_theta * ray_dir[1] + \
                   cos_alpha * cos_theta * ray_dir[2];
    ray_dir_p.normalize();
                   
    double quadric[10] = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -2.0*C[0], -2.0*C[1], -2.0*C[2], C[0]*C[0]+C[1]*C[1]+C[2]*C[2]-R*R};
    solve_quadric(quadric, ray_origin, ray_dir_p, features);
    
    double features_scale = 320.0 / M_PI;
    double u = atan2(features[FEATURE_NORMAL_X], features[FEATURE_NORMAL_Z]);
    double v = acos(features[FEATURE_NORMAL_Y]);
    features[FEATURE_TEXTURE_S] = u * features_scale;
    features[FEATURE_TEXTURE_T] = v * features_scale;
    
    double cos_u = cos(u);
    double sin_u = sin(u);
    features[FEATURE_TANGENT_X] = cos_u;
    features[FEATURE_TANGENT_Y] = 0.0;
    features[FEATURE_TANGENT_Z] = -sin_u;
    
    double cos_v = cos(v);
    double sin_v = sin(v);
    features[FEATURE_BINORMAL_X] = -sin_u * cos_v;
    features[FEATURE_BINORMAL_Y] = sin_v;
    features[FEATURE_BINORMAL_Z] = -cos_u * cos_v;
    
    fill_features_dir_time(features, -ray_dir_p, x[2]);
    
    features_sigma[FEATURE_TIME] = w[2];
    Vector22d features_variance;
    for (int i = 1; i < shader->ndims; i++) {
        features_variance[i] = 0.0;
    }
    
    double h = 1e-8;
    double ray_dir_cache;
    Vector22d features_temp;
    
    // compute values for x+h
    if (w[0] > 0) {
        ray_dir_cache = ray_dir[0];
        ray_dir[0] = x[0] + h - bound_hi[0] / 2.0;
        ray_dir_p[0] = cos_theta * ray_dir[0] + \
                       sin_alpha * sin_theta * ray_dir[1] + \
                       cos_alpha * sin_theta * ray_dir[2];
        ray_dir_p[1] = cos_alpha * ray_dir[1] - sin_alpha * ray_dir[2];
        ray_dir_p[2] = -sin_theta * ray_dir[0] + \
                       sin_alpha * cos_theta * ray_dir[1] + \
                       cos_alpha * cos_theta * ray_dir[2];
        ray_dir_p.normalize();
        solve_quadric(quadric, ray_origin, ray_dir_p, features_temp);
        u = atan2(features_temp[FEATURE_NORMAL_X], features_temp[FEATURE_NORMAL_Z]);
        v = acos(features_temp[FEATURE_NORMAL_Y]);
        accum_features_variance(features, features_temp, ray_dir_p, u, v, features_scale, features_variance, w[0]*w[0], h);
        ray_dir[0] = ray_dir_cache;
    }
    // compute values for y+h
    if (w[1] > 0) {
        ray_dir_cache = ray_dir[1];
        ray_dir[1] = x[1] + h - bound_hi[1] / 2.0;
        ray_dir_p[0] = cos_theta * ray_dir[0] + \
                       sin_alpha * sin_theta * ray_dir[1] + \
                       cos_alpha * sin_theta * ray_dir[2];
        ray_dir_p[1] = cos_alpha * ray_dir[1] - sin_alpha * ray_dir[2];
        ray_dir_p[2] = -sin_theta * ray_dir[0] + \
                       sin_alpha * cos_theta * ray_dir[1] + \
                       cos_alpha * cos_theta * ray_dir[2];
        ray_dir_p.normalize();
        solve_quadric(quadric, ray_origin, ray_dir_p, features_temp);
        u = atan2(features_temp[FEATURE_NORMAL_X], features_temp[FEATURE_NORMAL_Z]);
        v = acos(features_temp[FEATURE_NORMAL_Y]);
        accum_features_variance(features, features_temp, ray_dir_p, u, v, features_scale, features_variance, w[1]*w[1], h);
        ray_dir[1] = ray_dir_cache;
    }
    // compute values for t+h
    if (w[2] > 0) {
        if (camera_path == 1) {
            theta = -(x[2] + h);
        } else if (camera_path == 2) {
            theta = (x[2] + h);
        }
        sin_theta = sin(theta);
        cos_theta = cos(theta);
        
        ray_origin[0] = C[0] + camera_radius * sin_theta;
        ray_origin[2] = C[2] + camera_radius * cos_theta;
        
        ray_dir_p[0] = cos_theta * ray_dir[0] + \
                       sin_alpha * sin_theta * ray_dir[1] + \
                       cos_alpha * sin_theta * ray_dir[2];
        ray_dir_p[1] = cos_alpha * ray_dir[1] - sin_alpha * ray_dir[2];
        ray_dir_p[2] = -sin_theta * ray_dir[0] + \
                       sin_alpha * cos_theta * ray_dir[1] + \
                       cos_alpha * cos_theta * ray_dir[2];
        ray_dir_p.normalize();
        solve_quadric(quadric, ray_origin, ray_dir_p, features_temp);
        u = atan2(features_temp[FEATURE_NORMAL_X], features_temp[FEATURE_NORMAL_Z]);
        v = acos(features_temp[FEATURE_NORMAL_Y]);
        accum_features_variance(features, features_temp, ray_dir_p, u, v, features_scale, features_variance, w[2]*w[2], h);
    }
    
    features_sigma[FEATURE_POSITION_X] = sqrt(features_variance[FEATURE_POSITION_X]);
    features_sigma[FEATURE_POSITION_Y] = sqrt(features_variance[FEATURE_POSITION_Y]);
    features_sigma[FEATURE_POSITION_Z] = sqrt(features_variance[FEATURE_POSITION_Z]);
    features_sigma[FEATURE_RAY_DIR_X] = sqrt(features_variance[FEATURE_RAY_DIR_X]);
    features_sigma[FEATURE_RAY_DIR_Y] = sqrt(features_variance[FEATURE_RAY_DIR_Y]);
    features_sigma[FEATURE_RAY_DIR_Z] = sqrt(features_variance[FEATURE_RAY_DIR_Z]);
    features_sigma[FEATURE_TEXTURE_S] = sqrt(features_variance[FEATURE_TEXTURE_S]);
    features_sigma[FEATURE_TEXTURE_T] = sqrt(features_variance[FEATURE_TEXTURE_T]);
    features_sigma[FEATURE_BINORMAL_X] = sqrt(features_variance[FEATURE_BINORMAL_X]);
    features_sigma[FEATURE_BINORMAL_Z] = sqrt(features_variance[FEATURE_BINORMAL_Z]);
    features_sigma[FEATURE_IS_VALID] = sqrt(features_variance[FEATURE_IS_VALID]);
    
    features_sigma[FEATURE_NORMAL_X] = features_sigma[FEATURE_POSITION_X] / R;
    features_sigma[FEATURE_NORMAL_Y] = features_sigma[FEATURE_POSITION_Y] / R;
    features_sigma[FEATURE_NORMAL_Z] = features_sigma[FEATURE_POSITION_Z] / R;
    
    double du = features_sigma[FEATURE_TEXTURE_S] / features_scale;
    features_sigma[FEATURE_TANGENT_X] = -sin_u * du;
    features_sigma[FEATURE_TANGENT_Y] = 0.0;
    features_sigma[FEATURE_TANGENT_Z] = -cos_u * du;
    
    features_sigma[FEATURE_BINORMAL_Y] = cos_v * features_sigma[FEATURE_TEXTURE_T] / features_scale;
}

inline void SphereShader::accum_features_variance(const Vector22d &features, const Vector22d &features_temp, const Vector3d &ray_dir_p, const double u, const double v, const double features_scale, Vector22d &features_variance, double var, double h) {
    features_variance[FEATURE_POSITION_X] += square((features_temp[FEATURE_POSITION_X] - features[FEATURE_POSITION_X]) / h) * var;
    features_variance[FEATURE_POSITION_Y] += square((features_temp[FEATURE_POSITION_Y] - features[FEATURE_POSITION_Y]) / h) * var;
    features_variance[FEATURE_POSITION_Z] += square((features_temp[FEATURE_POSITION_Z] - features[FEATURE_POSITION_Z]) / h) * var;
    features_variance[FEATURE_RAY_DIR_X] += square((-ray_dir_p[0] - features[FEATURE_RAY_DIR_X]) / h) * var;
    features_variance[FEATURE_RAY_DIR_Y] += square((-ray_dir_p[1] - features[FEATURE_RAY_DIR_Y]) / h) * var;
    features_variance[FEATURE_RAY_DIR_Z] += square((-ray_dir_p[2] - features[FEATURE_RAY_DIR_Z]) / h) * var;
    features_variance[FEATURE_TEXTURE_S] += square((u * features_scale - features[FEATURE_TEXTURE_S]) / h) * var;
    features_variance[FEATURE_TEXTURE_T] += square((v * features_scale - features[FEATURE_TEXTURE_T]) / h) * var;
    double sin_u, cos_u, sin_v, cos_v;
    sin_u = sin(u);
    cos_u = cos(u);
    sin_v = sin(v);
    cos_v = cos(v);
    features_variance[FEATURE_BINORMAL_X] += square((-sin_u * cos_v - features[FEATURE_BINORMAL_X]) / h) * var;
    features_variance[FEATURE_BINORMAL_Z] += square((-cos_u * cos_v - features[FEATURE_BINORMAL_Z]) / h) * var;
    features_variance[FEATURE_IS_VALID] += square((features_temp[FEATURE_IS_VALID] - features[FEATURE_IS_VALID]) / h) * var;
}

void solve_quadric(double quadric[10], const Vector3d &ray_origin, const Vector3d &ray_dir_p, Vector22d &features) {
    double Aq = quadric[0] * square(ray_dir_p[0]) + \
                quadric[1] * square(ray_dir_p[1]) + \
                quadric[2] * square(ray_dir_p[2]) + \
                quadric[3] * ray_dir_p[0] * ray_dir_p[1] + \
                quadric[4] * ray_dir_p[0] * ray_dir_p[2] + \
                quadric[5] * ray_dir_p[1] * ray_dir_p[2];
    double Bq = 2.0 * (quadric[0] * ray_origin[0] * ray_dir_p[0] + \
                       quadric[1] * ray_origin[1] * ray_dir_p[1] + \
                       quadric[2] * ray_origin[2] * ray_dir_p[2]) + \
                quadric[3] * (ray_origin[0] * ray_dir_p[1] + ray_origin[1] * ray_dir_p[0]) + \
                quadric[4] * (ray_origin[0] * ray_dir_p[2] + ray_origin[2] * ray_dir_p[0]) + \
                quadric[5] * (ray_origin[1] * ray_dir_p[2] + ray_origin[2] * ray_dir_p[1]) + \
                quadric[6] * ray_dir_p[0] + \
                quadric[7] * ray_dir_p[1] + \
                quadric[8] * ray_dir_p[2]; 
    double Cq = quadric[0] * square(ray_origin[0]) + \
                quadric[1] * square(ray_origin[1]) +\
                quadric[2] * square(ray_origin[2]) + \
                quadric[3] * ray_origin[0] * ray_origin[1] + \
                quadric[4] * ray_origin[0] * ray_origin[2] + \
                quadric[5] * ray_origin[1] * ray_origin[2] + \
                quadric[6] * ray_origin[0] + \
                quadric[7] * ray_origin[1] + \
                quadric[8] * ray_origin[2] + \
                quadric[9];   
                
    double t_ray;
    double root2 = Bq * Bq - 4.0 * Aq * Cq;
    
    if (abs(Aq) < 1e-8) {
        t_ray = -Cq / Bq;
    } else {
        if (root2 >= 0.0) {
            double sqrt_root2 = sqrt(root2);
            double t0 = (-Bq - sqrt_root2) / (2.0 * Aq);
            double t1 = (-Bq + sqrt_root2) / (2.0 * Aq);
            if (Aq < 0) {
                double t_mid = t0;
                t0 = t1;
                t1 = t_mid;
            }
            if (t0 >= 0) {
                t_ray = t0;
            } else if (t1 >= 0) {
                t_ray = t1;
            } else {
                root2 = -1;
            }
        }
    }
    
    if (root2 < 0.0) {
        for (int i = 0; i < features.size(); i++) {
            features[i] = NAN;
        }
        features[FEATURE_IS_VALID] = root2;
        return;
    }
    
    Vector3d intersect_pos;
    intersect_pos = ray_origin + t_ray * ray_dir_p;
    
    Vector3d normal;
    normal[0] = quadric[6] + 2.0 * quadric[0] * intersect_pos[0] + \
                quadric[3] * intersect_pos[1] + quadric[4] * intersect_pos[2];
    normal[1] = quadric[7] + 2.0 * quadric[1] * intersect_pos[1] + \
                quadric[3] * intersect_pos[0] + quadric[5] * intersect_pos[2];
    normal[2] = quadric[8] + 2.0 * quadric[2] * intersect_pos[2] + \
                quadric[4] * intersect_pos[0] + quadric[5] * intersect_pos[1];
    normal.normalize();
                
    if (normal.dot(ray_dir_p) > 0.0) {
        normal *= -1;
    }
    features[FEATURE_POSITION_X] = intersect_pos[0];
    features[FEATURE_POSITION_Y] = intersect_pos[1];
    features[FEATURE_POSITION_Z] = intersect_pos[2];
    features[FEATURE_NORMAL_X] = normal[0];
    features[FEATURE_NORMAL_Y] = normal[1];
    features[FEATURE_NORMAL_Z] = normal[2];
    features[FEATURE_IS_VALID] = root2;
}

inline Vector3d rotate_x(const Vector3d &x, double alpha) {
    Vector3d ans(x);
    double cos_alpha = cos(alpha);
    double sin_alpha = sin(alpha);
    ans[1] = x[1] * cos_alpha - x[2] * sin_alpha;
    ans[2] = x[1] * sin_alpha + x[2] * cos_alpha;
    return ans;
}

inline Vector3d rotate_y(const Vector3d &x, double alpha) {
    Vector3d ans(x);
    double cos_alpha = cos(alpha);
    double sin_alpha = sin(alpha);
    ans[0] = x[0] * cos_alpha + x[2] * sin_alpha;
    ans[2] = -x[0] * sin_alpha + x[2] * cos_alpha;
    return ans;
}

inline Vector3d rotate_z(const Vector3d &x, double alpha) {
    Vector3d ans(x);
    double cos_alpha = cos(alpha);
    double sin_alpha = sin(alpha);
    ans[0] = x[0] * cos_alpha - x[1] * sin_alpha;
    ans[1] = x[0] * sin_alpha + x[1] * cos_alpha;
    return ans;
}

inline void fill_features_light_dir_time(Vector22d &features, const Vector3d &light_dir, const Vector3d &ray_dir_p, double time) {
    features[FEATURE_LIGHT_DIR_X] = light_dir[0];
    features[FEATURE_LIGHT_DIR_Y] = light_dir[1];
    features[FEATURE_LIGHT_DIR_Z] = light_dir[2];
    fill_features_dir_time(features, ray_dir_p, time);
}

inline void fill_features_dir_time(Vector22d &features, const Vector3d &ray_dir_p, double time) {
    features[FEATURE_RAY_DIR_X] = ray_dir_p[0];
    features[FEATURE_RAY_DIR_Y] = ray_dir_p[1];
    features[FEATURE_RAY_DIR_Z] = ray_dir_p[2];
    features[FEATURE_TIME] = time;
}

void save_image(const VectorXd &L, const string &filename, int width, int height, int nchannels) {
    if (L.size() != width * height * nchannels) {
        fprintf(stderr, "size_mismatch\n");
        throw runtime_error("Size mismatch in ImageAverage::save");
    }
    printf("encoding image of size %dx%d (%d)\n", width, height, L.size());
    
    vector<unsigned char> image_8(L.size()*4);
    int i = 0;
    int read_idx = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int k = 0; k < nchannels; k++) {
                double fval = L[read_idx];
                if (std::isnan(fval)) { fval = 0; }
                else if (fval < 0) { fval = 0; }
                else if (fval > 1) { fval = 1; }
                unsigned char c = int(fval * 256.0-1e-4);
                image_8[i] = c;
                i++;
                read_idx++;
            }
            if (nchannels == 1) {
                image_8[i] = image_8[i-1];
                image_8[i+1] = image_8[i-1];
                image_8[i+2] = 255;
                i += 3;
            }
            else {
                image_8[i] = 255;
                i++;
            }
        }
    }
    
    unsigned error = lodepng::encode((filename + string(".png")).c_str(), image_8, width, height);
    
    if (error) {
        std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
        throw runtime_error("Encoder error");
    }
}

VectorXd initialize_gaussian_sample(int ncache, int seed) {
    VectorXd sample(ncache);
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < ncache; i++) {
        sample[i] = distribution(generator);
    }
    return sample;
}

void init_gaussian_rng_samples(int seed) {
    Gaussian_RNG::sample = initialize_gaussian_sample(Gaussian_RNG::ncache, seed);
    Gaussian_RNG::sample_color = initialize_gaussian_sample(Gaussian_RNG::ncache, seed);
}

void reset_gaussian_rng() {
    Gaussian_RNG::index = Gaussian_RNG::index_r = Gaussian_RNG::index_g = Gaussian_RNG::index_b = 0;
}

void seed_all_rngs(int seed) {
    srand(seed);
    our_srand(seed);
    reset_gaussian_rng();
}

thread_local VectorXd Gaussian_RNG::sample;
thread_local VectorXd Gaussian_RNG::sample_color;
thread_local unsigned int Gaussian_RNG::index = 0;
thread_local unsigned int Gaussian_RNG::index_r = 0;
thread_local unsigned int Gaussian_RNG::index_g = 0;
thread_local unsigned int Gaussian_RNG::index_b = 0;
