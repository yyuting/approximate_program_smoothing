
#ifndef _problem_h
#define _problem_h

#include <vector>
#include <stdexcept>
#include <random>
#include <fstream>
#include "util.h"

#define MAX_SIGMA_RATIO 0.5                 /* Default maximum sigma as a function of the range of the sample domain. */

using namespace std;
class Problem;

DECLARE_STRING_EXCEPTION(CheckFailed)

#define DEFAULT_ERROR_SAMPLES   100
#define DEFAULT_GROUND_SAMPLES  10000

#define PASS_IN_CONSTANT_FEATURES           1
#define PYTHON_GEOMETRY_OPT                 1

/* Shader features */
#define FEATURE_TIME        0
#define FEATURE_POSITION_X  1
#define FEATURE_POSITION_Y  2
#define FEATURE_POSITION_Z  3
#define FEATURE_RAY_DIR_X   4
#define FEATURE_RAY_DIR_Y   5
#define FEATURE_RAY_DIR_Z   6
#define FEATURE_TEXTURE_S   7
#define FEATURE_TEXTURE_T   8
#define FEATURE_NORMAL_X    9
#define FEATURE_NORMAL_Y    10
#define FEATURE_NORMAL_Z    11
#define FEATURE_TANGENT_X   12
#define FEATURE_TANGENT_Y   13
#define FEATURE_TANGENT_Z   14
#define FEATURE_BINORMAL_X  15
#define FEATURE_BINORMAL_Y  16
#define FEATURE_BINORMAL_Z  17
#define FEATURE_IS_VALID    18
#define FEATURE_LIGHT_DIR_X 19
#define FEATURE_LIGHT_DIR_Y 20
#define FEATURE_LIGHT_DIR_Z 21

#define NUM_FEATURES 22

/** Problem class for minimizer. */
class Problem { public:
    int ndims;                                                        /**< Number of dimensions for problem. */
    VectorXd bound_lo, bound_hi;                                      /**< Vector of (lower, upper) bounds for each variable passed to solver. */
    VectorXd vec_output;                                              /**< Additional vector output field, can be used as desired by the application. */
    VectorXd adjust_var;                                              /**< Additional vector input field, used to tune variances. */
    bool probability_select_used;                                     /**< Additional output field, whether probabilistic model was used in g, so that resampling g is appropriate. */
    bool geometry_only;
    virtual double f(const VectorXd &x) = 0;                          /**< Objective function (OF) at given variable vector. */
    virtual double g(const VectorXd &x, const VectorXd &w) = 0;       /**< Approximately bandlimited objective with sample spacings w along each dimension. */
    Problem();
    virtual ~Problem();

    /** Helper function: convert scalar v to vector (v, v, v, ...) of size ndims */
    VectorXd make_vector(double v=0.0);
    
    /** Get a maximum reasonable number of dimensions. */
    virtual int max_ndims();
    
    /** A random sample of variables from the valid bounding box (bound_lo, bound_hi) (precondition: x has length ndims) */
    virtual void sample_vars(VectorXd &x);
    
    /** A slower variant of sample_vars(VectorXd &x), which returns the sampled vector instead. By default, simply calls the one argument sample_vars(). */
    virtual VectorXd sample_vars();
    
    /** Time different methods. */
    virtual void time(double &f_time, double &g_time, double timeout=0.05);
    
    /** Obtain Lp error between g (estimated), and g (ground truth) (normalized by dividing by nsamples_error).
        We use nsamples_error samples from the sample_vars() domain using sigma drawn from [0.0, MAX_SIGMA_RATIO*range], where
        range is the range of the sample domain. Ground truth is estimated using nsamples_ground samples. */
    virtual double error(int nsamples_error=DEFAULT_ERROR_SAMPLES, int nsamples_ground=DEFAULT_GROUND_SAMPLES, int seed=1, double pnorm=2.0, bool is_g=true, const vector<double> *render_sigma=NULL, VectorXd *ground_truth_cache=NULL);
    
    virtual void print_info();
    
    virtual void accum_output(VectorXd &out, double val, int nchannel);
    
    virtual void geometry_time(double &f_time, double &g_time, double timeout);
};

/** Problem class that tracks correlation coefficients associated for operators (+, -, *, /)
    if special methods (log_rho_plus, ...) are called in place of operators. */
class ProblemLogRho: public Problem { public:
    VectorXd a_accum, b_accum, a2_accum, b2_accum, ab_accum;        /**< Accumulators for a, b, a**2, b**2, a*b. */
    int n_samples;
    int target_f_calls;
    ProblemLogRho();
    void log_rho(int index, double a, double b);
    double log_rho_plus(int index, double a, double b);
    double log_rho_minus(int index, double a, double b);
    double log_rho_times(int index, double a, double b);
    double log_rho_divide(int index, double a, double b);
    void log_rho_reserve(int n);                                  /**< Reserve given number of accumulators. */
    void log_rho_reset();                                         /**< Set accumulators to zero. */
    void print_info();                                            /**< Compute and print correlations. */
};

class SampledProblem : public Problem { public:
    Problem *problem;
    int nsamples;
    
    SampledProblem(Problem *problem_, int nsamples_=5);
    
    void draw_sample(VectorXd &x_sample, int sample_index, const VectorXd &x, const VectorXd &w);
    
    double f(const VectorXd &x);
    double g(const VectorXd &x, const VectorXd &w);
    
    void sample_vars(VectorXd &x);
    VectorXd sample_vars();
    void geometry_time(double &f_time, double &g_time, double timeout);
};

class ShaderProblem : public Problem { public:
    Problem *shader;
    int camera_path;
    int nonconstant_indices;
    static thread_local VectorXd local_features, local_features_sigma;
    ShaderProblem(int ndims_, Problem *shader_, int height, int width, double min_time, double max_time, int path);
    
    virtual void get_features(const VectorXd &x, Vector22d &features)=0;
    void get_features_sigma(const VectorXd &x, const Vector22d &features, const VectorXd &w, Vector22d &features_sigma);
    virtual void get_feature_and_sigma(const VectorXd &x, Vector22d &features, const VectorXd &w, Vector22d &features_sigma);
    void geometry_time(double &f_time, double &g_time, double timeout);
    
    double f(const VectorXd &x);
    double g(const VectorXd &x, const VectorXd &w);
};

class PlaneShader : public ShaderProblem { public:
    PlaneShader(int ndims_, Problem *shader_, int height, int width, double min_time, double max_time, int path) : ShaderProblem(ndims_, shader_, height, width, min_time, max_time, path) {
#if PASS_IN_CONSTANT_FEATURES
#if PYTHON_GEOMETRY_OPT
        nonconstant_indices = 7;
#else
        nonconstant_indices = 9;
#endif
#endif
    }
    void get_features(const VectorXd &x, Vector22d &features);
    void get_feature_and_sigma(const VectorXd &x, Vector22d &features, const VectorXd &w, Vector22d &features_sigma);
    inline void accum_features_variance(const Vector22d &features, const Vector3d &intersect_pos, const Vector3d &ray_dir_p, Vector22d &features_variance, double var, double h);
};

class HyperbolicShader : public ShaderProblem { public:
    HyperbolicShader(int ndims_, Problem *shader_, int height, int width, double min_time, double max_time, int path) : ShaderProblem(ndims_, shader_, height, width, min_time, max_time, path) {
#if PASS_IN_CONSTANT_FEATURES
        nonconstant_indices = 19;
#endif
    };
    void get_features(const VectorXd &x, Vector22d &features);
    void get_feature_and_sigma(const VectorXd &x, Vector22d &features, const VectorXd &w, Vector22d &features_sigma);
    inline void accum_features_variance(const Vector22d &features, const Vector22d &features_temp, const Vector3d &ray_dir, Vector22d &features_variance, double var, double h);
};

class SphereShader : public ShaderProblem { public:
    SphereShader(int ndims_, Problem *shader_, int height, int width, double min_time, double max_time, int path) : ShaderProblem(ndims_, shader_, height, width, min_time, max_time, path) {
#if PASS_IN_CONSTANT_FEATURES
        nonconstant_indices = 19;
#endif
    }
    void get_features(const VectorXd &x, Vector22d &features);
    void get_feature_and_sigma(const VectorXd &x, Vector22d &features, const VectorXd &w, Vector22d &features_sigma);
    inline void accum_features_variance(const Vector22d &features, const Vector22d &features_temp, const Vector3d &ray_dir_p, const double u, const double v, const double features_scale, Vector22d &features_variance, double var, double h);
};

void solve_quadric(double quadric[10], const Vector3d &ray_origin, const Vector3d &ray_dir_p, Vector22d&features);
inline Vector3d rotate_x(const Vector3d &x, double alpha);
inline Vector3d rotate_y(const Vector3d &x, double alpha);
inline Vector3d rotate_y(const Vector3d &x, double alpha);
inline void fill_features_light_dir_time(Vector22d &features, const Vector3d &light_dir, const Vector3d &ray_dir_p, double time);
inline void fill_features_dir_time(Vector22d &features, const Vector3d &ray_dir_p, double time);

Problem *create_problem(const string name, int ndims);                          /**< Create problem given problem name string and number of dimensions. */

void save_image(const VectorXd &L, const string &filename, int width, int height, int nchannels);

class Gaussian_RNG { public:
    static thread_local VectorXd sample;
    static thread_local VectorXd sample_color;
    static thread_local unsigned int index;
    static thread_local unsigned int index_r;
    static thread_local unsigned int index_g;
    static thread_local unsigned int index_b;
    const static unsigned int ncache = 8192;
    
    inline static double get(int offset) {
        unsigned int idx = (index + offset) & (ncache - 1);
        return sample[idx];
    }
    
    inline static void advance(int n) {
        index += n;
    }
};

VectorXd initialize_gaussian_sample(int ncache, int seed);

void init_gaussian_rng_samples(int seed);

void reset_gaussian_rng();

void seed_all_rngs(int seed);

#endif
