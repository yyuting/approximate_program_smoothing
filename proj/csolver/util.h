#ifndef _util_h
#define _util_h

#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <cassert>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include <map>

#include "timer.h"

typedef __float128 float128;

using Eigen::VectorXd;
using Eigen::VectorXf;
using Eigen::VectorXi;
using Eigen::Map;
using Eigen::Ref;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd;
typedef Eigen::Matrix<double, 3, 1> Vector3d;
typedef Eigen::Matrix<double, 22, 1> Vector22d;
using namespace std;

template<class real>
using VectorX = Eigen::Matrix<real, Eigen::Dynamic, 1>;

#define DEFAULT_PRECISION 16

#define DECLARE_STRING_EXCEPTION(cls) \
class cls: public runtime_error { public: \
    cls(const string s) :runtime_error(s) { } \
};

#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif

#ifndef ROOT_2_PI
#define ROOT_2_PI           2.5066282746310002
#endif

#ifndef TWO_OVER_ROOT_PI
#define TWO_OVER_ROOT_PI      1.1283791670955126
#endif

#ifndef ROOT_PI
#define ROOT_PI             1.7724538509055159
#endif

#ifndef ROOT_2OVERPI
#define ROOT_2OVERPI        0.7978845608028654
#endif

#ifndef SQRT2
#define SQRT2               1.4142135623730951
#endif

#ifndef SQRT3
#define SQRT3               1.7320508075688772
#endif

#ifndef GAUSS_SIGMA_FACTOR
#define GAUSS_SIGMA_FACTOR  0.6366197723675814
#endif

#define GABOR_VARIANCE_NOISE 0

#define randf() (rand() * (1.0/(RAND_MAX+1.0)))

#define random_uniform(a, b) rand_uniform(a, b)
#define our_select(a, b, c) ((a) ? (b): (c))

inline double square(double x) { return x*x; }

vector<double> read_file_doubles(const string &filename);

void our_srand(int seed);
double rand_uniform(double a, double b);

/** Convert vector to string, mimicking C++11 to_string(double) function. */
string to_string(const VectorXd &a, int precision=DEFAULT_PRECISION);

/** Convert matrix to string, mimicking C++11 to_string(double) function. */
string to_string(const MatrixXd &a, int precision=DEFAULT_PRECISION);

string to_string(const string &a);

/** Return true iff string a starts with string b. */
bool startswith(const string &a, const string &b);

/** Convert vector to comma-separated string surrounded by brackets. */
template<class T>
string to_string(const vector<T> &a) {
    string ans("[");
    for (int i = 0; i < (int) a.size(); i++) {
        ans += to_string(a[i]);
        if (i < (int) (a.size()-1)) { ans += ", "; }
    }
    ans += "]";
    return ans;
}

/** Helper function: return ith canonical basis vector in ndims dimensions */
VectorXd basis_vector(int ndims, int i);

/** Helper function: len = len(orig) = len(cum), and is the C implementation on numpy.cumsum. */
void cumsum(const VectorXd &orig, VectorXd &cum, int len);

/** Split string based on character delimiter. */
vector<string> split(const string &s, char delim);

/** Get value from map associated with given key, or if missing, get default value. */
template <typename K, typename V>
V get_default(const map <K, V> &m, const K &key, const V &defval) {
   typename std::map<K,V>::const_iterator it = m.find(key);
   if (it == m.end()) {
      return defval;
   } else {
      return it->second;
   }
}

void get_options(int argc, char *argv[], vector<string> &positional_args, map<string, string> &named_args);

template<class scalar>
vector<scalar> string_to_scalar_vector(const string &arg_value) {
    vector<scalar> ans;
    vector<string> ansL_str = split(arg_value, ',');
    for (int i = 0; i < (int) ansL_str.size(); i++) {
        ans.push_back(scalar(stof(ansL_str[i])));
    }
    return ans;
}

inline double our_sign(double x) {
    if (std::isnan(x)) { return NAN; }
    return (0.0 < x) - (x < 0.0);
}

inline double our_sign_down(double x) {
    return 2*(x > 0.0)-1;
}

inline double our_sign_up(double x) {
    return 2*(x >= 0.0)-1;
}

inline double atan_approx(double x) {
    double a = 16*x/M_PI;
    return (8*x/(3+sqrt(25+a*a)));
}

inline double fmod_round_down(double x, double y) {
    double ans = fmod(x, y);
    if (ans < 0) { ans += y; }
    return ans;
}

inline double fract(double x) {
    return x - floor(x);
}

inline double triangle_wave(double x) {
    const double T = 2.0;
    double xT_floor = floor(x/T);
    double TxT_floor = T*floor(x/T);
    double xf = x-T*xT_floor;
    return min(xf, T-xf);
}

inline void prng_bits(unsigned *st) {
    unsigned old = *st;
    *st *= 3039177861u;
    *st += (*st == old);
}

inline double prng_uniform_0_1(unsigned *st) {
    prng_bits( st );
    return (double) ((*st) / (float) 0xffffffffu);
}

inline double prng_uniform( unsigned * st, double min, double max ) {
    return min + prng_uniform_0_1( st ) * ( max - min );
}

inline unsigned prng_poisson(unsigned *st, double mean) {
    double g = exp(-mean);
    unsigned em = 0;
    double t = prng_uniform_0_1(st);
    while (g < t) {
        em += 1;
        t *= prng_uniform_0_1(st);
    }
    return em;
}

inline double gabor(double K, double a, double F_0, double omega_0, double x, double y) {
    double gau = K * exp(-M_PI*a*a*(x*x+y*y));
    double sinusoid = cos(2.0*M_PI*F_0*(x*cos(omega_0)+y*sin(omega_0)));
    return gau * sinusoid;
}

inline void filtered_gabor_param(double *K, double *a, double *F_0, double *omega_0, double tx, double ty) {
    //double sigmasq = GAUSS_SIGMA_FACTOR * GAUSS_SIGMA_FACTOR;
    double sigmasq = 1.0;
    double a2 = (*a)*(*a);
    double sigma_G = a2 / (2.0 * M_PI);
    double sigma_FG_x = a2 / (4.0*M_PI*M_PI*sigmasq*tx*a2 + 2.0*M_PI);
    double sigma_FG_y = a2 / (4.0*M_PI*M_PI*sigmasq*ty*a2 + 2.0*M_PI);
    double sigma_x = sigma_G + 1.0 / (4.0*M_PI*M_PI*sigmasq*tx);
    double sigma_y = sigma_G + 1.0 / (4.0*M_PI*M_PI*sigmasq*ty);
    double mu_G_x = (*F_0) * cos(*omega_0);
    double mu_G_y = (*F_0) * sin(*omega_0);
    double mu_FG_x = sigma_FG_x / sigma_G * mu_G_x;
    double mu_FG_y = sigma_FG_y / sigma_G * mu_G_y;
    
    double apsq = 2.0 * M_PI * sqrt(sigma_FG_x * sigma_FG_y);
    
    double ap = sqrt(apsq);
    double Kp = (*K)*apsq/(a2)*exp(-0.5*(mu_G_x*mu_G_x/sigma_x+mu_G_y*mu_G_y/sigma_y));
    double F_0p = sqrt(mu_FG_x * mu_FG_x + mu_FG_y * mu_FG_y);
    double omega_0p = atan2(mu_FG_y, mu_FG_x);
    
    *a = ap;
    *K = Kp;
    *F_0 = F_0p;
    *omega_0 = omega_0p;
}

inline double gabor_noise_impl(double K, double a, double F_0, double omega_0, double impulses, int period, double x, double y) {
    
    double radius = 0.9765097024771845 / a;    /* Magic constant is sqrt(-log(0.05) / pi). */
    x /= radius;
    y /= radius;
    double frac_x = fract(x);
    double frac_y = fract(y);
    int i = (int) (x - frac_x);
    int j = (int) (y - frac_y);
    double noise = 0.0;
    
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            unsigned s = (unsigned) (((j + dj) % period) * period + (i + di) % period);
            double number_of_impulses_per_cell = impulses / M_PI;
            unsigned number_of_impulses = prng_poisson(&s, number_of_impulses_per_cell);
            for (int k = 0; k < number_of_impulses; k++) {
                double x_k = prng_uniform_0_1(&s);
                double y_k = prng_uniform_0_1(&s);
                double w_k = prng_uniform(&s, -1.0, 1.0);
                double omega_0_k = prng_uniform(&s, 0.0, 2.0*M_PI);
                double x_k_x = frac_x - di - x_k;
                double y_k_y = frac_y - dj - y_k;
                if ((x_k_x * x_k_x) + (y_k_y * y_k_y) < 1.0) {
                    // isotropic
                    // assume independence for each gabor()
                    noise += w_k * gabor(K, a, F_0, omega_0_k, x_k_x * radius, y_k_y * radius); // isotropic
                    //noise += w_k * gabor(K, a, F_0, omega_0, x_k_x * radius, y_k_y * radius); // anisotropic
                }
            }
        }
    }
    return noise;
}

#define MINUS_LOG_0_05   2.995732273553991          /* -log(0.05) */

inline double gabor_noise(double K, double a, double F_0, double omega_0, double impulses, int period, float x, float y) {
    double var = (K*K) / (4.0*a*a) * (1.0 + exp(-2.0*M_PI*F_0*F_0 / (a*a)));
    var *= impulses / (3.0 * (MINUS_LOG_0_05 / (a * a)));
    double scale = 3.0 * sqrt(var);
    
    return gabor_noise_impl(K, a, F_0, omega_0, impulses, period, x, y) / scale;
    //return gabor_noise_impl(K, a, F_0, omega_0, impulses, period, x, y);
}

inline double filtered_gabor_noise(double K, double a, double F_0, double omega_0, double impulses, int period, double x, double y, double tx, double ty, bool is_var=false) {
    // Replicating Dorn's code
    //filtered_gabor_param(&K, &a, &F_0, &omega_0, tx, ty);
    double var = (K*K) / (4.0*a*a) * (1.0 + exp((-2.0*M_PI*M_PI)*F_0*F_0 / (a*a)));
    var *= 1.0 / 3.0 * impulses / (MINUS_LOG_0_05 / (a*a));
    double scale = 3.0 * sqrt(var);
    
    filtered_gabor_param(&K, &a, &F_0, &omega_0, tx, ty);
    if (!is_var) {
        return gabor_noise_impl(K, a, F_0, omega_0, impulses, period, x, y) / scale;
        //return gabor_noise_impl(K, a, F_0, omega_0, impulses, period, x, y);
    }
    else {
        double new_var = (K*K) / (4.0*a*a) * (1.0 + exp(-2.0*M_PI*M_PI*F_0*F_0 / (a*a)));
        new_var *= 1.0 / 3.0 * impulses / (MINUS_LOG_0_05 / (a*a));
        return new_var / (scale * scale);
    }
}

inline double filtered_gabor_noise_variance(double K, double a, double F_0, double omega_0, double impulses, int period, double x, double y, double tx, double ty) {
    return filtered_gabor_noise(K, a, F_0, omega_0, impulses, period, x, y, tx, ty, true);
}

#endif
