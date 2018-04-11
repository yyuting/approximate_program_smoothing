
#include "util.h"
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <ios>
#include <random>

std::mt19937 our_generator(1031);
std::uniform_real_distribution<double> our_dis(0.0, 1.0);

void our_srand(int seed) {
    our_generator.seed(seed);
}

double rand_uniform(double a, double b) {
    double t = our_dis(our_generator);
    return a + (b-a) * t;
}

string to_string(double x, int precision) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << x;
    return ss.str();
}

string to_string(const VectorXd &a, int precision) {
    string ans("[");
    for (int i = 0; i < (int) a.size(); i++) {
        ans += to_string(a[i], precision);
        if (i < ((int) a.size()) - 1) { ans += ", "; }
    }
    ans += "]";
    return ans;
}

string to_string(const MatrixXd &a, int precision) {
    string ans("[");
    for (int r = 0; r < a.rows(); r++) {
        ans += (r == 0 ? "[": " ");
        for (int c = 0; c < a.cols(); c++) {
            ans += to_string(a(r, c), precision);
            if (c < a.cols() - 1) { ans += ", "; }
        }
        if (r < a.rows() - 1) { ans += "]\n"; }
    }
    ans += "]";
    return ans;
}

string to_string(const string &a) {
    return a;
}

bool startswith(const string &a, const string &b) {
    return a.compare(0, b.length(), b) == 0;
}

/** Helper function: return ith canonical basis vector in ndims dimensions */
VectorXd basis_vector(int ndims, int i) {
    VectorXd ans(VectorXd::Constant(ndims, 0.0));
    ans[i] = 1.0;
    return ans;
}
    
/** Helper function: cum is of lenght len(orig)+2, and is the C implementation on numpy.cumsum. */
void cumsum(const VectorXd &orig, VectorXd &cum, int len) {
    cum[0] = orig[0];
    for (int i = 1; i < len; i++) {
        cum[i] = cum[i-1] + orig[i];
    }
}

vector<string> split(const string &s, char delim) {
    vector<string> L;
    stringstream ss;
    ss.str(s);
    string item;
    while (getline(ss, item, delim)) {
        L.push_back(item);
    }
    return L;
}

void get_options(int argc, char *argv[], vector<string> &positional_args, map<string, string> &named_args) {
    argc--;
    argv++;
    for (int i = 0; i < argc; i++) {
        string s(argv[i]);
        if (s.size() >= 2 && s[0] == '-' && s[1] == '-' && i + 1 < argc) {
            named_args[s.substr(2)] = argv[i+1];
            i++;
            continue;
        } else {
            positional_args.push_back(argv[i]);
        }
    }
}

vector<double> read_file_doubles(const string &filename) {
    FILE *f = fopen(filename.c_str(), "rt");
    if (!f) { fprintf(stderr, "could not read %s\n", filename.c_str()); exit(1); }
    vector<double> ans;
    while (1) {
        double value = 0.0;
        if (fscanf(f, "%lf", &value) != 1) {
            break;
        }
        ans.push_back(value);
        if (feof(f)) { break; }
    }
    fclose(f);
    return ans;
}
