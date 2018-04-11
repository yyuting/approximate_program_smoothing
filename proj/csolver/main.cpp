
#include "problems.h"
#include "check_compiler.h"

#define USE_ODE 0

int main(int argc, char *argv[]) {
    vector<string> positional_args;
    map<string, string> named_args;
    get_options(argc, argv, positional_args, named_args);
    
    argc--;
    argv++;
    
    if (positional_args.size() == 0) {
        fprintf(stderr, "main command\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Commands:\n");
        fprintf(stderr, "  check_compiler        ---   Run compiler problem (requires --ndims argument)\n");
        fprintf(stderr, "              --samples n:   Use n samples to estimate g from f.\n");
        fprintf(stderr, "              --ndims n:     Specify which dimension the comparison is given.\n");
        fprintf(stderr, "              --error nerror: Estimate error by using 'nerror' samples (use nerror of -1 for default).\n");
        fprintf(stderr, "              --ground nground: When --error supplied, set number of samples to estimate ground truth.\n");
        fprintf(stderr, "              --seed seed:    Seed for RNG for error estimation.\n");
        fprintf(stderr, "              --print_rho b:  If b=1, print correlation coefficients (rho)\n");
        fprintf(stderr, "              --render w,h:   Render with given viewport size\n");
        fprintf(stderr, "              --render_t t:   Times for rendering: a single float or a comma-separated list of floats\n");
        fprintf(stderr, "              --render_sigma sx,sy,st:   Sigma for rendering for spatial (x, y) and time (t)\n");
        fprintf(stderr, "              --outdir s:     Use given output directory (string s should end without a slash).\n");
        fprintf(stderr, "              --gname s:      Optional prefix for g (smoothed) output files, default is g\n");
        fprintf(stderr, "              --do_f b:       If 1, compute f. If 0, do not compute f (default 1).\n");
        fprintf(stderr, "              --render_index i: Start index for rendered frames (default 0).\n");
        fprintf(stderr, "              --skip_save b:  If 1, skip saving image when rendering (default 0).\n");
        fprintf(stderr, "              --shader_only b: If 1, use C++ geometry wrapper (default 0).\n");
        fprintf(stderr, "              --geometry_name s: Geometry wrapper name.\n");
        fprintf(stderr, "              --nfeatures n:  Number of features needed for shader.\n");
        fprintf(stderr, "              --camera_path n: Specify camera_path.\n");
        fprintf(stderr, "              --min_time t1:  Minimum time value (for error estimates of shaders).\n");
        fprintf(stderr, "              --max_time t2:  Maximum time value (for error estimates of shaders).\n");
        fprintf(stderr, "              --timeout t:    Use timeout t seconds for estimating running times of f, g.\n");
        fprintf(stderr, "              --gt_file s:    Store and/or load ground truth error samples from filename s\n");
        fprintf(stderr, "              --denoise b:    If 1, denoise g using certain parameters.\n");
        fprintf(stderr, "              --patch_size n: Patch size in denoising.\n");
        fprintf(stderr, "              --search_size n: Search size in denoising.\n");
        fprintf(stderr, "              --denoise_param h: Parameter for denoising.\n");
        fprintf(stderr, "              --grounddir s:  Directory used to find ground truth image location.\n");
        fprintf(stderr, "              --is_gt b:      Specify whether computing ground truth.\n");
        exit(1);
    }
    
    /* Parse some common options. */
    
    string outdir(get_default(named_args, string("outdir"), string(".")));
    string gname(get_default(named_args, string("gname"), string("g")));
    string grounddir(get_default(named_args, string("grounddir"), string(".")));
    
    double timeout = 0.2;
    int verbose = stoi(get_default(named_args, string("verbose"), string("0")));
    bool do_f = bool(stoi(get_default(named_args, string("do_f"), string("1"))));
    bool skip_save = bool(stoi(get_default(named_args, string("skip_save"), string("0"))));
    bool shader_only = bool(stoi(get_default(named_args, string("shader_only"), string("0"))));
    bool is_gt = bool(stoi(get_default(named_args, string("is_gt"), string("0"))));
    string gt_file(get_default(named_args, string("gt_file"), string("")));

    bool denoise = bool(stoi(get_default(named_args, string("denoise"), string("0"))));
    int patch_size = stoi(get_default(named_args, string("patch_size"), string("3")));
    int search_size = stoi(get_default(named_args, string("search_size"), string("10")));
    double denoise_param = stod(get_default(named_args, string("denoise_param"), string("30.0")));
    
    string geometry_name(get_default(named_args, string("geometry_name"), string("plane")));
    int nfeatures = stoi(get_default(named_args, string("nfeatures"), string("0")));
    int camera_path = stoi(get_default(named_args, string("camera_path"), string("1")));

    double min_time = stod(get_default(named_args, string("min_time"), string("0.0")));
    double max_time = stod(get_default(named_args, string("max_time"), string("0.0")));

    int print_rho = stoi(get_default(named_args, string("print_rho"), string("0")));
    int nerror = stoi(get_default(named_args, string("error"), string("-2")));
    int nground = stoi(get_default(named_args, string("ground"), string("-1")));
    int seed = stoi(get_default(named_args, string("seed"), string("1")));
    
    int nsamples;
    if (named_args.count("samples")) {
        nsamples = stoi(named_args["samples"]);
    }
    
    timeout = stod(get_default(named_args, string("timeout"), to_string(timeout))); //1.0; //0.002;

    vector<int> render_viewport({-1, -1});
    vector<double> render_sigma({0.5, 0.5, 0.01});
    vector<double> render_t({0.0}); //1.0; //0.002;
    int render_index(stoi(get_default(named_args, string("render_index"), to_string("0"))));

    if (named_args.count("render")) {
        render_viewport = string_to_scalar_vector<int>(named_args["render"]);
    }

    if (named_args.count("render_sigma")) {
        render_sigma = string_to_scalar_vector<double>(named_args["render_sigma"]);
    }

    if (named_args.count("render_t")) {
        render_t = string_to_scalar_vector<double>(named_args["render_t"]);
    }
    
    vector<int> ndimsL({10000, 2000, 1000, 100, 20, 10, 4});
    
    if (named_args.count("ndims")) {
        ndimsL = string_to_scalar_vector<int>(named_args["ndims"]);
    }

    check_compiler(named_args, ndimsL, 1, 0, nerror, nground, seed, nsamples, print_rho, render_viewport, render_sigma, render_t, 1, outdir, do_f, gname, render_index, skip_save, shader_only, geometry_name, nfeatures, camera_path, min_time, max_time, 0, 0, 0, timeout, gt_file, denoise, patch_size, search_size, denoise_param, string(""), grounddir, is_gt, 0);
    
    return 0;
}
