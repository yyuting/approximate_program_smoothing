#ifndef _check_compiler_h
#define _check_compiler_h

class CompilerProblem;
Problem *create_compiler_problem(int ndims, int nsamples, CompilerProblem **compiler_problem=NULL);
Problem *create_geometry_problem(int ndims, int nsamples, int nfeatures, const string geometry_name, int height, int width, double min_time, double max_time, int camera_path);

void check_compiler(map<string, string> named_args, const vector<int> &ndimsL, int repeats, int check_g, int nerror, int nground, int seed, int nsamples, int print_rho, const vector<int> &render_viewport, const vector<double> &render_sigma, const vector<double> &render_t, int g_samples, const string &outdir, bool do_f, const string gname, int render_index, bool skip_save, bool shader_only, const string geometry_name, int nfeatures, int camera_path, double min_time, double max_time, bool spatially_varying, bool gradient, bool adjust_var, double timeout, const string &ground_truth_filename, bool denoise, int patch_size, int search_size, double denoise_param, const string &load_adjust_var, const string &grounddir, bool is_gt, bool allow_var_error);

void compare_g(vector<double> valueL, int ndims, vector<double>sigmaL, Problem *orig_problem, Problem *sampled_problem);

#endif

