
from render_util import *
from test_util import *
from render_single import render_single
import traceback
import importlib
import time
import util
import shutil
import dill
from fine_tune_shader import *

dynamic_rewrite       = True        # Dynamically rewrite function calls to use updated (e.g. bug fixed) in compiler
dynamic_rewrite_funcs = ['fract']   # Functions to dynamically rewrite (or None to rewrite all)

parallel_render     = True

def system(s):
    print(s)
    return os.system(s)

def get_tune_dirname(base_dir, shader_name, normal_map, geometry):
    orig_dir = base_dir.endswith('@')
    if orig_dir:
        base_dir = base_dir.rstrip('@')
    out_dir = get_shader_dirname(base_dir, shader_name, normal_map, geometry, render_prefix=True)
    if not orig_dir:
        out_tune_dir = os.path.join(out_dir, 'tune')
    else:
        out_tune_dir = os.path.join(out_dir, 'orig_tune')
    if not os.path.exists(out_tune_dir):
        os.makedirs(out_tune_dir)
    return out_tune_dir

def render_after(args):
    if not '--only' in args:
        render_cmd = 'python tune_shader.py render ' + ' '.join(args[1:5])
        if '--fast' in args or True:
            render_cmd += ' --fast'
        print(render_cmd)
        os.system(render_cmd)

def tune_shader(base_dir, shader_name, geometry, normal_map, is_short=False, args=(), out_tune_dir=None, render=False, pop=None, webpage=False):
    T0_tune = time.time()
    print_command       = True
    verbose             = False
    skip_check_failed   = True      # Should we skip failing checks?
    compare_modes       = False
    halt_on_non_finite  = False

    if is_short:
        population_size = 10
        generations = 5
    else:
        population_size = 40
        generations = 20

    #print(args)
    fast = False
    user_generations = False
    if '--generations' in args:
        generations = int(args[args.index('--generations')+1])
        user_generations = True
    if '--population-size' in args:
        population_size = int(args[args.index('--population-size')+1])
    if '--fast' in args:
        fast = True
    resume_dir = None
    if '--resume-dir' in args:
        resume_dir = args[args.index('--resume-dir')+1].split(',')
        resume_dir = ','.join([get_tune_dirname(resume_dir_part, shader_name, normal_map, geometry) for resume_dir_part in resume_dir])
    index = None
    if '--index' in args:
        index = [int(arg_val) for arg_val in args[args.index('--index')+1].split(',')]
    nframes = 1
    if '--nframes' in args:
        nframes = int(args[args.index('--nframes')+1])
    #print(population_size)
    finetune = ('--finetune' in args)

    msaa_only = False
    allowed_approx_modes = list(all_approx_modes)
    allowed_approx_rho_modes = None
    if '--ours-only' in args:
        allowed_approx_modes.remove(APPROX_MC)
    if '--mc-only' in args:
        allowed_approx_modes = [APPROX_MC]
        if not user_generations:
            generations = 0
        population_size = builtin_min(population_size, 5)
    if '--msaa-only' in args:
        msaa_only = True
    if '--dorn-only' in args:
        allowed_approx_modes = [APPROX_NONE, APPROX_DORN]
    if '--gaussian-only' in args:
        allowed_approx_modes = [APPROX_NONE, APPROX_GAUSSIAN]
    if '--no-mc' in args:
        if APPROX_MC in allowed_approx_modes:
            allowed_approx_modes.remove(APPROX_MC)
    if '--no-rho' in args:
        allowed_approx_rho_modes = APPROX_RHO_ZERO
    end_t = None
    if '--end-t' in args:
        end_t = float(args[args.index('--end-t')+1])
    time_limit = None
    if '--time-limit' in args:
        time_limit = float(args[args.index('--time-limit')+1])
    override_denoise_h = None
    if '--denoise-h' in args:
        override_denoise_h = float(args[args.index('--denoise-h')+1])
    from_tuned = False
    if '--tune-from-plane' in args or '--use-tuned' in args:
        from_tuned = True

    c = CompilerParams(compute_g=True)
    
    m = importlib.import_module(shader_name)
    is_color = m.is_color
    if is_color:
        c.constructor_code = [OUTPUT_ARRAY + '.resize(3);']
    
    render_dir = os.path.split(out_tune_dir)[0]
    if render:
        print('begin render', render_dir)
        #print(base_dir)
        #print(out_tune_dir)
        #print(render_dir)
        #sys.exit(1)
        for filename in glob.glob(os.path.join(render_dir, '*.png')):
            os.remove(filename)
        has_error_g_image = any([hasattr(indiv, 'error_g_image') for indiv in pop])
        if has_error_g_image:
            pareto = [ind for ind in range(len(pop)) if hasattr(pop[ind], 'error_g_image')]
        else:
            pareto = tuner.population_pareto(pop, indices=True)

        if parallel_render:
            pool = tuner.ProcessingPool(tuner.nproc)

        if index is not None:
#            assert index in pareto, 'expected index %d to be in pareto %r' % (index, pareto)
            pareto = index
            ground_files = glob.glob(os.path.join(render_dir, 'ground*.png'))
            for ground_file in ground_files:
                os.remove(ground_file)

        print('pareto:', pareto)
        print('population:', len(pop))

        def num_format(a, b):
            if nframes > 1:
                return '%05d%05d' % (a, b)
            return '%05d' % a

        for time_error in [True, False] if not fast else [False]:
            def render_pareto(pareto_idx_pair):
                our_id = util.machine_process_id()
                our_dir = os.path.join(render_dir, '.render_%s' % our_id)
                print('our_dir:', our_dir)
                if not os.path.exists(our_dir):
                    os.makedirs(our_dir)
                    
                (pareto_idx, i) = pareto_idx_pair
                print('rendering individual on pareto frontier: index %d, population size: %d' % (i, len(pop)))
                ground_truth_samples = None
                if pareto_idx != 0 and not time_error:
                    ground_truth_samples = 1
                if '--no-ground-truth' in args:
                    ground_truth_samples = 1
                msaa_samples = getattr(pop[i], 'msaa_samples', None)

                print_header('Individual Expr tree:')
                pprint.pprint(pop[i].e)
                
                denoise_h = getattr(pop[i], 'denoise_h', None)
                if override_denoise_h is not None:
                    denoise_h = override_denoise_h
                print('denoise in render:', denoise_h)
                
                if dynamic_rewrite:
                    objective = copy.deepcopy(pop[i].e)
                    for parent_p in objective.all_nodes():
                        for (child_i, child_p) in enumerate(parent_p.children):
                            if isinstance(child_p, Call): # and child_p.name == 'fract':
                                rewrite_found = child_p.name in globals() and (dynamic_rewrite_funcs is None or child_p.name in dynamic_rewrite_funcs)
                                if rewrite_found:
                                    print('dynamic rewrite: flagged %s node for replacement, found=%d' % (child_p.name, rewrite_found))
                                    #print('  children: ', child_p.children)
                                    replace_node = globals()[child_p.name](*child_p.children[1:])
                                    replace_node.set_approx_info(child_p.get_approx_info())
                                    #replace_node.set_approx_info(APPROX_NONE)
                                    parent_p.children[child_i] = replace_node
                else:
                    objective = copy.deepcopy(pop[i].e)
                denoise_cmd = ''
                denoise = False
                grounddir = None
                if denoise_h is not None:
                    denoise_cmd = ' --denoise 1 --denoise_param %f'%(denoise_h)
                    grounddir = os.path.join(render_dir, 'tune')

                ans = render_single(base_dir, shader_name, geometry, normal_map, args, time_error=time_error, get_objective=False, nframes=nframes, use_objective=objective, outdir=our_dir, render_kw={'ground_truth_samples': ground_truth_samples, 'denoise': denoise, 'denoise_h': denoise_h, 'grounddir': grounddir, 'msaa_samples': msaa_samples, 'from_tuned': from_tuned}, check_kw={'extra_args': denoise_cmd}, end_t=end_t)
                print('done calling render_single')
                if time_error or fast:
                    shutil.copyfile(os.path.join(our_dir, 'render_info.json'), os.path.join(render_dir, 'render_info%05d.json'%pareto_idx))
                if (not time_error) or fast:
                    for prefix in ['f', 'g', 'ground']:
                        for iframe in range(nframes):
                            filename = os.path.join(our_dir, prefix + '%05d.png'%iframe)
                            dest_filename = os.path.join(out_tune_dir, prefix + num_format(pareto_idx, iframe) + '.png')
                            shutil.copyfile(filename, dest_filename)
                            os.remove(filename)
                        #print('Temporarily wrote %s' % dest_filename)
                    with open(os.path.join(render_dir, 'pareto_index%05d.txt'%pareto_idx), 'wt') as pareto_index_f:
                        pareto_index_f.write(str(i) + '\n')
                shutil.rmtree(our_dir)


            if parallel_render and (not time_error or fast):
                pool.map(render_pareto, enumerate(pareto))
            else:
                res = [render_pareto(p) for p in enumerate(pareto)]

        for (pareto_idx, i) in enumerate(pareto):
            for prefix in ['f', 'g', 'ground']:
                for iframe in range(nframes):
                    filename = os.path.join(out_tune_dir, prefix + num_format(pareto_idx if prefix != 'ground' else 0, iframe) + '.png')
                    dest_filename = os.path.join(render_dir, prefix + num_format(pareto_idx, iframe) + '.png')
                    shutil.copyfile(filename, dest_filename)
                    if prefix != 'ground':
                        os.remove(filename)
                    print('Moved %s => %s' % (filename, dest_filename))
        T_render_only = time.time()

        print('Total time rendering: %f secs' % (time.time()-T0_tune))
    else:
        f = render_single(base_dir, shader_name, geometry, normal_map, args, time_error=True, get_objective=True)
    #print('objective from render_single:')
    #pprint.pprint(f)
#    f.set_approx_recurse(APPROX_REGULAR_STACK_LERP1)
#    f.approx = APPROX_NONE #GAUSSIAN
#    f.repair_consistency(copy.deepcopy(c))
#    print(check(f, c, time_error=True))

    if render or webpage:
        desc = 'Timings: fast' if fast else 'Timings: accurate'
        system('python collect_webpage.py %s %s "%s" --no-predict --subdir%s%s%s' % (render_dir, os.path.join(render_dir, 'html' if nframes == 1 else 'video'), desc, (' --video' if nframes > 1 else ''), (' --no-ground-truth' if '--no-ground-truth' in args else ''), (' --heatmap' if '--heatmap' in args else '')))
        return

    gt_file = os.path.join(out_tune_dir, 'gt.npy')
    def check_func(e, compiler_params):
        msaa_samples = compiler_params.msaa_samples
        if msaa_samples <= 1:
            msaa_samples = None
        ans = render_single(base_dir, shader_name, geometry, normal_map, args, time_error=True, use_objective=e, verbose=verbose, nframes=1, check_kw={'extra_args': ' --gt_file %s' % (os.path.abspath(gt_file))}, render_kw={'msaa_samples': msaa_samples})
        print('check_func, ans:', ans)
        return ans

    def check_samples_func(e, compiler_params, nsamples, precompute):
        raise NotImplementedError #return render_single(base_dir, shader_name, geometry, normal_map, args, time_error=False, use_objective=f) # TODO: implement this

    print('Tuning:')
    print('Population size:', population_size)
    print('Generations:', generations)
    tuner.tune(c, f, population_size=population_size, generations=generations, nsampled=5, print_command=print_command, verbose=verbose, check_func=check_func, check_samples_func=check_samples_func, allow_sampled=False, skip_check_failed=skip_check_failed, compare_modes=compare_modes, halt_on_non_finite=halt_on_non_finite, ymax=ymax, out_tune_dir=out_tune_dir, allowed_approx_modes=allowed_approx_modes, rescale_time=rescale_time, time_units=time_units, resume_from_dir=resume_dir, allowed_approx_rho_modes=allowed_approx_rho_modes, msaa_only=msaa_only, time_limit=time_limit)

    if finetune:
        cmd = 'python tune_shader.py finetune %s %s %s %s%s' % (base_dir, shader_name, geometry, normal_map, (' --no-adjust-var' if '--no-adjust-var' in args else '') + (' --no-nlm' if '--no-nlm' in args else ''))
        print(cmd)
        os.system(cmd)

    if not '--no-render' in args:
        render_after(args)

#    tuner.plot_from_disk()
#    tuner.tune(c, f, population_size=5, generations=5, nsampled=5, verbose=True)

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        print('python tune_shader.py command base_dir shadername geometry normal_map [optional_args]  -- Make a tuner run')
        print('Commands:')
        print('  short          --- Make a short tuner run (for testing), and do render (see below)')
        print('  full           --- Make a full tuner run, and do render (see below)')
        print('  replot         --- Recreate the plots from a previous tuner run (if commas in base_dir, make multiple plots)')
        print('                     The @ symbol can be placed at the end of a base_dir to use the orig_tune subdirectory instead.')
        print('  pareto         --- List the individuals on the Pareto frontier for a previous tuner run')
        print('  render         --- Render shader images for the Pareto frontier of a previous tuner run')
        print('  webpage        --- Generate html output in subdirectories based on previous rendering (use --nframes for video)')
        print('  stats base_dir --- Report statistics of different points on Pareto frontiers, for each subdir of base_dir')
        print('  finetune       --- Refine an existing Pareto frontier, by fine-tuning with NLM denoising and tuning of variances')
        print('                     The original tune is backed up to an orig_tune subdirectory (and html => orig_html)')
        print('')
        print('  Tunes a single shader by name (selects the best approximations at each Expr), with specified geometry and normal map.')
        print('Optional arguments (which must come last):')
        print('  --generations g       --  Specify generations')
        print('  --population-size p   --  Specify population size')
        print('  --ours-only           --  Use only gaussian, dorn, none')
        print('  --dorn-only           --  Use only dorn, none')
        print('  --mc-only             --  Use only Monte Carlo approx with fixed sample count for shader, and one Gaussian footprint')
        print('                            based on linear approx for ray-geometry intersection (in a sense Monte Carlo + our approx).')
        print('  --msaa-only           --  Use conventional MSAA')
        print('  --gaussian-only       --  Use only gaussian, none')
        print('  --only                --  If short or full, do not render shaders at the end (default is to render)')
        print('  --fast                --  For render, produce less accurate timings (for a faster preview of results)')
        print('  --no-mc               --  Disable Monte Carlo approximation')
        print('  --no-rho              --  Disable rho approximations')
        print('  --no-render           --  Disable rendering after tuning')
        print('  --resume-dir d        --  Resume by reading population from given base dir (can be comma-separated for multiple dirs)')
        print('  --index i             --  Select a given individual by index for rendering (called "Pareto index" in the html output)')
        print('                            Here i can also be a comma-separated list')
        print('  --nframes n           --  Use the given number of frames for video, rather than the default of 1 (still image only)')
        print('  --end-t t             --  End time for video')
        print('  --finetune            --  For the commands short or full, fine tune before rendering')
        print('  --no-nlm              --  For fine-tuning, disable NLM denoising')
        print('  --no-adjust-var       --  For fine-tuning, disable adjusting var')
        print('  --finetune-dirs d     --  Additional directories (comma-separated) for reading from for fine-tuning')
        print('  --time-limit s        --  Use s seconds time-out for each program (compile+run)')
        print('  --no-orig             --  Disable using original program for fine-tuning')
        print('  --merge d             --  Merge the given base_dir results into the Pareto frontier')
        print('  --legend-text s       --  Quoted comma-separated list of text for legend')
        print('  --no-ground-truth     --  In render or webpage modes, do not compute/display ground truth video')
        print('  --denoise-h h         --  Always denoise using given denoising parameter (e.g. 15)')
        print('  --min-f-time          --  Use (and print) minimum time for f when creating plot')
        print('  --tune-from-plane     --  Adjust tuning result from plane to other geometry')
        print('  --use-tuned geometry  --  Used variant tuned from another geometry')
        print('  --no-parallel-render   --  No parallel render for more accurate timing')
        sys.exit(1)
    command = args[0]
    
    if command != 'stats':
        (base_dir, shader_name, geometry, normal_map) = args[1:5]
        out_tune_dir = get_tune_dirname(base_dir, shader_name, normal_map, geometry)
    else:
        base_dir = args[1]
        
    if '--no-parallel-render' in args:
        print("no parallel render")
        parallel_render = False
        i = args.index('--no-parallel-render')
        args = args[:i] + args[i+1:]
    
    legend_text = None
    if '--legend-text' in args:
        i = args.index('--legend-text')
        legend_text = args[i+1]
        args = args[:i] + args[i+2:]

    if command == 'replot':
        base_dir_L = base_dir.split(',')
        out_tune_dir_L = [get_tune_dirname(base_dir, shader_name, normal_map, geometry) for base_dir in base_dir_L]
        merge_dir = None
        if '--merge' in args:
            merge_dir = args[args.index('--merge')+1]
            merge_dir = get_tune_dirname(merge_dir, shader_name, normal_map, geometry)
        min_f_time = '--min-f-time' in args
        tuner.multi_plot_from_disk(out_tune_dir_L, ymax=ymax, rescale_time=rescale_time, time_units=time_units, merge_dir=merge_dir, legend_text=legend_text, min_f_time=min_f_time)
        tuner.multi_plot_from_disk(out_tune_dir_L, ymax=ymax, rescale_time=rescale_time, time_units=time_units, merge_dir=merge_dir, legend_text='', generation=99997, min_f_time=min_f_time)
    elif command == 'short':
        is_short=True
        tune_shader(base_dir, shader_name, geometry, normal_map, is_short, args, out_tune_dir)
        #if not '--finetune' in args:
        #    render_after(args)
    elif command == 'webpage':
        is_short=True
        tune_shader(base_dir, shader_name, geometry, normal_map, is_short, args, out_tune_dir, webpage=True)
    elif command == 'full':
        is_short=False
        tune_shader(base_dir, shader_name, geometry, normal_map, is_short, args, out_tune_dir)
        #if not '--finetune' in args:
        #    render_after(args)
    elif command == 'pareto':
        pop = tuner.load_from_disk(out_tune_dir)[tuner.ours_label]
        tuner.print_pareto(pop)
    elif command == 'render':
        is_short=False
        if '--use-tuned' in args:
            i = args.index('--use-tuned')
            tuned_geometry = args[i+1]
            #args = args[:i] + args[i+2:]
            tuned_dir = get_tune_dirname(base_dir, shader_name, normal_map, tuned_geometry)
            if os.path.exists(out_tune_dir):
                shutil.rmtree(out_tune_dir)
            shutil.copytree(tuned_dir, out_tune_dir)
        else:
            tuned_dir = out_tune_dir
        pop = tuner.load_from_disk(tuned_dir)[tuner.ours_label]
        tune_shader(base_dir, shader_name, geometry, normal_map, is_short, args, out_tune_dir, render=True, pop=pop)
    elif command == 'finetune':
        T0 = time.time()
        fine_tune_shader(base_dir, shader_name, geometry, normal_map, args, out_tune_dir)
        if not '--no-render' in args:
            render_after(args)
        print('Total time fine-tuning and rendering:', time.time()-T0)
    elif command == 'stats':
        index = None
        if '--index' in args:
            index = [int(arg_val) for arg_val in args[args.index('--index')+1].split(',')][0]
        print('selected index:', index)

        #stats_modes = all_approx_modes
        stats_modes = [APPROX_MC, APPROX_NONE, APPROX_DORN, APPROX_GAUSSIAN, 'rho_zero', 'rho_constant', 'rho_gradient', 'mc2', 'mc4', 'mc8', 'mc16', 'mc32']
        warning_str = []
        
        def get_stats(indiv_L):
            ans = {}
            for mode in stats_modes:
                ans.setdefault(mode, 0.0)
            for indiv in indiv_L:
                nodes = indiv.e.all_approx_nodes()
                w = 1.0/(len(nodes)*len(indiv_L))
                for node in nodes:
                    approx = node.approx
                    ans[approx] += w
        
                    if approx == APPROX_MC:
                        n_mc = node.approx_mc_samples
                        if n_mc in [2, 4, 8, 16, 32]:
                            ans['mc' + str(n_mc)] += w
                        else:
                            print('warning: unknown sample count %d' % n_mc)

                    if node.approx == APPROX_GAUSSIAN:
                        approx_rho = node.approx_rho
                        rho_key = 'rho_' + approx_rho
                        if rho_key in ans:
                            ans[rho_key] += w
                        else:
                            warning_str.append('no key %s' % rho_key + '\n')
            return ans
        
        def print_stats(name, name_b, d):
            L = ['%-40s' % name, '%-10s' % name_b]
            for key in sorted(d.keys()):
                L.append('%s: %5.3f%%' % (key, d[key]*100.0))
            print(', '.join(L))
        
        stats_name = ['left', 'middle', 'right', 'all']
        
        all_selected = []
        for subdir in sorted(glob.glob(os.path.join(base_dir, '*'))):
            all_file = os.path.join(subdir, 'tune/all.dill')
            if os.path.isdir(subdir) and os.path.exists(all_file):
                try:
                    all_pop = dill.loads(open(all_file, 'rb').read())
                except:
                    print('warning: could not read %s, skipping' % all_file)
                    continue
                all_pop = all_pop[tuner.ours_label]
                #print('loaded', subdir, len(all_pop))
                if len(all_pop) == 0:
                    continue
                I = tuner.population_pareto(all_pop, indices=True)
                
                target_idx_left = I[0]
                target_idx_right = I[-1]
                print('choosing index:', index, len(all_pop), index is not None and index < len(all_pop))
                if index is not None and index < len(all_pop):
                    target_idx_left = target_idx_right = index
                
                left = all_pop[target_idx_left]
                right = all_pop[target_idx_right]
                middle_t_target = (left.time_g + right.time_g)*0.5
                middle = builtin_min((abs(all_pop[i].time_g - middle_t_target), all_pop[i]) for i in I)[1]
                all_s = [all_pop[i] for i in I]

                selected = [[left], [middle], [right], all_s]
                
                sub_name = os.path.split(subdir)[1]
                for j in range(len(stats_name)):
                    print_stats(sub_name, stats_name[j], get_stats(selected[j]))
                
                all_selected.append(selected)

        for j in range(len(stats_name)):
            print_stats('all shaders', stats_name[j], get_stats(sum([s[j] for s in all_selected], [])))
        
        for warn_msg in sorted(set(warning_str)):
            print('Warning:', warn_msg)
    else:
        raise ValueError('unknown command %s' % args[0])

if __name__ == '__main__':
    main()
