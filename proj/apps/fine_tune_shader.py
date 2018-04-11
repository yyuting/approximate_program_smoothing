
from render_util import *
from test_util import *
from render_single import render_single
import time
import util
import shutil
import dill
import pprint
import traceback

# Parameters controlling which fine tunings to make
do_nlm_denoise  = True

fine_tune_parallel = True           # Whether to parallelize fine-tuning
ignore_fine_tune_errors = True

ymax       = 0.2
time_units = '[ms]'

def rescale_time(t):
    return t * default_render_size[0] * default_render_size[1] * 1000.0

def nlm_denoise(base_dir, shader_name, geometry, normal_map, args, out_tune_dir, indiv):
    """
    Given existing individual 'indiv', return list of individuals with adjusted options for NLM denoising.
    
    Note that the render command in tune_shader.py (in function tune_shader) must also be modified to properly
    render the returned modified individuals.
    """
    print('individual before nlm denoising:')
    pprint.pprint(indiv)
    ans_indiv = []
    for h in [5.0, 10.0, 15.0, 20.0]:
        extra_cmd = ' --denoise 1 --denoise_param %f'%(h)
        try:
            ans = render_single(base_dir, shader_name, geometry, normal_map, args, time_error=True, log_intermediates=False, nframes=1, use_objective=indiv.e, check_kw={'extra_args': extra_cmd}, render_kw={'denoise': True, 'denoise_h': h}, check_keys=True)
        except:
            if ignore_fine_tune_errors:
                print('Warning: error was raised during fine-tuning')
                traceback.print_exc()
                return []
            else:
                raise
        if hasattr(indiv, 'msaa_samples'):
            ans['msaa_samples'] = indiv.msaa_samples
        print('denoise ans:', ans)
        ans_indiv.append(tuner.Individual(indiv.e, ans, 'denoise'))
    return ans_indiv

def load_pop_from(out_tune_dir, fail_on_error=True):
    orig_tune = os.path.join(os.path.split(out_tune_dir)[0], 'orig_tune')
    orig_html = os.path.join(os.path.split(out_tune_dir)[0], 'orig_html')
    try:
        pop = tuner.load_from_disk(orig_tune)
    except:
        try:
            pop = tuner.load_from_disk(out_tune_dir)
        except:
            if fail_on_error:
                raise
            else:
                print('Warning: could not load population from %s' % out_tune_dir)
                return
    pop = pop[tuner.ours_label]

    return pop

def fine_tune_shader(base_dir, shader_name, geometry, normal_map, args, out_tune_dir):
    import tune_shader
    if fine_tune_parallel:
        pool = tuner.ProcessingPool(tuner.nproc)
        print('Parallel fine-tuning with %d processes'%tuner.nproc)
    
    allow_orig = not '--no-orig' in args
    
    # Get pareto population (list of tuner.Individual instances)
    orig_tune = os.path.join(os.path.split(out_tune_dir)[0], 'orig_tune')
    orig_html = os.path.join(os.path.split(out_tune_dir)[0], 'orig_html')
    cur_html = os.path.join(os.path.split(out_tune_dir)[0], 'html')
    
#    print('do_nlm_denoise:', do_nlm_denoise)
    
    pop = load_pop_from(out_tune_dir)
    print('Read population %d from first directory' % len(pop))
    
    if '--finetune-dirs' in args:
        finetune_dirs = args[args.index('--finetune-dirs')+1].split(',')
        print('Loading additional directories for fine-tuning:', finetune_dirs)
        for finetune_dir in finetune_dirs:
            finetune_dir = tune_shader.get_tune_dirname(finetune_dir, shader_name, normal_map, geometry)
            pop_p = load_pop_from(finetune_dir, False)
            if pop_p is not None:
                pop.extend(pop_p)
    print('Population size after reading all fine-tuning dirs:', len(pop))

    pareto = tuner.population_pareto(pop, individuals=True)
    print('Pareto size for fine-tuning:', len(pareto))
    
    # newly added variants doesn't have error_g_image attribute
    # remove error_g_image attribute in original variants
    # so pareto frontier can be selected normally later
    for indiv in pareto:
        if hasattr(indiv, 'error_g_image'):
            delattr(indiv, 'error_g_image')

    if not os.path.exists(out_tune_dir):
        print('tune directory does not exist: %s' % out_tune_dir)
        print('cannot finetune, exiting')
    
    if not os.path.exists(orig_tune):
        shutil.copytree(out_tune_dir, orig_tune)

    if not os.path.exists(orig_html) and os.path.exists(cur_html):
        shutil.copytree(cur_html, orig_html)
        
    if os.path.exists(os.path.join(out_tune_dir, 'all_image.dill')):
        os.remove(os.path.join(out_tune_dir, 'all_image.dill'))
    
    def add_variants(L):
        """
        Add list of program variants to resulting population (except element 0, which is considered to already be in the population).
        
        Return list of variants on Pareto frontier.
        """
        orig_len = len(result_pop)
        result_pop.extend(L[1:])
        pareto_ids = set([id(x) for x in tuner.population_pareto(result_pop, individuals=True)])
        return [x for x in L if id(x) in pareto_ids]

    def enhance_indivs(L, f):
        ans = []
        for indiv in L:
            indiv_variants = add_variants([indiv] + f(base_dir, shader_name, geometry, normal_map, args, out_tune_dir, indiv))
            ans.extend(indiv_variants)
        return ans

    def get_finetuned_individuals(indiv):
        indiv_variants = [indiv]
        
        global do_nlm_denoise
        if '--no-nlm' in args:
            do_nlm_denoise = False
        
        if do_nlm_denoise:
            has_mc = False
            for indiv_variant in indiv_variants:
                base_node = indiv_variant.e
                for node in base_node.all_approx_nodes():
                    if node.approx == APPROX_MC:
                        has_mc = True
                        break
                if hasattr(indiv_variant, 'msaa_samples'):
                    has_mc = True
            if has_mc:
                indiv_variants = enhance_indivs(indiv_variants, nlm_denoise)
        begin = []
        if allow_orig:
            begin = [indiv]
        return begin + indiv_variants

    result_pop = []         # Resulting individuals for a given process (no effort is made to sync this variable)

    map_func = map
    if fine_tune_parallel:
        map_func = pool.map

    finetuned = list(map_func(get_finetuned_individuals, pareto))
    finetuned = sum(finetuned, [])

    result_pop = finetuned
    
    print('Fine-tuned results:')
    pprint.pprint(finetuned)

    result_d = {tuner.ours_label: finetuned}
    s = dill.dumps(result_d)
    with open(os.path.join(out_tune_dir, 'all.dill'), 'wb') as f:
        f.write(s)

    xmax = tuner.plot_get_xmax(result_pop, max_slowdown=35.0, rescale_time=rescale_time)
    tuner.plot(out_tune_dir, result_d, ymax=ymax, generation=99998, xmax=xmax, rescale_time=rescale_time, time_units=time_units)
