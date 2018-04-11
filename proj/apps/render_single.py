
from render_util import *
import importlib
import time
import sys
import json

def render_single(base_dir, shader_name, geometry, normal_map, args, time_error=False, get_objective=False, use_objective=None, verbose=True, nframes=1, check_kw={}, outdir=None, render_kw={}, end_t=None, check_keys=False):
    """
    Render a single shader.

    Arguments:
      - get_objective: if True, then just return the objective function (an Expr instance).
      - use_objective: if True, then use the given objective in place of generating one.
      - nframes:       number of frames to render (use 1 for fast test, None for default)
    """
    end_t   = end_t if end_t is not None else 1.0    # End time of animation sequence

    T0 = time.time()
    if verbose:
        print('Rendering %s' % shader_name, 'normal_map=', normal_map)
    kw = dict(render_kw)

    if '--no-ground-truth' in args:
        kw['ground_truth_samples'] = 1
    if '--novel-camera-view' in args:
        kw['num_cameras'] = 2
    else:
        kw['num_cameras'] = 1
    if '--camera-path' in args:
        try:
            camera_path_ind = args.index('--camera-path')
            kw['specific_camera_path'] = int(args[camera_path_ind+1])
        except:
            pass

    m = importlib.import_module(shader_name)
    shaders = m.shaders
    kw['approx_mode'] = 'gaussian'

    check_kw = dict(check_kw)
    if time_error or '--time-error' in args:
        check_kw['time_error'] = True
        check_kw['nerror'] = 5000
        check_kw['nground'] = 1000
        check_kw['skip_save'] = True
        nframes = 1
        kw['ground_truth_samples'] = 0
    if not verbose:
        check_kw['print_command'] = False

    ans = render_any_shaders(shaders, default_render_size, use_triangle_wave=False, base_dir=base_dir, is_color=m.is_color, end_t=end_t, normal_map=normal_map, geometry=geometry, nframes=nframes, check_kw=check_kw, get_objective=get_objective, use_objective=use_objective, verbose=verbose, outdir=outdir, **kw)
    T1 = time.time()

    subdir = get_shader_dirname(base_dir, shader_name, normal_map, geometry, render_prefix=True)
    with open(os.path.join(subdir, 'render_command.txt'), 'wt') as f:
        f.write('python render_single.py %s %s %s %s' % (base_dir, shader_name, geometry, normal_map))
    with open(os.path.join(subdir, 'render_time.txt'), 'wt') as f:
        f.write(str(T1-T0))
    while isinstance(ans, list):
        assert len(ans) == 1
        ans = ans[0]
    if isinstance(ans, dict):
        save_dir = subdir
        if outdir is not None:
            save_dir = outdir
        with open(os.path.join(save_dir, 'render_info.json'), 'wt') as f:
            f.write(json.dumps(ans))

    if check_keys:
        check_single_keys(ans)
    return ans

def check_single_keys(ret_d):
    """
    Check that required keys are in the returned dictionary from render_single().
    """
    keys = ['time_f', 'time_g', 'error_f', 'error_g']
    for key in keys:
        if key not in ret_d:
            raise ValueError('missing key: %s' % key)

def main():
    args = sys.argv[1:]
    if len(args) < 4:
        print('python render_single.py base_dir shadername geometry normal_map [--no-ground-truth] [--novel-camera-view] [--time-error] [--camera-path i]')
        print('  Renders a single shader by name, with specified geometry and normal map.')
        sys.exit(1)

    (base_dir, shader_name, geometry, normal_map) = args[:4]

    render_single(base_dir, shader_name, geometry, normal_map, args)

if __name__ == '__main__':
    main()
