
"""
Utility functions for rendering.
"""

import math
import pprint
import sys; sys.path += ['../compiler']
import time
import hashlib
import os, os.path
import shutil
import copy
import multiprocessing
from process_util import *
from compiler import *

default_render_size = (320, 480)
default_is_color = True

geometry_transfer = True

def unique_id():
    return hashlib.md5(str(time.time()).encode('utf-8')).hexdigest()

def vec(prefix, point):
    """
    Declare several Var() instances and return them as a numpy array.
    
     - prefix: string prefix for the variable names.
     - point:  array-like object storing the point (e.g. containing floats or Exprs), up to 4D
    """
    ans = numpy.zeros(len(point), dtype='object')
    for i in range(len(point)):
        ans[i] = Var(prefix + '_' + 'xyzw'[i], point[i])
    return ans
    
def set_channels(v):
    assert len(v) == 3
    v[0].channel = 'r'
    v[1].channel = 'g'
    v[2].channel = 'b'

def vec_color(prefix, point):
    assert len(point) == 3
    ans = vec(prefix, point)
    set_channels(ans)
    return ans
    
    
def vec_long(prefix, point):
    """
    Declare several Var() instances and return them as a numpy array.
    
     - prefix: string prefix for the variable names.
     - point:  array-like object storing the point (e.g. containing floats or Exprs), up to 4D
    """
    ans = numpy.zeros(len(point), dtype='object')
    for i in range(len(point)):
        ans[i] = Var(prefix + '_' + str(i), point[i])
    return ans

def normalize(point):
    """
    Normalize a given numpy array of constants or Exprs, returning a new array with the resulting normalized Exprs.
    """
    prefix = unique_id()
    if isinstance(point[0], Var):
        prefix = point[0].name
        if prefix.endswith('_x'):
            prefix = prefix[:-2]
        prefix = prefix + '_normalized'
    var_norm = Var(prefix + '_norm', sqrt(sum(x**2 for x in point)))
    return vec(prefix, point / var_norm)

def normalize_const(v):
    """
    Normalize a numpy array of floats or doubles.
    """
    return v / numpy.linalg.norm(v)

def dot(u, v):
    """
    Take dot product between two arrays.
    """
    return sum(u[i]*v[i] for i in range(len(u)))
    
def det2x2(A):
    """
    Calculate determinant of a 2x2 matrix
    """
    return A[0,0]*A[1,1] - A[0,1]*A[1,0]
    
def det3x3(A):
    """
    Calculate determinant of a 3x3 matrix
    """
    return A[0, 0] * A[1, 1] * A[2, 2] + A[0, 2] * A[1, 0] * A[2, 1] + \
           A[0, 1] * A[1, 2] * A[2, 0] - A[0, 2] * A[1, 1] * A[2, 0] - \
           A[0, 1] * A[1, 0] * A[2, 2] - A[0, 0] * A[1, 2] * A[2, 1]

def transpose_3x3(A):
    """
    transpose a 3x3 matrix
    """
    B = A[:]
    
           
def inv3x3(A):
    """
    Inverse of a 3x3 matrix
    """
    a00 = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    a01 = A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]
    a02 = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    a10 = A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]
    a11 = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    a12 = A[0, 2] * A[1, 0] - A[0, 0] * A[1, 2]
    a20 = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
    a21 = A[0, 1] * A[2, 0] - A[0, 0] * A[2, 1]
    a22 = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    det = det3x3(A)
    return numpy.array([[a00, a01, a02],
                        [a10, a11, a12],
                        [a20, a21, a22]]) / det
                        
def matrix_vec_mul3(A, x):
    """
    Matrix multiplication with vector x
    Matrix is 3x3, x is 1x3
    """
    y0 = A[0, 0] * x[0] + A[0, 1] * x[1] + A[0, 2] * x[2]
    y1 = A[1, 0] * x[0] + A[1, 1] * x[1] + A[1, 2] * x[2]
    y2 = A[2, 0] * x[0] + A[2, 1] * x[1] + A[2, 2] * x[2]
    return numpy.array([y0, y1, y2])
    
def cross(x, y):
    """
    cross product of 2 length 3 vectors
    """
    z0 = x[1] * y[2] - x[2] * y[1]
    z1 = x[2] * y[0] - x[0] * y[2]
    z2 = x[0] * y[1] - x[1] * y[0]
    return numpy.array([z0, z1, z2])

def list_or_scalar_to_str(render_t):
    if hasattr(render_t, '__len__'):
        return ','.join(str(x) for x in render_t)
    else:
        return str(render_t)

def get_shader_dirname(base_dir, objective_functor, normal_map, geometry, render_prefix=False):
    if normal_map == '':
        normal_map = 'none'
    if isinstance(objective_functor, str):
        if render_prefix:
            objective_functor = objective_functor[len('render_'):]
        objective_name = objective_functor
    else:
        objective_name = objective_functor.__name__
    outdir = os.path.join(base_dir, objective_name + '_' + geometry + '_normal_' + normal_map)
    return outdir

def render_shader(objective_functor, window_size, trace=False, outdir=None, ground_truth_samples=None, use_triangle_wave=False, is_color=default_is_color, normal_map='', nproc=None, geometry='', nframes=None, start_t=0.0, end_t=1.0, base_dir='out', base_ind=0, check_kw={}, get_objective=False, use_objective=None, verbose=True, camera_path=1, nfeatures=22, denoise=False, denoise_h=None, grounddir=None, msaa_samples=None, from_tuned = False):
    """
    Low-level routine for rendering a shader.
    
    If ground_truth_samples is positive, use that many samples MSAA for estimating ground truth: if 0 or negative, skip evaluation of ground truth.
    If nproc is given, use that many processes in parallel.
    If normal_map is given, it is a string description of a kind of normal mapping.
    If get_objective is True, then just return the objective function.
    If use_objective is not None, then use the given objective in place of generating one.
    """
    if ground_truth_samples is None:
        ground_truth_samples = 1000

    (height, width) = window_size
    if outdir is None:
        outdir = get_shader_dirname(base_dir, objective_functor, normal_map, geometry)
    if grounddir is None:
        grounddir = os.path.join(outdir, 'tune')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if verbose:
        print('Rendering shader to %s' % outdir)

    T0 = time.time()
    if use_objective:
        f = use_objective
    
    if from_tuned or not use_objective:
        if geometry in C_geometry:
            X = ArgumentArray(ndims=nfeatures, bounds=(1e-10, 1e10))
        else:
            X = ArgumentArray(ndims=3, bounds=[(0.0, width), (0.0, height), (start_t, end_t)])

        f2 = objective_functor(X, use_triangle_wave)
        if get_objective:
            return f2
        if not use_objective:
            f = f2
            
    if use_objective and from_tuned:
        transfer_ast(f, f2)
        f = f2

    c = CompilerParams(compute_g=True, trace=trace)
    if is_color:
        c.constructor_code = [OUTPUT_ARRAY + '.resize(3);']
    c0 = copy.deepcopy(c)

    if nframes is None:
        nframes = 20
    if nframes > 1:
        render_t = numpy.linspace(start_t, end_t, nframes)
    else:
        render_t = numpy.array([start_t])

    render_sigma_x = render_sigma_y = 0.5
    render_sigma_t = 0.0

    extra_args = '--render %d,%d --render_sigma %f,%f,%f --outdir %s --min_time %f --max_time %f --grounddir %s'%(width,height, render_sigma_x, render_sigma_y, render_sigma_t, os.path.abspath(outdir), start_t, end_t, os.path.abspath(grounddir))
    
    if geometry in C_geometry:
        extra_args += ' --shader_only 1 --geometry_name %s --nfeatures %d --camera_path %d'%(geometry, nfeatures, camera_path)
        check_kw['ndims'] = 3

    check_kw = dict(check_kw)
    check_kw.setdefault('print_command', True)
    if 'extra_args' in check_kw:
        extra_args += check_kw['extra_args']
        del check_kw['extra_args']
    check(f, c, do_run=False, extra_args=extra_args, **check_kw)
    
    cmdL = []
    for (i, t) in enumerate(render_t):
        cmdL.append(check(f, c, do_compile=False, get_command=True, extra_args=extra_args + ' --render_t ' + str(t) + ' --render_index ' + str(i+base_ind) + ('' if msaa_samples is None else (' --samples %d' % msaa_samples)), **check_kw))

    if ground_truth_samples > 0 and not check_kw.get('time_error', False):   # Compute ground truth image only if tuner not being used (via time_error check() keyword argument check_kw)
        c = c0
        c.log_intermediates = False
        for (i, t) in enumerate(render_t):
            cmdL.append(check(f, c, do_compile=False, get_command=True, extra_args=extra_args + ' --samples ' + str(ground_truth_samples) + ' --gname ground --do_f 0' + ' --render_t ' + str(t) + ' --render_index ' + str(i+base_ind) + ' --is_gt 1', **check_kw))

    is_print = check_kw.get('print_command', True)
    print('Render command list:')
    print('\n'.join(cmdL))

    T_start_run = time.time()
    out = system_parallel(cmdL, nproc, verbose=is_print)
    T_end_run = time.time()
    if print_benchmark:
        print('Run program (%s): %f secs' % (cmdL[0], T_end_run - T_start_run))

    if verbose:
        print(outdir + ': OK' + ' (%f secs)'%(time.time()-T0))

    times_f = parse_output_float(out, 'time_f', True)
    times_g = parse_output_float(out, 'time_g', True)
    errors_f = parse_output_float(out, 'error_f', True)
    errors_g = parse_output_float(out, 'error_g', True)
    times_f_geom = parse_output_float(out, 'time_f_geom', True)
    times_g_geom = parse_output_float(out, 'time_g_geom', True)

    ans_d = {}
    if len(times_f) or len(times_g) and len(errors_f) and len(errors_g):
        ans_d.update({'time_f': numpy.mean(times_f),
                 'time_g': numpy.mean(times_g),
                 'error_f': numpy.mean(errors_f),
                 'error_g': numpy.mean(errors_g),
                 'time_f_geom': numpy.mean(times_f_geom),
                 'time_g_geom': numpy.mean(times_g_geom)})

    if denoise:
        times_g_nlm = parse_output_float(out, 'time_g_nlm', True)
        ans_d.update({'denoise_h': denoise_h,
                      'time_g_nlm': numpy.mean(times_g_nlm)})
    return ans_d

def render_any_shader(objective_functor, window_size, *args, **kw):
    """
    Render any shader allowing multiple camera paths.
    For multiple camera paths, compile each one, and append the index
    """
    ans = []
    geometry = kw['geometry']
    if geometry in C_geometry:
        render_func = globals()['geometry_wrapper']
    else:
        render_func = globals()['render_' + geometry + '_shader']
    num_cameras = kw.get('num_cameras', 1)
    nframes = kw['nframes']
    for i in range(num_cameras):
        kw['base_ind'] = i * nframes
        kw['camera_path'] = kw.get('specific_camera_path', i+1)
        ans.append(render_func(objective_functor, window_size, *args, **kw))
    return ans

def normal_mapping(time, tex_coords, viewer_dir, unit_normal, tangent_t, tangent_b, *args, **kw):
    """
    Modifies normal according to normal mapping
    """
    displace = 'parallax_normal'
    normal_map = kw.get('normal_map', 'none')
    if normal_map == 'none':
        unit_new_normal = unit_normal
    else:
        cross_tangent = vec('cross_tangent', cross(tangent_t, tangent_b))
        normal = cross_tangent
        u = tex_coords[0]
        v = tex_coords[1]
        
        if normal_map == 'spheres':
                f = 0.5
                fu = fract(f*u)*2-1
                fv = fract(f*v)*2-1
                
                h2 = Var('h2', 1-fu**2-fv**2)
                h = Var('h', max_nosmooth(h2, 1e-5)**0.5)
                valid = Var('valid', h2 > 0.0)
                dhdu = Var('dhdu', select(valid, -2*f*fu * h**(-0.5), 0.0))
                dhdv = Var('dhdv', select(valid, -2*f*fv * h**(-0.5), 0.0))
        elif normal_map == 'bumps':
            f = 1.0
            fu = f*u
            fv = f*v
            h = sin(fu)*sin(fv)
            dhdu = f*cos(fu)*sin(fv)
            dhdv = f*cos(fv)*sin(fu)
        elif normal_map == 'ripples':
            f = 3.0
            velocity = 15.0
            a = 1.0/3
            
            t = time
            r2 = (u**2 + v**2)
            r = r2**0.5
            theta = Var('theta2', r*f-t*velocity)
            h = a*sin(theta)
            dhdu = Var('dhdu', a*f*u*r2**(-0.5)*cos(theta))
            dhdv = Var('dhdv', a*f*v*r2**(-0.5)*cos(theta))
        else:
            raise ValueError('unknown normal_map shader')
            
        small_t = vec('small_t', cross(unit_normal, tangent_b))
        small_b = vec('small_b', cross(tangent_t, unit_normal))
            
        new_normal = normal - dhdu * small_t - dhdv * small_b
        Nl = Var('Nl', (new_normal[0]**2 + new_normal[1]**2 + new_normal[2]**2)**0.5)
        unit_new_normal = vec('unit_new_normal', new_normal / Nl)

        surface_matrix_inv = numpy.transpose(numpy.array([tangent_t, tangent_b, normal]))
        surface_matrix = inv3x3(surface_matrix_inv)
        v_ref = matrix_vec_mul3(surface_matrix, viewer_dir)
        
        # Parallax mapping from Szirmay-Kalos and Umenhoffer 2006, Displacement Mapping on the GPU — State of the Art
        if displace == 'parallax':
            tex_coords[0] = u + h * v_ref[0] / v_ref[2]
            tex_coords[1] = v + h * v_ref[1] / v_ref[2]
        elif displace == 'parallax_normal':
            new_normal_ref = vec('new_normal_ref', matrix_vec_mul3(surface_matrix, new_normal))
            scale = Var('scale', h * new_normal_ref[2])
            tex_coords[0] = u + scale * v_ref[0]
            tex_coords[1] = v + scale * v_ref[1]
        elif displace == 'none':
            pass
        else:
            raise ValueError
            
    return (unit_new_normal, tex_coords)
   
def clear_extra_keywords(kw):
    kw2 = kw.copy()
    #kw2.pop('camera_path', 0)
    kw2.pop('num_cameras', 0)
    kw2.pop('need_time', 0)
    kw2.pop('specific_camera_path', 0)
    kw2.pop('approx_mode', 0)
    return kw2

def geometry_wrapper(objective_functor, window_size, *args, **kw):
    def objective(X, *args):
        intersect_pos = numpy.array([X[1], X[2], X[3]])
        tex_coords = numpy.array([X[7], X[8]])
        normal = numpy.array([X[9], X[10], X[11]])
        light_dir = numpy.array([X[19], X[20], X[21]])
        viewer_dir = numpy.array([X[4], X[5], X[6]])
        time = X[0]
        is_intersect = X[18]
        tangent_t = numpy.array([X[12], X[13], X[14]])
        tangent_b = numpy.array([X[15], X[16], X[17]])
        
        if kw['geometry'] == 'plane':
            tex_coords = vec('tex_coords', numpy.array([intersect_pos[0], intersect_pos[1]]))
            normal = vec('normal', numpy.array([0.0, 0.0, 1.0]))
            light_dir = vec('light_dir', normalize_const(numpy.array([0.3, 0.8, 1.0])))
            is_intersect = 1.0
            tangent_t = vec('tangent_t', numpy.array([1.0, 0.0, 0.0]))
            tangent_b = vec('tangent_b', numpy.array([0.0, 1.0, 0.0]))
        elif kw['geometry'] in ['sphere', 'hyperboloid1']:
            tex_coords = vec('tex_coords', numpy.array([X[7], X[8]]))
            normal = vec('normal', numpy.array([X[9], X[10], X[11]]))
            light_dir = vec('light_dir', normalize_const(numpy.array([-0.3, -0.5, 1.0])))
            is_intersect = X[18]
            tangent_t = vec('tangent_t', numpy.array([X[12], X[13], X[14]]))
            tangent_b = vec('tangent_b', numpy.array([X[15], X[16], X[17]]))

        (normal, tex_coords) = normal_mapping(time, tex_coords, viewer_dir, normal, tangent_t, tangent_b, *args, **kw)

        approx_mode = kw.get('approx_mode', 'gaussian')
        if approx_mode == 'gaussian':
            target_approx = APPROX_GAUSSIAN
        elif approx_mode == 'regular':
            target_approx = APPROX_REGULAR_INTEGRATION
        elif approx_mode == 'mc':
            target_approx = APPROX_MC
        else:
            raise ValueError('unknown approx mode: %s' % approx_mode)

        if kw['geometry'] == 'plane' or (geometry_transfer and kw.get('from_tuned', False)):
            return objective_functor(intersect_pos, tex_coords, normal, light_dir, viewer_dir, *args, time=time)
        
        ans = objective_functor(intersect_pos, tex_coords, normal, light_dir, viewer_dir, *args, time=time)
        return wrap_is_intersect(ans, is_intersect, kw['is_color'], target_approx)
    objective.__name__ = objective_functor.__name__
    if kw['geometry'] == 'plane':
        kw['nfeatures'] = 7
    elif kw['geometry'] in ['sphere', 'hyperboloid1']:
        kw['nfeatures'] = 19
    kw2 = clear_extra_keywords(kw)
    return render_shader(objective, window_size, *args, **kw2)
    
def wrap_is_intersect(f, is_intersect, is_color, target_approx):
    compare = is_intersect >= 0.0
    compare.approx = target_approx
    if is_color:
        if isinstance(f, Compound) and len(f.children) == 3 and all(isinstance(child, Assign) for child in f.children):
            ans = [child.children[1] for child in f.children]
        out_intensity_R = select_nosmooth(compare, ans[0], 0.0)
        out_intensity_G = select_nosmooth(compare, ans[1], 0.0)
        out_intensity_B = select_nosmooth(compare, ans[2], 0.0)
        out_intensity_R.approx = APPROX_NONE
        out_intensity_G.approx = APPROX_NONE
        out_intensity_B.approx = APPROX_NONE
        out_intensity = numpy.array([out_intensity_R, out_intensity_G, out_intensity_B])
        ans = output_color(out_intensity)
    else:
        ans = select_nosmooth(compare, f, 0.0)
        ans.approx = APPROX_NONE
    return ans
    
def multiple_shaders(base_func):
    def f(objective_functor_L, *args, **kw):
        ans = []
        for objective_functor in objective_functor_L:
            ans.append(base_func(objective_functor, *args, **kw))
        return ans
    return f

def output_color(c):
    """
    Given a array-like object with 3 Exprs, output an RGB color for a shader. Has the side effect of setting channels of c.
    """
    set_channels(c)
    return Compound([Assign(GetItem(ArgumentArray(OUTPUT_ARRAY), i), c[i]) for i in range(3)])

def intersect_keys(d, target_keys):
    return dict([(key, d.get(key, 0.0)) for key in target_keys])
    
def transfer_ast(f_old, f_new):
    """
    f_old is old ast stored in other tuned geometry (more complicated)
    f_new is current geometry (less complicated)
    """
    f_new.set_approx_info(f_old.get_approx_info())
    if type(f_old) == type(f_new):
        children_old = f_old.children
        children_new = f_new.children
        assert len(children_old) == len(children_new)
    else:
        if isinstance(f_new, ConstExpr):
            return True
        else:
            if isinstance(f_new, Var) and f_new.name in FIX_NAMES:
                return True
            if isinstance(f_old, Call) and f_old.name == 'our_select':
                if isinstance(f_old.children[1].children[1].children[1].children[1].children[1], GetItem) and f_old.children[1].children[1].children[1].children[1].children[1].children[1] == 18:
                    transfer_ast(f_old.children[2], f_new)
                    return False
            raise
            
    if isinstance(f_new, (GetItem, ConstExpr)):
        return False
    
    for i in range(len(children_new)):
        if not isinstance(children_new[i], Expr):
            assert children_old[i] == children_new[i]
        else:
            transfer_ast(children_old[i], children_new[i])
    return False
  
def fix_ast(f_old, f_new):
    """
    f_old is old ast stored in dill
    f_new is new ast contains variable name for geometry inputs
    replace geometry input nodes in f_old using nodes in f_new
    so that geometry transfer is possible
    """
    if type(f_old) == type(f_new):
        children_old = f_old.children
        children_new = f_new.children
        assert len(children_old) == len(children_new)
    else:
        if isinstance(f_old, ConstExpr):
            return True
        else:
            if isinstance(f_new, Var) and f_new.name in FIX_NAMES:
                return True
            raise    
            
    if isinstance(f_old, BinaryOp):
        assert children_old[0] == children_new[0]
        if fix_ast(children_old[1], children_new[1]):
            set_adjust_var = isinstance(f_old.a, ConstExpr)
            f_old.children[1] = f_new.children[1]
            f_old.a = f_new.a
            if not set_adjust_var:
                f_old.a.no_approx_adjust_var = False
        if fix_ast(children_old[2], children_new[2]):
            set_adjust_var = isinstance(f_old.b, ConstExpr)
            f_old.children[2] = f_new.children[2]
            f_old.b = f_new.b
            if not set_adjust_var:
                f_old.b.no_approx_adjust_var = False
    elif isinstance(f_old, (Call, Var)):
        for i in range(len(children_old)):
            if not isinstance(children_old[i], Expr):
                assert children_old[i] == children_new[i]
            else:
                if fix_ast(children_old[i], children_new[i]):
                    set_adjust_var = isinstance(f_old.children[i], ConstExpr)
                    f_old.children[i] = f_new.children[i]
                    if not set_adjust_var:
                        f_old.children[i].no_approx_adjust_var = False
                    if isinstance(f_old, Var) and set_adjust_var:
                        f_old.no_approx_adjust_var = True
        if isinstance(f_old, Call):
            no_adjust_vars = [getattr(child, 'no_approx_adjust_var', False) for child in f_old.children if isinstance(child, Expr)]
            if all(no_adjust_vars):
                f_old.no_approx_adjust_var = True
    else:
        if f_old.children != f_new.children:
            raise
        
    return False

render_shaders = multiple_shaders(render_shader)
render_any_shaders = multiple_shaders(render_any_shader)
C_geometry = [
'hyperboloid1',
'hyperboloid2',
'plane',
'sphere']
FIX_NAMES = [
'tex_coords_x',
'tex_coords_y',
'tex_coords_z',
'normal_x',
'normal_y',
'normal_z',
'light_dir_x',
'light_dir_y',
'light_dir_z',
'tangent_t_x',
'tangent_t_y',
'tangent_t_z',
'tangent_b_x',
'tangent_b_y',
'tangent_b_z']
