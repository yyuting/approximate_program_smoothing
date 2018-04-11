
"""
A color (3 channel) shader which renders staggered brick

Re-implementation of shader described in Dorn et al. paper
"""

from render_util import *
from render_single import render_single

approx_mode = 'mc'

def bricks(position, tex_coords, normal, light_dir, viewer_dir, use_triangle_wave, time):
    """
    Shader arguments:
     - position:   3 vector of x, y, z position on surface
     - tex_coords: 2 vector for s, t texture coords
     - normal:     3 vector of unit normal vector on surface
     - light_dir:  3 vector of unit direction towards light source
     - viewer_dir: 3 vector of unit direction towards camera
     - use_triangle_wave: if True then triangle_wave() should be used instead of fract() for shaders that perform tiling.
    """

    #Sources
    #https://www.shadertoy.com/view/Mll3z4
    #Dorn et al. 2015

    #####
    # Parameter Settings
    #####

    #set discrete brick color, vs. just applying mortar grid over noise distribution
    #discrete is more similar to original Dorn implementation.
    #NOTE: currently having trouble setting Var to be a boolean, using 1.0 -> True
    discrete_bricks = Var('discrete_bricks', 0.0)

    #lighting
    specular_pow = 25.0

    #brick dimensions
    brick_height = 5.0
    brick_width = 20.0
    mortar_width = 0.5

    #colors
    brick_color_light = vec_color('brick_color_light', [0.49, 0.02, 0.11]) #http://rgb.to/keyword/4261/1/brick
    brick_color_dark = vec_color('brick_color_dark', [0.7, 0.02, 0.05]) #Dorn et al. 2015
    mortar_color =  vec_color('mortar_color', [0.7, 0.7, 0.7]) #Dorn et al. 2015

    #gabor noise parameters
    K = 1.0
    a = 0.05
    F_0 = 0.05
    omega_0 = 0.0
    impulses = 64.0
    period = 256

    x = tex_coords[0]
    y = tex_coords[1]

    #used to identify mortar location
    x_mod = x % brick_width
    x_abs = x_mod * sign(x_mod)
    y_mod = y % brick_height
    y_abs = y_mod * sign(y_mod)


    #alternate staggered brick rows
    double_brick_height = brick_height * 2.0
    row_indicator = 0.5 - 0.5 * sign(brick_height - y % double_brick_height)
    staggered_x_abs = (x_abs + (brick_width / 2.0)) % brick_width

    #optional use of discretized bricks to set noise parameter.
    x_discrete = select(1.0 == row_indicator, x - x_abs, x - staggered_x_abs)
    y_discrete = y - y_abs

    x_noise = x
    y_noise = y

    noise = Var('noise', 0.5 + 0.5 * gabor_noise(K, a, F_0, omega_0, impulses, period, x_noise, y_noise))
    
    #compute pixel color
    brick_color = brick_color_light * noise + brick_color_dark * (1.0 - noise)
    #horizontal mortar lines
    brick_color = select(mortar_width >= y_abs, mortar_color, brick_color)
    #vertical mortar lines (staggered)
    vertical_brick_boundary = select(1.0 == row_indicator, x_abs, staggered_x_abs)
    brick_color = select(mortar_width >= vertical_brick_boundary, mortar_color, brick_color)

    #lighting computations
    LN = Var('LN', max(dot(light_dir, normal), 0.0))    # Diffuse term
    R = vec('R', 2.0 * LN * normal - light_dir)
    specular_intensity = Var('specular_intensity', (LN > 0) * max(dot(R, viewer_dir), 0.0) ** specular_pow)
    diffuse = brick_color * LN

    #get output color
    ans = output_color(diffuse + specular_intensity)

    if approx_mode == 'gaussian':
        ans.set_approx_recurse(APPROX_GAUSSIAN)
    elif approx_mode == 'mc':
        ans.set_approx_recurse(APPROX_MC)
        ans.set_approx_mc_samples_recurse(16)
    else:
        raise ValueError
    return ans

shaders = [bricks]                 # This list should be declared so a calling script can programmatically locate the shaders
log_intermediates = False
is_color = True
normal_map = True

def main():
    for time_error in [True, False]:
        render_single('out', 'render_bricks', 'plane', 'none', sys.argv[1:], nframes=1, time_error=time_error)

if __name__ == '__main__':
    main()
