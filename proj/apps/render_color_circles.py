
"""
A color (3 channel) shader that creates colorful circles of varying radii.
"""

from render_util import *
from render_single import render_single
approx_mode = 'gaussian'

def color_circles(position, tex_coords, normal, light_dir, viewer_dir, use_triangle_wave, time=None):
    """
    Shader arguments:
     - position:   3 vector of x, y, z position on surface
     - tex_coords: 2 vector for s, t texture coords
     - normal:     3 vector of unit normal vector on surface
     - light_dir:  3 vector of unit direction towards light source
     - viewer_dir: 3 vector of unit direction towards camera
     - use_triangle_wave: if True then triangle_wave() should be used instead of fract() for shaders that perform tiling.
    """
    
    specular_pow = 25.0
    circle_r = 25.0 / 3.0
    gap = 5.0 / 3.0
    circle_d = 2*circle_r+gap*2
    def floor2(x):              # Work around floor() not being finished (TODO: fix floor())
        return x - fract(x)
    
    x = tex_coords[0]
    y = tex_coords[1]
    if use_triangle_wave:
        xm = Var('xm', (triangle_wave(x/circle_d)*circle_d-gap))
        ym = Var('ym', (triangle_wave(y/circle_d)*circle_d-gap))
    else:
        xf = fract(x/circle_d)
        yf = fract(y/circle_d)
        xm = Var('xm', (xf*circle_d-gap))
        ym = Var('ym', (yf*circle_d-gap))
    r2 = Var('r2', (xm-circle_r)**2 + (ym-circle_r)**2)
    r = Var('r', r2**0.5)
    xf = floor2(x/circle_d)
    yf = floor2(y/circle_d)
    circle_r_varying = (0.8+sin(xf+yf)*0.1+0.1*cos(xf-yf))*circle_r
    circle_indicator = 0.5 - 0.5 * sign(r - circle_r_varying)
    
    circle_color = vec('circle_color', [0.6+0.4*sin(x/10), 0, 0.5+0.5*cos(y/10)])
    LN = Var('LN', max(dot(light_dir, normal), 0.0))    # Diffuse term
    R = vec('R', 2.0 * LN * normal - light_dir)
    specular_intensity = Var('specular_intensity', (LN > 0) * max(dot(R, viewer_dir), 0.0) ** specular_pow)
    diffuse_before = circle_color * LN                                      # Include only diffuse term: no specular
    diffuse = diffuse_before * circle_indicator
    ans = output_color(diffuse + specular_intensity)

    if approx_mode == 'gaussian':
        ans.set_approx_recurse(APPROX_GAUSSIAN)
    elif approx_mode == 'mc':
        ans.set_approx_recurse(APPROX_MC)
    else:
        raise ValueError
    return ans

shaders = [color_circles]                   # This list should be declared so a calling script can programmatically locate the shaders
is_color = True                             # Set to True if the shader is 3 channel (color) and False if shader is greyscale

def main():
    render_single('out', 'render_color_circles', 'plane', 'none', sys.argv[1:], nframes=1, time_error=False)

if __name__ == '__main__':
    main()
