
"""
A greyscale (1 channel) shader that creates tiled circles.
"""

from render_util import *
from render_single import *

approx_mode = 'gaussian'
#approx_mode = 'dorn'

def circles(position, tex_coords, normal, light_dir, viewer_dir, use_triangle_wave, time):
    """
    Shader arguments:
     - position:   3 vector of x, y, z position on surface
     - tex_coords: 2 vector for s, t texture coords
     - normal:     3 vector of unit normal vector on surface
     - light_dir:  3 vector of unit direction towards light source
     - viewer_dir: 3 vector of unit direction towards camera
     - use_triangle_wave: if True then triangle_wave() should be used instead of fract() for shaders that perform tiling.
    """
    circle_r = 25.0 / 3.0
    gap = 5.0 / 3.0
    circle_d = 2*circle_r+gap*2
    
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
    sign_op = sign(r - circle_r)
    ans = 0.5 - 0.5 * sign_op
    
    LN = Var('LN', max(dot(light_dir, normal), 0.0))    # Diffuse term
    ans = LN * ans                                      # Include only diffuse term: no specular

    if approx_mode == 'dorn':
        sign_op.approx = APPROX_DORN
    else:
        ans.set_approx_recurse(APPROX_GAUSSIAN)
        
    return ans

shaders = [circles]                 # This list should be declared so a calling script can programmatically locate the shaders
is_color = False                    # Set to True if the shader is 3 channel (color) and False if shader is greyscale

def main():
    for time_error in [True, False]:
        render_single('out', 'render_circles', 'plane', 'none', sys.argv[1:], nframes=1, time_error=time_error)

if __name__ == '__main__':
    main()
