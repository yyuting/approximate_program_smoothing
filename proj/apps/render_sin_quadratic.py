"""
A color (3-channel) shader that renders a function which
takes the sin of a quadratic function.

Note: This is a non-stationary function (interesting test for approximation)
"""

from render_util import *
from render_single import *

approx_mode = 'gaussian'

def sin_quadratic(position, tex_coords, normal, light_dir, viewer_dir, use_triangle_wave, time):
    """
    Shader arguments:
     - position:   3 vector of x, y, z position on surface
     - tex_coords: 2 vector for s, t texture coords
     - normal:     3 vector of unit normal vector on surface
     - light_dir:  3 vector of unit direction towards light source
     - viewer_dir: 3 vector of unit direction towards camera
     - use_triangle_wave: if True then triangle_wave() should be used instead of fract() for shaders that perform tiling.
    """

    ###
    # Parameters
    ###

    #lighting
    specular_pow = 25.0

    #translate origin into viewing area
    x_origin = 0.0
    y_origin = -55.0

    #sin function parameters
    A = 1.0
    w = 0.001

    #quadratic parameters
    const = 0.01
    a = 3.0 * cos(time) + const
    b = 3.0 * sin(time) + const
    c =  3.0 * sin(time) * cos(time) + const
    d = 0.0
    e = 0.0
    f = 0.0

    #colors (perform two-color blend)
    color_1 = vec('color_1', [1.0, 1.0, 1.0]) #silver
    color_2 = vec('color_2', [0.0, 0.0, 1.0]) #blue

    x = tex_coords[0]
    y = tex_coords[1]

    #center spiral on origin
    centered_x = x - x_origin
    centered_y = y - y_origin

    #compute A*sin(w*(a*x^2 + b*y^2 + c*xy + d*x + e*y + f)) to determine color blend weight
    color_weight = A * fract(sin(centered_x+centered_y)*0.2 + 3*(w * (a * centered_x ** 2 + b * centered_y ** 2 + c * centered_x * centered_y + d * centered_x + e * centered_y + f)))
    #blend colors according to color weight
    color = vec('color', color_weight * color_1 + (1.0 - color_weight) * color_2)

    #lighting computations
    #Diffuse
    LN = Var('LN', max(dot(light_dir, normal), 0.0))
    diffuse = color * LN
    #Specular
    R = vec('R', 2.0 * LN * normal - light_dir)
    specular_intensity = Var('specular_intensity', (LN > 0) * max(dot(R, viewer_dir), 0.0) ** specular_pow)

    #get output color

    ans = output_color(diffuse + specular_intensity)
    ans.set_approx_recurse(APPROX_GAUSSIAN)

    return ans

shaders = [sin_quadratic]                 # This list should be declared so a calling script can programmatically locate the shaders
is_color = True
normal_map = True
log_intermediates = False

def main():
	render_single('out', 'render_sin_quadratic', 'plane', 'none', sys.argv[1:], nframes=1, time_error=False)

if __name__ == '__main__':
    main()
