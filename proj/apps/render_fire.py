"""
A color (3-channel) shader that renders an animated fire-like texture

Currently there are four "fire" animations, each of which is contained in a
separate interpolate_color function.
"""

from render_util import *
from render_single import *

approx_mode = 'gaussian'

def lerp(color_1, color_2, s):
    """
    Helper function to interpolate between two colors, where s is a floating
    point number between 0.0 and 1.0

    Reimplementation from bandlimiting codebase (api3d.h)
    """
    return color_1 * s + color_2 * (1.0 - s)

def get_color_palette(include_boundary_colors=True):
    """
    Returns set of available colors as a list.

    include_boundary_colors flag specifies inclusion of white/black colors.
    """
    #https://www.w3schools.com/colors/colors_groups.asp
    black = vec('black', [0.0, 0.0, 0.0])
    red = vec('red', [1.0, 0.0, 0.0])
    orange_red = vec('orange_red', [1.0, 0.27, 0.0])
    orange = vec('orange', [1.0, 0.65, 0.0])
    yellow = vec('yellow', [1.0, 1.0, 0.0])
    white = vec('white', [1.0, 1.0, 1.0])
    if include_boundary_colors:
        return [black, red, orange_red, orange, yellow, white]
    else:
        return [red, orange_red, orange, yellow]

def get_color_interpolation(palette, s, include_boundary_colors=True):
    """
    Given a list of colors (palette) and a parameterization s, compute an
    interpolated color and return it.

    include_boundary_colors flag specifies inclusion of white/black colors.
    """

    num_colors = len(palette)

    s_mod = s % num_colors
    s_frac = 1.0 - fract(s)

    color = lerp(palette[0], palette[1], s_frac)
    color = select(s_mod >= 1, lerp(palette[1], palette[2], s_frac), color)
    color = select(s_mod >= 2, lerp(palette[2], palette[3], s_frac), color)
    if include_boundary_colors:
        color = select(s_mod >= 3, lerp(palette[3], palette[4], s_frac), color)
        color = select(s_mod >= 4, lerp(palette[4], palette[5], s_frac), color)
        color = select(s_mod >= 5, lerp(palette[5], palette[0], s_frac), color)
    else:
        color = select(s_mod >= 3, lerp(palette[3], palette[0], s_frac), color)

    return color

def interpolate_color_4(x, y, t):
    """
    Modification of zigzag shader where flames appear to move upward along
    the surface
    """

    #sources:
    #https://krazydad.com/tutorials/makecolors.php
    palette = get_color_palette()
    num_colors = len(palette)

    sin_arg = x + 0.8 * sin(y)
    modulation_1 = 0.5+0.5*sign(0.5*(sin(sin_arg)))

    s_1 = 2.0 * sin(0.005 * x**2 + t) + y + 5.0*t
    s_2 = 2.0 * sin(0.005 * x**2 + t) + y + 5.0*t + len(palette) / 4.0

    return select(modulation_1 >= 1.0, get_color_interpolation(palette, s_1), get_color_interpolation(palette, s_2))

def interpolate_color_3(x, y, t):
    """
    Sends out dynamic "waves" along the x-axis
    """

    #sources:
    #https://www.openprocessing.org/sketch/6731

    palette = get_color_palette()
    num_colors = len(palette)

    #curve function
    A = 10.0
    w = 0.005
    a = 3.0
    curve_time = A * sin(w*(x**2) - a*t) - y
    curve_time = curve_time - x

    return get_color_interpolation(palette, curve_time)

def interpolate_color_2(x, y, t):
    """
    Generates undulating sin waves
    """

    #sources:
    #https://krazydad.com/tutorials/makecolors.php
    #http://colalg.math.csusb.edu/~devel/precalcdemo/param/src/param.html
    #https://www.varsitytutors.com/hotmath/hotmath_help/topics/graphing-sine-function

    palette = get_color_palette()
    num_colors = len(palette)

    #curve function
    A = 10.0
    w = 0.05
    curve = A * sin(w*x) - y
    curve_mod = curve % num_colors

    curve_time = A * sin(w*x - t) - y
    curve_time = curve_time - x

    return get_color_interpolation(palette, curve_time)

def interpolate_color(x, y, t):
    """
    Upward-moving waves with varying velocity
    """

    palette = get_color_palette()

    w = 0.0005
    r = (x ** 2 + y ** 2) ** 0.5
    palette_selector = 20.0 * sin(w * y ** 2) + (t - x/50.0) ** 2 + y + t ** 2 + 10.0 * sin(0.005 * x**2)

    base_color =  get_color_interpolation(palette, palette_selector)

    return base_color



def fire(position, tex_coords, normal, light_dir, viewer_dir, use_triangle_wave, time):
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


    x = tex_coords[0]
    y = tex_coords[1]

    #center spiral on origin
    centered_x = x - x_origin
    centered_y = y - y_origin

    #get color at current (x, y, t) location
    color = interpolate_color_4(centered_x, centered_y, time)

    #lighting computations
    #Diffuse
    LN = Var('LN', max(dot(light_dir, normal), 0.0))
    diffuse = color * LN
    #Specular
    R = vec('R', 2.0 * LN * normal - light_dir)
    specular_intensity = Var('specular_intensity', (LN > 0) * max(dot(R, viewer_dir), 0.0) ** specular_pow)

    ans = output_color(diffuse + specular_intensity)
    ans.set_approx_recurse(APPROX_GAUSSIAN)

    return ans

shaders = [fire]                 # This list should be declared so a calling script can programmatically locate the shaders
is_color = True
normal_map = True

def main():
	render_single('out', 'render_fire', 'plane', 'none', sys.argv[1:], nframes=1, time_error=False)

if __name__ == '__main__':
    main()
