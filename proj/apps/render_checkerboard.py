from render_util import *
from render_single import *

def checkerboard(position, tex_coords, normal, light_dir, viewer_dir, use_triangle_wave, time):
    """
    Shader arguments:
     - position:   3 vector of x, y, z position on surface
     - tex_coords: 2 vector for s, t texture coords
     - normal:     3 vector of unit normal vector on surface
     - light_dir:  3 vector of unit direction towards light source
     - viewer_dir: 3 vector of unit direction towards camera
     - use_triangle_wave: if True then triangle_wave() should be used instead of fract() for shaders that perform tiling.
    """
    specular_pow = 50.0
    checkerboard_w = 20.0
    
    x = tex_coords[0]
    y = tex_coords[1]

    xdiv = x/checkerboard_w
    ydiv = y/checkerboard_w
    xs = fract(xdiv)
    ys = fract(ydiv)
    xs_compare = xs >= 0.5
    ys_compare = ys >= 0.5
    ss = select(xs_compare, 1.0, 0.0)
    tt = select(ys_compare, 1.0, 0.0)

    mul1 = ss * tt
    mul2 = (1.0 - ss) * (1.0 - tt)
    
    ans0 = (mul1 + mul2)
    
    LN = Var('LN', max(dot(light_dir, normal), 0.0))    # Diffuse term
    R = vec('R', 2.0 * LN * normal - light_dir)
    specular_intensity = Var('specular_intensity', max(dot(R, viewer_dir), 0.0) ** specular_pow)
    ans = LN * ans0 + specular_intensity                              # Include only diffuse term: no specular

    ans.set_approx_recurse(APPROX_GAUSSIAN)
    specular_intensity.approx = APPROX_NONE
    LN.approx = APPROX_NONE
    for nn in normal:
        nn.set_approx_recurse(APPROX_MC)

    return ans

shaders = [checkerboard]            # This list should be declared so a calling script can programmatically locate the shaders
is_color = False                    # Set to True if the shader is 3 channel (color) and False if shader is greyscale

def main():
    for time_error in [True, False]:
        render_single('out', 'render_checkerboard', 'plane', 'none', sys.argv[1:], nframes=1, time_error=time_error)

if __name__ == '__main__':
    main()
    
