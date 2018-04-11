
"""
A color (3 channel) shader that creates a zigzag pattern.
"""

from render_util import *
from render_single import *

def zigzag(position, tex_coords, normal, light_dir, viewer_dir, use_triangle_wave, time=None):
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
    
    LN = Var('LN', dot(light_dir, normal))
    R = vec('R', 2.0 * LN * normal - light_dir)
    diffuse_intensity = Var('diffuse_intensity', max(dot(light_dir, normal), 0.0))
    specular_intensity = Var('specular_intensity', max_nosmooth(dot(R, viewer_dir), 0.0) ** specular_pow)
    
    freq = numpy.array([1.0, 1.0, 1.0])

    xarg = Var('xarg', freq[0]*tex_coords[0])
    yarg = Var('yarg', freq[1]*tex_coords[1])

    sin_arg = freq[2]*(xarg) + 0.8*sin(yarg)
    modulation1 = Var('modulation1', 0.5+0.5*sign(0.5*(sin(sin_arg)))) 
    diffuse = 0.8
    specular = 1.0
    diffuse_c1 = numpy.array([1.0, 1.0, 1.0])
    diffuse_c2 = numpy.array([0.3, 0.3, 1.0])
    ambient = numpy.array([0.1, 0.1, 0.1])
    
    diffuse_sum = vec('diffuse_sum', modulation1*(diffuse_c1 + (diffuse_c2-diffuse_c1)*(0.5+0.5*(cos(sin_arg*0.5)))))# + modulation2*diffuse_c2)
    
    base_diffuse = Var('base_diffuse', diffuse * diffuse_intensity)
    base_specular = Var('base_specular', specular * specular_intensity)
    
    output_intensity_r = Var('output_intensity_r', base_diffuse*diffuse_sum[0] + base_specular + ambient[0])
    output_intensity_g = Var('output_intensity_g', base_diffuse*diffuse_sum[1] + base_specular + ambient[1])
    output_intensity_b = Var('output_intensity_b', base_diffuse*diffuse_sum[2] + base_specular + ambient[2])
    output_intensity = numpy.array([output_intensity_r, output_intensity_g, output_intensity_b])
    
    ans = output_color(output_intensity)

    ans.set_approx_recurse(APPROX_GAUSSIAN)
    
    return ans

shaders = [zigzag]                 # This list should be declared so a calling script can programmatically locate the shaders
is_color = True                    # Set to True if the shader is 3 channel (color) and False if shader is greyscale

def main():
    render_single('out', 'render_zigzag', 'plane', 'none', sys.argv[1:], nframes=1, time_error=False)

if __name__ == '__main__':
    main()
