
"""
Render all shaders.

Usage:

python render_all.py                      --- Render all shaders
python render_all.py --no-ground-truth    --- Render all shaders with no ground truth
"""

from render_util import *
import os

# Small list of shaders for quick tests
shader_modules = """
render_bricks
render_checkerboard
render_circles
render_color_circles
render_fire
render_sin_quadratic
render_zigzag
""".split()

def main():
    args = sys.argv[1:]
    if len(args) == 0 or args[0].startswith('--'):
        print('python render_all.py base_dir [--print]')
        sys.exit(1)

    base_dir = args[0]

    cmdL = []
    full_names = []
    for shader_name in shader_modules:
        for normal_map in ['ripples', 'spheres', 'none']:
            for geometry in ['plane', 'sphere', 'hyperboloid1']:
                cmdL.append('python render_single.py %s %s %s %s' % (base_dir, shader_name, geometry, normal_map))
                full_names.append(get_shader_dirname(base_dir, shader_name[len('render_'):], normal_map, geometry))
    print('Commands that will be run:')
    print('\n'.join(cmdL))
    print('')
    if '--print' in args:
        return

    times = ''
    for i in range(len(cmdL)):

        T0 = time.time()
        print(cmdL[i])
        os.system(cmdL[i])
        T1 = time.time()
        times += full_names[i] + ' ' + str(T1-T0) + ' secs\n'

        print('')
        print('-'*70)
        print('Rendering times:')
        print('-'*70)
        print(times)
        print('')

if __name__ == '__main__':
    main()
