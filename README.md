# approximate_program_smoothing
This is the source code for compiler associated with paper "Approximate Program Smoothing Using Mean-Variance Statistics, with Application to Procedural Shader Bandlimiting" by Yuting Yang, Connelly Barnes, Eurographics 2018.

[Project Page](https://www.cs.virginia.edu/~yy2bb/docs/eg_2018.html)

The goal of this project is to smooth an arbitrary program by approximating the convolution of the program with a Gaussian kernel.

# Installation

First, install the dependencies of proj/csolver and make sure proj/csolver/main can be built and run. See proj/csolver/README.md for instructions on how to do this.

Second, install Python 3 (preferably Anaconda). Install the following Python packages (with `pip install X` or `conda install X`):

    filelock
    matplotlib
    pathos
    pyinterval
    
Then try running one of the shaders, e.g.

    $ cd proj/apps
    $ python render_circles.py
    
# Usage

Example of tuning a shader:

    python tune_shader.py full out_dir shader geometry parallax_mapping
    
Our test suite provides 7 shaders:

    render_bricks
    render_checkerboard
    render_circles
    render_color_circles
    render_fire
    render_sin_quadratic
    render_zigzag
    
And 3 geometries:

    plane
    sphere
    hyperboloid1
    
With 3 parallax mappings:

    none
    ripples
    spheres
    
Please download our example tuning outputs from [here](http://www.cs.virginia.edu/~yy2bb/docs/tuner_result.zip). The rendering outputs are saved to html files in each subdirectory.

To re-render outputs from tuning outputs, run:

    python tune_shader.py render out_dir shader geometry parallax_mapping
    
# Citation

If you find this useful, please cite our paper:

Yuting Yang, Connelly Barnes. Approximate Program Smoothing Using Mean-Variance Statistics, with Application to Procedural Shader Bandlimiting. Eurographics 2018.

# License

This project is licensed under the [MIT license](https://opensource.org/licenses/MIT):

Copyright (c) 2018 University of Virginia, Yuting Yang, Connelly Barnes, and other contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
