# approximate_program_smoothing
This is the source code for compiler associated with paper "Approximate Program Smoothing Using Mean-Variance Statistics, with Application to Procedural Shader Bandlimiting" by Yuting Yang, Connelly Barnes, Eurographics 2018.

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