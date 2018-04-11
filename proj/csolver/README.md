
## Build instructions

* If [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html) is not in your compiler's header directories, install Eigen in `../../extern/eigen`
* Install [opencv](https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html) and place shared object files in `../../extern/opencv/lib`
* Install [lodepng](https://github.com/lvandeve/lodepng) in '../../extern/lodepng' and use build commands such as:

    g++-5 -c lodepng.cpp -O3
    ar rs liblodepng.a lodepng.o
    
* Install [cnpy](https://github.com/rogersce/cnpy) in '../../extern/cnpy' and use commit 0fcddfe to build, using commands such as: 
    
    git checkout 0fcddfe
    mkdir build
    cd build
    CXX=g++-5; export CC=gcc-5; cmake ..; make; sudo make install
    
* Run `make`, then test the main program by running `./main`

## Debugging instructions

* Build in debug mode: `make clean; make debug`
* Debug with gdb: `gdb --args ./main [args]`
* Debug with valgrind: `valgrind --gen-suppressions=yes --track-origins=yes ./main [args]`
