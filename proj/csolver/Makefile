include ../Makefile.common

MAIN = main
DEPS =
PRECOMPILED_H = problems.h
PRECOMPILED_H_EXT = $(PRECOMPILED_H).gch
PRECOMPILED_H_DEPS = check_compiler.h problems.h timer.h util.h
OBJ = problems.o util.o main.o timer.o
CHECK_COMPILER = check_compiler.o
OBJ_ALL = $(OBJ) $(CHECK_COMPILER)

OS=$(shell uname)

all: $(MAIN) $(CHECK_COMPILER) nonfinal

nonfinal: compiler_problem $(OBJ)

EXTRAFLAGS=-O3
debug: EXTRAFLAGS=-g
debug: $(MAIN) $(PRECOMPILED_H_EXT)

# Adept flags: -ladept -DADEPT_STACK_THREAD_UNSAFE
# Stan math flags: -I ../../extern/stan_math/ -I ../../extern/stan_math/lib/cvodes_2.9.0/include/

CFLAGS = -std=c++11 $(EXTRAFLAGS) -fmax-errors=3 -I ../../extern/cnpy -I ../../extern/eigen -I ../../extern/lodepng -w -DNDEBUG -lcnpy -lz -L ../../extern/lodepng -L ../../extern/cnpy/build -L ../../extern/opencv/lib -llodepng -fopenmp -lopencv_core -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc
LFLAGS =

ifeq ($(OS), Linux)
	CFLAGS += -Wl,-rpath=../../extern/cnpy/build -lcnpy -Wl,-rpath=../../extern/opencv/lib
endif

compiler_problem:
	-cp -n compiler_problem_orig.h compiler_problem.h

$(MAIN): $(OBJ) $(PRECOMPILED_H_EXT) $(CHECK_COMPILER)
	$(CXX) -o $@ $(OBJ_ALL) $(CFLAGS) $(LFLAGS)

$(PRECOMPILED_H_EXT): $(PRECOMPILED_H_DEPS)
	$(CXX) -c -o $@ $(PRECOMPILED_H) $(CFLAGS) $(LFLAGS)

-include $(OBJ_ALL:.o=.d)

%.o: %.cpp $(DEPS) $(PRECOMPILED_H_EXT)
	$(CXX) -c -o $@ $< $(CFLAGS)
	$(CXX) -MM $*.cpp $(CFLAGS) > $*.d

clean:
	rm -f $(OBJ) $(MAIN) *.d *.gch
