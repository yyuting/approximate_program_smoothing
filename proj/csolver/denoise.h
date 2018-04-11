#ifndef _denoise_h
#define _denoise_h
#include "util.h"

#define fTiny 0.00000001
#define THREAD_PARALLEL2 0

using namespace std;

VectorXd denoise(const VectorXd &input, int width, int height, int channels, double sigma);

#endif