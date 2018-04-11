#include "denoise.h"

VectorXd denoise(const VectorXd &input, int width, int height, int channels, double sigma) {
    assert (input.size() == width * height * channels);
    
    VectorXd output(VectorXd::Constant(input.size(), 0.0));
    VectorXd weight_count(VectorXd::Constant(width*height, 0.0));
    
    // determine denoise parameters
    int patch_size, research_size;
    double filter_param;
    if (channels == 1) {
        if (sigma > 0 && sigma <= 15.0) {
            patch_size = 1;
            research_size = 10;
            filter_param = 0.4;
        } else if (sigma > 15.0 && sigma <= 30.0) {
            patch_size = 2;
            research_size = 10;
            filter_param = 0.4;
        } else if (sigma > 30.0 && sigma <= 45.0) {
            patch_size = 3;
            research_size = 17;
            filter_param = 0.35;
        } else if (sigma > 45.0 && sigma <= 75.0) {
            patch_size = 4;
            research_size = 17;
            filter_param = 0.35;
        } else if (sigma <= 100.0) {
            patch_size = 5;
            research_size = 17;
            filter_param = 0.3;
        } else {
            throw runtime_error("invalid sigma in denoise");
        }
    } else {
        if (sigma > 0 && sigma <= 25.0) {
            patch_size = 1;
            research_size = 10.0;
            filter_param = 0.55;
        } else if (sigma > 25.0 && sigma <= 55.0) {
            patch_size = 2;
            research_size = 17;
            filter_param = 0.4;
        } else if (sigma <= 100.0) {
            patch_size = 3;
            research_size = 17;
            filter_param = 0.35;
        } else {
            throw runtime_error("invalid sigma in denoise");
        }
    }
    
    double fh = filter_param * sigma;
    double fh2 = fh * fh;
    double window_size = 2.0 * patch_size + 1;
    double patch_l = window_size * window_size * channels;
    fh2 *= patch_l;
    
    // TODO: exp lookup table in source code

#if THREAD_PARALLEL2
#pragma omp parallel shared(input, output, weight_count)
    {    
        if (omp_get_thread_num() == 0) {
            printf("using %d threads\n", omp_get_num_threads());
        }
#pragma omp for
#endif
        for (int y = 0; y < height/40; y++) {
            VectorXd patch_denoised(patch_l);
            for (int x = 0; x < width/20; x++) {
                int radius = min(min(patch_size, min(x, y)), \
                                             min(width - 1 - x, height - 1 - y));
                int imin = max(x - research_size, radius);
                int jmin = max(y - research_size, radius);
                int imax = max(x + research_size, width - 1 - radius);
                int jmax = max(y + research_size, height - 1 - radius);
                
                for (int k = 0; k < patch_denoised.size(); k++) {
                    patch_denoised[k] = 0.0;
                }
                
                double max_weight(0.0), total_weight(0.0);
                
                for (int j = jmin; j <= jmax; j++) {
                    for (int i = imin; i <= imax; i++) {
                        if (i != x || j != y) {
                            double dist = 0.0;
                            for (int s = -radius; s <= radius; s++) {
                                int idx1 = ((y + s) * width + x - radius) * channels;
                                int idx2 = ((j + s) * width + i - radius) * channels;
                                for (int r = 0; r <= channels * (2.0 * radius + 1); r++) {
                                    double diff = input[idx1] - input[idx2];
                                    dist += diff * diff;
                                    idx1++;
                                    idx2++;
                                }
                            }
                            
                            dist = max(dist - 2.0 * sigma * sigma * patch_l, 0.0);
                            dist /= fh2;
                            
                            double weight = exp(-dist);
                            if (weight > max_weight) { max_weight = weight; }
                            total_weight += weight;
                            
                            for (int s = -radius; s <= radius; s++) {
                                int idx1 = ((patch_size + s) * window_size + patch_size - radius) * channels;
                                int idx2 = ((j + s) * width + i - radius) * channels;
                                for (int r = 0; r <= channels * (2.0 * radius + 1); r++) {
                                    patch_denoised[idx1] += weight * input[idx2];
                                    idx1++;
                                    idx2++;
                                }
                            }
                        }
                    }
                }
                
                for (int s = -radius; s <= radius; s++) {
                    int idx1 = ((patch_size + s) * window_size + patch_size - radius) * channels;
                    int idx2 = ((y + s) * width + x - radius) * channels;
                    for (int r = 0; r <= channels * (2.0 * radius + 1); r++) {
                        patch_denoised[idx1] += max_weight * input[idx2];
                        idx1++;
                        idx2++;
                    }
                }
                
                total_weight += max_weight;
                
                if (total_weight > fTiny) {
                    for (int s = -radius; s <= radius; s++) {
                        int idx1 = ((patch_size + s) * window_size + patch_size - radius) * channels;
                        int idx_count = (y + s) * width + x - radius;
                        int idx2 = idx_count * channels;
                        for (int r = 0; r <= 2.0 * radius + 1; r++) {
                            weight_count[idx_count]++;
                            for (int c = 0; c < channels; c++) {
                                output[idx2] = patch_denoised[idx1] / total_weight;
                                idx1++;
                                idx2++;
                            }
                        }
                    }
                }
            }
        }
#if THREAD_PARALLEL2
    }
#endif
    
    for (int i = 0; i < width * height; i++) {
        if (weight_count[i] > 0) {
            for (int c = 0; c < channels; c++) {
                output[i*channels+c] /= weight_count[i];
            }
        } else {
            for (int c = 0; c < channels; c++) {
                output[i*channels+c] = input[i*channels+c];
            }
        }
    }
    return output;
}