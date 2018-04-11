import skimage.filters

def metric_L2(ground_I, predict_I):
    return ((ground_I-predict_I)**2).mean()**0.5

def highfreq(I):
    return I - skimage.filters.gaussian(I, I.shape[0]/200.0)

def metric_highfreq(ground_I, predict_I):
    return metric_L2(highfreq(ground_I), highfreq(predict_I))

