import math
import numpy

def bandlimit_poly(n, ret_coeffs=False, var=False):
    """
    Return bandlimited function g(x, w) for f(x) = x ** n, with a Gaussian kernel.
    
    If ret_coeffs is True then return a function coeff_func(w) that given w returns the x polynomial coefficients.
    
    If var is True then make g be a function of the variance v = sqrt(w) instead.
    """
    fact = math.factorial
    k_range = int(n/2.0)+1
    coeffs = [fact(n) / (2**k * fact(n-2*k) * fact(k)) for k in range(k_range)]
#    print(coeffs)

    if ret_coeffs:
        def coeff_func(w):
            xcoeffs = numpy.zeros(n+1)
            for k in range(k_range):
                xcoeffs[2*k] = w**(2*k if not var else k)*coeffs[k]
            return xcoeffs
        return coeff_func
    else:
        def g(x, w):
            return sum([x**(n-2*k)*w**(2*k if not var else k)*coeffs[k] for k in range(k_range)])
        return g

