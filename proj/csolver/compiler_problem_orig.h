
/* compiler_problem.h is a problem automatically generated by our compiler infrastructure.
   compiler_problem_orig.h is a stub version of the same file that allows the project to initially compile.
   Only include this header from one source file. */

#ifndef _compiler_problem_h
#define _compiler_problem_h

#include "problems.h"

class CompilerProblem: public Problem { public:
    CompilerProblem(int ndims_) {
        ndims = ndims_;
        bound_lo = VectorXd::Constant(ndims, -1.0);
        bound_hi = VectorXd::Constant(ndims, 1.0);
        sample_problem();
    }
    
    void sample_problem() {
    }
    
    double f(const VectorXd &L) {
        double ans = 0.0;
        for (int i = 0; i < ndims; i++) {
            ans += L[i]*L[i];
        }
        return ans;
    }
    
    double g(const VectorXd &L, const VectorXd &w) {
        return f(L);
    }
};

#endif
