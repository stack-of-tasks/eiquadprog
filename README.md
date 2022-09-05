# eiquadprog

[![Pipeline status](https://gitlab.laas.fr/stack-of-tasks/eiquadprog/badges/master/pipeline.svg)](https://gitlab.laas.fr/stack-of-tasks/eiquadprog/commits/master)
[![Coverage report](https://gitlab.laas.fr/stack-of-tasks/eiquadprog/badges/master/coverage.svg?job=doc-coverage)](https://gepettoweb.laas.fr/doc/stack-of-tasks/eiquadprog/master/coverage/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/stack-of-tasks/eiquadprog/master.svg)](https://results.pre-commit.ci/latest/github/stack-of-tasks/eiquadprog)

This repo contains different C++ implementations of the algorithm of Goldfarb and Idnani for the solution of a (convex) Quadratic Programming problem by means of a dual method.

The problem is in the form:
 min 0.5 * x G x + g0 x
 s.t.
 CE^T x + ce0 = 0
 CI^T x + ci0 >= 0

There are 3 implementations:
- `eiquadprog.hpp`: the original C++ implementation
- `eiquadprog-fast.hpp`: an improved version employing a wrapper, avoiding dynamic memory allocation
- `eiquadprog-rt.hpp`: similar to the above, it employs fixed-size Eigen vectors. This requires the problem dimensions to be known at compile time and is recommended only for small problems.

Please refer to the unit tests for examples of usage.

## Authors

[Eiquadprog](https://github.com/stack-of-tasks/eiquadprog) was created at LAAS-CNRS by Gabriele Buondonno, based on
parts from [TSID](https://github.com/stack-of-tasks/tsid) by Andrea Del Prete.

This work was based on previous libraries:
- [QuadProg++](https://github.com/liuq/QuadProgpp), Copyright (C) 2007-2016 Luca Di Gaspero, MIT License.
- [uQuadProg](https://github.com/fx74/uQuadProg), Copyright (C) 2006 - 2017 Angelo Furfaro, LGPL v3.
- [eiquadprog.hpp](http://www.cs.cmu.edu/~bstephe1/eiquadprog.hpp), Copyright (2011) Benjamin Stephens, GPL v2.
- [quadprog++](https://gitlab.inria.fr/alta/alta/blob/3c11d5a4ed6cd15ed39f938a1da1aecad1a4b31e/external/quadprog++/QuadProg++.cc)
  Copyright (C) 2014-2015 Gael Guennebaud, LGPL v3.
