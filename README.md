# eiquadprog

[![Pipeline status](https://gepgitlab.laas.fr/stack-of-tasks/eiquadprog/badges/master/pipeline.svg)](https://gepgitlab.laas.fr/stack-of-tasks/eiquadprog/commits/master)
[![Coverage report](https://gepgitlab.laas.fr/stack-of-tasks/eiquadprog/badges/master/coverage.svg?job=doc-coverage)](http://projects.laas.fr/gepetto/doc/stack-of-tasks/eiquadprog/master/coverage/)

This repo contains a C++ implementation of the algorithm of Goldfarb and Idnani for the solution of a (convex) Quadratic Programming problem by means of a dual method.

The problem is in the form:
 min 0.5 * x G x + g0 x
 s.t.
 CE^T x + ce0 = 0
 CI^T x + ci0 >= 0
