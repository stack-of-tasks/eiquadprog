//
// Copyright (c) 2019,2022 CNRS INRIA
//
// This file is part of eiquadprog.
//
// eiquadprog is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

// eiquadprog is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public License
// along with eiquadprog.  If not, see <https://www.gnu.org/licenses/>.

#ifndef _EIGEN_QUADSOLVE_HPP_
#define _EIGEN_QUADSOLVE_HPP_

/*
 FILE eiquadprog.hpp
 NOTE: this is a modified of uQuadProg++ package, working with Eigen data
 structures. uQuadProg++ is itself a port made by Angelo Furfaro of QuadProg++
 originally developed by Luca Di Gaspero, working with ublas data structures.
 The quadprog_solve() function implements the algorithm of Goldfarb and Idnani
 for the solution of a (convex) Quadratic Programming problem
 by means of a dual method.
 The problem is in the form:
 min 0.5 * x G x + g0 x
 s.t.
 CE^T x + ce0 = 0
 CI^T x + ci0 >= 0
 The matrix and vectors dimensions are as follows:
 G: n * n
 g0: n
 CE: n * p
 ce0: p
 CI: n * m
 ci0: m
 x: n
 The function will return the cost of the solution written in the x vector or
 std::numeric_limits::infinity() if the problem is infeasible. In the latter
 case the value of the x vector is not correct. References: D. Goldfarb, A.
 Idnani. A numerically stable dual method for solving strictly convex quadratic
 programs. Mathematical Programming 27 (1983) pp. 1-33. Notes:
 1. pay attention in setting up the vectors ce0 and ci0.
 If the constraints of your problem are specified in the form
 A^T x = b and C^T x >= d, then you should set ce0 = -b and ci0 = -d.
 2. The matrix G is modified within the function since it is used to compute
 the G = L^T L cholesky factorization for further computations inside the
 function. If you need the original matrix G you should make a copy of it and
 pass the copy to the function. The author will be grateful if the researchers
 using this software will acknowledge the contribution of this modified function
 and of Di Gaspero's original version in their research papers. LICENSE
 Copyright (2011) Benjamin Stephens
 Copyright (2010) Gael Guennebaud
 Copyright (2008) Angelo Furfaro
 Copyright (2006) Luca Di Gaspero
 This file is a porting of QuadProg++ routine, originally developed
 by Luca Di Gaspero, exploiting uBlas data structures for vectors and
 matrices instead of native C++ array.
 uquadprog is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.
 uquadprog is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 You should have received a copy of the GNU General Public License
 along with uquadprog; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "eiquadprog/deprecated.hpp"
#include "eiquadprog/eiquadprog-utils.hxx"

// namespace internal {
namespace eiquadprog {
namespace solvers {

inline void compute_d(Eigen::VectorXd &d, const Eigen::MatrixXd &J,
                      const Eigen::VectorXd &np) {
  d.noalias() = J.adjoint() * np;
}

inline void update_z(Eigen::VectorXd &z, const Eigen::MatrixXd &J,
                     const Eigen::VectorXd &d, size_t iq) {
  z.noalias() = J.rightCols(z.size() - iq) * d.tail(d.size() - iq);
}

inline void update_r(const Eigen::MatrixXd &R, Eigen::VectorXd &r,
                     const Eigen::VectorXd &d, size_t iq) {
  r.head(iq) = d.head(iq);
  R.topLeftCorner(iq, iq).triangularView<Eigen::Upper>().solveInPlace(
      r.head(iq));
}

bool add_constraint(Eigen::MatrixXd &R, Eigen::MatrixXd &J, Eigen::VectorXd &d,
                    size_t &iq, double &R_norm);
void delete_constraint(Eigen::MatrixXd &R, Eigen::MatrixXd &J,
                       Eigen::VectorXi &A, Eigen::VectorXd &u, size_t p,
                       size_t &iq, size_t l);

double solve_quadprog(Eigen::LLT<Eigen::MatrixXd, Eigen::Lower> &chol,
                      double c1, Eigen::VectorXd &g0, const Eigen::MatrixXd &CE,
                      const Eigen::VectorXd &ce0, const Eigen::MatrixXd &CI,
                      const Eigen::VectorXd &ci0, Eigen::VectorXd &x,
                      Eigen::VectorXi &A, size_t &q);

double solve_quadprog(Eigen::LLT<Eigen::MatrixXd, Eigen::Lower> &chol,
                      double c1, Eigen::VectorXd &g0, const Eigen::MatrixXd &CE,
                      const Eigen::VectorXd &ce0, const Eigen::MatrixXd &CI,
                      const Eigen::VectorXd &ci0, Eigen::VectorXd &x,
                      Eigen::VectorXd &y, Eigen::VectorXi &A, size_t &q);

EIQUADPROG_DEPRECATED
double solve_quadprog2(Eigen::LLT<Eigen::MatrixXd, Eigen::Lower> &chol,
                       double c1, Eigen::VectorXd &g0,
                       const Eigen::MatrixXd &CE, const Eigen::VectorXd &ce0,
                       const Eigen::MatrixXd &CI, const Eigen::VectorXd &ci0,
                       Eigen::VectorXd &x, Eigen::VectorXi &A, size_t &q) {
  return solve_quadprog(chol, c1, g0, CE, ce0, CI, ci0, x, A, q);
}

/* solve_quadprog is used for on-demand QP solving */
double solve_quadprog(Eigen::MatrixXd &G, Eigen::VectorXd &g0,
                      const Eigen::MatrixXd &CE, const Eigen::VectorXd &ce0,
                      const Eigen::MatrixXd &CI, const Eigen::VectorXd &ci0,
                      Eigen::VectorXd &x, Eigen::VectorXi &activeSet,
                      size_t &activeSetSize);

double solve_quadprog(Eigen::MatrixXd &G, Eigen::VectorXd &g0,
                      const Eigen::MatrixXd &CE, const Eigen::VectorXd &ce0,
                      const Eigen::MatrixXd &CI, const Eigen::VectorXd &ci0,
                      Eigen::VectorXd &x, Eigen::VectorXd &y,
                      Eigen::VectorXi &activeSet, size_t &activeSetSize);
// }

}  // namespace solvers
}  // namespace eiquadprog

#endif
