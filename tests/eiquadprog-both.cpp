//
// Copyright (c) 2020 CNRS
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

#include <Eigen/Core>
#include <boost/test/unit_test.hpp>
#include <iostream>

#include "eiquadprog/eiquadprog-fast.hpp"
#include "eiquadprog/eiquadprog-rt.hpp"

using namespace eiquadprog::solvers;

/**
 * solves the problem
 * min. 0.5 * x' Hess x + g0' x
 * s.t. CE x + ce0 = 0
 *      CI x + ci0 >= 0
 */

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

// min ||x||^2

BOOST_AUTO_TEST_CASE(test_unbiased) {
  EiquadprogFast qp;
  qp.reset(2, 0, 0);

  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = 1.0;
  Q(1, 1) = 1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(0, 2);

  Eigen::VectorXd Beq(0);

  Eigen::MatrixXd Aineq(0, 2);

  Eigen::VectorXd Bineq(0);

  Eigen::VectorXd x(2);

  Eigen::VectorXd solution(2);
  solution.setZero();

  double val = 0.0;

  EiquadprogFast_status expected = EIQUADPROG_FAST_OPTIMAL;

  EiquadprogFast_status status =
      qp.solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x);

  BOOST_CHECK_EQUAL(status, expected);

  BOOST_CHECK_CLOSE(qp.getObjValue(), val, 1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

// min ||x-x_0||^2, x_0 = (1 1)^T

BOOST_AUTO_TEST_CASE(test_biased) {
  RtEiquadprog<2, 0, 0> qp;

  RtMatrixX<2, 2>::d Q;
  Q.setZero();
  Q(0, 0) = 1.0;
  Q(1, 1) = 1.0;

  RtVectorX<2>::d C;
  C(0) = -1.;
  C(1) = -1.;

  RtMatrixX<0, 2>::d Aeq;

  RtVectorX<0>::d Beq;

  RtMatrixX<0, 2>::d Aineq;

  RtVectorX<0>::d Bineq;

  RtVectorX<2>::d x;

  RtVectorX<2>::d solution;
  solution(0) = 1.;
  solution(1) = 1.;

  double val = -1.;

  RtEiquadprog_status expected = RT_EIQUADPROG_OPTIMAL;

  RtEiquadprog_status status =
      qp.solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x);

  BOOST_CHECK_EQUAL(status, expected);

  BOOST_CHECK_CLOSE(qp.getObjValue(), val, 1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

// min ||x||^2
//    s.t.
// x[1] = 1 - x[0]

BOOST_AUTO_TEST_CASE(test_equality_constraints) {
  RtEiquadprog<2, 1, 0> qp;

  RtMatrixX<2, 2>::d Q;
  Q.setZero();
  Q(0, 0) = 1.0;
  Q(1, 1) = 1.0;

  RtVectorX<2>::d C;
  C.setZero();

  RtMatrixX<1, 2>::d Aeq;
  Aeq(0, 0) = 1.;
  Aeq(0, 1) = 1.;

  RtVectorX<1>::d Beq;
  Beq(0) = -1.;

  RtMatrixX<0, 2>::d Aineq;

  RtVectorX<0>::d Bineq;

  RtVectorX<2>::d x;

  RtVectorX<2>::d solution;
  solution(0) = 0.5;
  solution(1) = 0.5;

  double val = 0.25;

  RtEiquadprog_status expected = RT_EIQUADPROG_OPTIMAL;

  RtEiquadprog_status status =
      qp.solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x);

  BOOST_CHECK_EQUAL(status, expected);

  BOOST_CHECK_CLOSE(qp.getObjValue(), val, 1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

BOOST_AUTO_TEST_SUITE_END()
