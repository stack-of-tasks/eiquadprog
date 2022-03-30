//
// Copyright (c) 2019 CNRS
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

#include "eiquadprog/eiquadprog-fast.hpp"

#include <Eigen/Core>
#include <boost/test/unit_test.hpp>
#include <iostream>

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
  EiquadprogFast qp;
  qp.reset(2, 0, 0);

  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = 1.0;
  Q(1, 1) = 1.0;

  Eigen::VectorXd C(2);
  C(0) = -1.;
  C(1) = -1.;

  Eigen::MatrixXd Aeq(0, 2);

  Eigen::VectorXd Beq(0);

  Eigen::MatrixXd Aineq(0, 2);

  Eigen::VectorXd Bineq(0);

  Eigen::VectorXd x(2);

  Eigen::VectorXd solution(2);
  solution(0) = 1.;
  solution(1) = 1.;

  double val = -1.;

  EiquadprogFast_status expected = EIQUADPROG_FAST_OPTIMAL;

  EiquadprogFast_status status =
      qp.solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x);

  BOOST_CHECK_EQUAL(status, expected);

  BOOST_CHECK_CLOSE(qp.getObjValue(), val, 1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

// min ||x||^2
//    s.t.
// x[1] = 1 - x[0]

BOOST_AUTO_TEST_CASE(test_equality_constraints) {
  EiquadprogFast qp;
  qp.reset(2, 1, 0);

  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = 1.0;
  Q(1, 1) = 1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(1, 2);
  Aeq(0, 0) = 1.;
  Aeq(0, 1) = 1.;

  Eigen::VectorXd Beq(1);
  Beq(0) = -1.;

  Eigen::MatrixXd Aineq(0, 2);

  Eigen::VectorXd Bineq(0);

  Eigen::VectorXd x(2);

  Eigen::VectorXd solution(2);
  solution(0) = 0.5;
  solution(1) = 0.5;

  double val = 0.25;

  EiquadprogFast_status expected = EIQUADPROG_FAST_OPTIMAL;

  EiquadprogFast_status status =
      qp.solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x);

  BOOST_CHECK_EQUAL(status, expected);

  BOOST_CHECK_CLOSE(qp.getObjValue(), val, 1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

// min ||x||^2
//    s.t.
// x[i] >= 1

BOOST_AUTO_TEST_CASE(test_inequality_constraints) {
  EiquadprogFast qp;
  qp.reset(2, 0, 2);

  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = 1.0;
  Q(1, 1) = 1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(0, 2);

  Eigen::VectorXd Beq(0);

  Eigen::MatrixXd Aineq(2, 2);
  Aineq.setZero();
  Aineq(0, 0) = 1.;
  Aineq(1, 1) = 1.;

  Eigen::VectorXd Bineq(2);
  Bineq(0) = -1.;
  Bineq(1) = -1.;

  Eigen::VectorXd x(2);

  Eigen::VectorXd solution(2);
  solution(0) = 1.;
  solution(1) = 1.;

  double val = 1.;

  EiquadprogFast_status expected = EIQUADPROG_FAST_OPTIMAL;

  EiquadprogFast_status status =
      qp.solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x);

  BOOST_CHECK_EQUAL(status, expected);

  BOOST_CHECK_CLOSE(qp.getObjValue(), val, 1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

// min ||x-x_0||^2, x_0 = (1 1)^T
//    s.t.
// x[1] = 5 - x[0]
// x[1] >= 3

BOOST_AUTO_TEST_CASE(test_full) {
  EiquadprogFast qp;
  qp.reset(2, 1, 1);

  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = 1.0;
  Q(1, 1) = 1.0;

  Eigen::VectorXd C(2);
  C(0) = -1.;
  C(1) = -1.;

  Eigen::MatrixXd Aeq(1, 2);
  Aeq(0, 0) = 1.;
  Aeq(0, 1) = 1.;

  Eigen::VectorXd Beq(1);
  Beq(0) = -5.;

  Eigen::MatrixXd Aineq(1, 2);
  Aineq.setZero();
  Aineq(0, 1) = 1.;

  Eigen::VectorXd Bineq(1);
  Bineq(0) = -3.;

  Eigen::VectorXd x(2);

  Eigen::VectorXd solution(2);
  solution(0) = 2.;
  solution(1) = 3.;

  double val = 1.5;

  EiquadprogFast_status expected = EIQUADPROG_FAST_OPTIMAL;

  EiquadprogFast_status status =
      qp.solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x);

  BOOST_CHECK_EQUAL(status, expected);

  BOOST_CHECK_CLOSE(qp.getObjValue(), val, 1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

// min ||x||^2
//    s.t.
// x[0] =  1
// x[0] = -1

BOOST_AUTO_TEST_CASE(test_unfeasible_equalities) {
  EiquadprogFast qp;
  qp.reset(2, 2, 0);

  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = 1.0;
  Q(1, 1) = 1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(2, 2);
  Aeq.setZero();
  Aeq(0, 0) = 1.;
  Aeq(1, 0) = 1.;

  Eigen::VectorXd Beq(2);
  Beq(0) = -1.;
  Beq(1) = 1.;

  Eigen::MatrixXd Aineq(0, 2);

  Eigen::VectorXd Bineq(0);

  Eigen::VectorXd x(2);

  EiquadprogFast_status expected = EIQUADPROG_FAST_REDUNDANT_EQUALITIES;

  EiquadprogFast_status status =
      qp.solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x);

  BOOST_CHECK_EQUAL(status, expected);
}

// min ||x||^2
//    s.t.
// x[0] >=  1
// x[0] <= -1
//
// correctly fails, but returns wrong error code

BOOST_AUTO_TEST_CASE(test_unfeasible_inequalities) {
  EiquadprogFast qp;
  qp.reset(2, 0, 2);

  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = 1.0;
  Q(1, 1) = 1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(0, 2);

  Eigen::VectorXd Beq(0);

  Eigen::MatrixXd Aineq(2, 2);
  Aineq.setZero();
  Aineq(0, 0) = 1.;
  Aineq(1, 0) = -1.;

  Eigen::VectorXd Bineq(2);
  Bineq(0) = -1;
  Bineq(1) = -1;

  Eigen::VectorXd x(2);

  EiquadprogFast_status expected = EIQUADPROG_FAST_INFEASIBLE;

  EiquadprogFast_status status =
      qp.solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x);

  BOOST_WARN_EQUAL(status, expected);
  BOOST_CHECK(status != EIQUADPROG_FAST_OPTIMAL);
}

// min ||x-x_0||^2, x_0 = (1 1)^T
//    s.t.
// x[1] = 1 - x[0]
// x[0] <= 0
// x[1] <= 0
//
// correctly fails, but returns wrong error code

BOOST_AUTO_TEST_CASE(test_unfeasible_constraints) {
  EiquadprogFast qp;
  qp.reset(2, 1, 2);

  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = 1.0;
  Q(1, 1) = 1.0;

  Eigen::VectorXd C(2);
  C(0) = -1.;
  C(1) = -1.;

  Eigen::MatrixXd Aeq(1, 2);
  Aeq(0, 0) = 1.;
  Aeq(0, 1) = 1.;

  Eigen::VectorXd Beq(1);
  Beq(0) = -1.;

  Eigen::MatrixXd Aineq(2, 2);
  Aineq.setZero();
  Aineq(0, 0) = -1.;
  Aineq(1, 1) = -1.;

  Eigen::VectorXd Bineq(2);
  Bineq.setZero();

  Eigen::VectorXd x(2);

  EiquadprogFast_status expected = EIQUADPROG_FAST_INFEASIBLE;

  EiquadprogFast_status status =
      qp.solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x);

  BOOST_WARN_EQUAL(status, expected);
  BOOST_CHECK(status != EIQUADPROG_FAST_OPTIMAL);
}

// min -||x||^2
// DOES NOT WORK!

BOOST_AUTO_TEST_CASE(test_unbounded) {
  EiquadprogFast qp;
  qp.reset(2, 0, 0);

  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = -1.0;
  Q(1, 1) = -1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(0, 2);

  Eigen::VectorXd Beq(0);

  Eigen::MatrixXd Aineq(0, 2);

  Eigen::VectorXd Bineq(0);

  Eigen::VectorXd x(2);

  EiquadprogFast_status expected = EIQUADPROG_FAST_UNBOUNDED;

  EiquadprogFast_status status =
      qp.solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x);

  BOOST_WARN_EQUAL(status, expected);
  BOOST_WARN(status != EIQUADPROG_FAST_OPTIMAL);  // SHOULD pass!
}

// min -||x||^2
//    s.t.
// 0<= x[0] <= 1
// 0<= x[1] <= 1
// DOES NOT WORK!

BOOST_AUTO_TEST_CASE(test_nonconvex) {
  EiquadprogFast qp;
  qp.reset(2, 0, 4);

  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = -1.0;
  Q(1, 1) = -1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(0, 2);

  Eigen::VectorXd Beq(0);

  Eigen::MatrixXd Aineq(4, 2);
  Aineq.setZero();
  Aineq(0, 0) = 1.;
  Aineq(1, 0) = -1.;
  Aineq(2, 1) = 1.;
  Aineq(3, 1) = -1.;

  Eigen::VectorXd Bineq(4);
  Bineq(0) = 0.;
  Bineq(1) = 1.;
  Bineq(2) = 0.;
  Bineq(3) = 1.;

  Eigen::VectorXd x(2);

  Eigen::VectorXd solution(2);
  solution(0) = 1.;
  solution(1) = 1.;

  double val = -1.;

  EiquadprogFast_status expected = EIQUADPROG_FAST_OPTIMAL;

  EiquadprogFast_status status =
      qp.solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x);

  BOOST_CHECK_EQUAL(status, expected);

  BOOST_WARN_CLOSE(qp.getObjValue(), val, 1e-6);

  BOOST_WARN(x.isApprox(solution));
}

BOOST_AUTO_TEST_SUITE_END()
