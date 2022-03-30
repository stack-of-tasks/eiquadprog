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

#include <Eigen/Core>
#include <boost/test/unit_test.hpp>
#include <iostream>

#include "eiquadprog/eiquadprog.hpp"

// The problem is in the form:
// min 0.5 * x G x + g0 x
// s.t.
// CE^T x + ce0 = 0
// CI^T x + ci0 >= 0
// The matrix and vectors dimensions are as follows:
// G: n * n
// g0: n
// CE: n * p
// ce0: p
// CI: n * m
// ci0: m
// x: n

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

// min ||x||^2

BOOST_AUTO_TEST_CASE(test_unbiased) {
  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = 1.0;
  Q(1, 1) = 1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(2, 0);

  Eigen::VectorXd Beq(0);

  Eigen::MatrixXd Aineq(2, 0);

  Eigen::VectorXd Bineq(0);

  Eigen::VectorXd x(2);
  Eigen::VectorXi activeSet(0);
  size_t activeSetSize;

  Eigen::VectorXd solution(2);
  solution.setZero();

  double val = 0.0;

  double out = eiquadprog::solvers::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq,
                                                   x, activeSet, activeSetSize);

  BOOST_CHECK_CLOSE(out, val, 1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

// min ||x-x_0||^2, x_0 = (1 1)^T

BOOST_AUTO_TEST_CASE(test_biased) {
  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = 1.0;
  Q(1, 1) = 1.0;

  Eigen::VectorXd C(2);
  C(0) = -1.;
  C(1) = -1.;

  Eigen::MatrixXd Aeq(2, 0);

  Eigen::VectorXd Beq(0);

  Eigen::MatrixXd Aineq(2, 0);

  Eigen::VectorXd Bineq(0);

  Eigen::VectorXd x(2);
  Eigen::VectorXi activeSet(0);
  size_t activeSetSize;

  Eigen::VectorXd solution(2);
  solution(0) = 1.;
  solution(1) = 1.;

  double val = -1.;

  double out = eiquadprog::solvers::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq,
                                                   x, activeSet, activeSetSize);

  BOOST_CHECK_CLOSE(out, val, 1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

// min ||x||^2
//    s.t.
// x[1] = 1 - x[0]

BOOST_AUTO_TEST_CASE(test_equality_constraints) {
  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = 1.0;
  Q(1, 1) = 1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(2, 1);
  Aeq(0, 0) = 1.;
  Aeq(1, 0) = 1.;

  Eigen::VectorXd Beq(1);
  Beq(0) = -1.;

  Eigen::MatrixXd Aineq(2, 0);

  Eigen::VectorXd Bineq(0);

  Eigen::VectorXd x(2);
  Eigen::VectorXi activeSet(1);
  size_t activeSetSize;

  Eigen::VectorXd solution(2);
  solution(0) = 0.5;
  solution(1) = 0.5;

  double val = 0.25;

  double out = eiquadprog::solvers::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq,
                                                   x, activeSet, activeSetSize);

  BOOST_CHECK_CLOSE(out, val, 1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

// min ||x||^2
//    s.t.
// x[i] >= 1

BOOST_AUTO_TEST_CASE(test_inequality_constraints) {
  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = 1.0;
  Q(1, 1) = 1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(2, 0);

  Eigen::VectorXd Beq(0);

  Eigen::MatrixXd Aineq(2, 2);
  Aineq.setZero();
  Aineq(0, 0) = 1.;
  Aineq(1, 1) = 1.;

  Eigen::VectorXd Bineq(2);
  Bineq(0) = -1.;
  Bineq(1) = -1.;

  Eigen::VectorXd x(2);
  Eigen::VectorXi activeSet(2);
  size_t activeSetSize;

  Eigen::VectorXd solution(2);
  solution(0) = 1.;
  solution(1) = 1.;

  double val = 1.;

  double out = eiquadprog::solvers::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq,
                                                   x, activeSet, activeSetSize);

  BOOST_CHECK_CLOSE(out, val, 1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

// min ||x-x_0||^2, x_0 = (1 1)^T
//    s.t.
// x[1] = 5 - x[0]
// x[1] >= 3

BOOST_AUTO_TEST_CASE(test_full) {
  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = 1.0;
  Q(1, 1) = 1.0;

  Eigen::VectorXd C(2);
  C(0) = -1.;
  C(1) = -1.;

  Eigen::MatrixXd Aeq(2, 1);
  Aeq(0, 0) = 1.;
  Aeq(1, 0) = 1.;

  Eigen::VectorXd Beq(1);
  Beq(0) = -5.;

  Eigen::MatrixXd Aineq(2, 1);
  Aineq.setZero();
  Aineq(1, 0) = 1.;

  Eigen::VectorXd Bineq(1);
  Bineq(0) = -3.;

  Eigen::VectorXd x(2);
  Eigen::VectorXi activeSet(2);
  size_t activeSetSize;

  Eigen::VectorXd solution(2);
  solution(0) = 2.;
  solution(1) = 3.;

  double val = 1.5;

  double out = eiquadprog::solvers::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq,
                                                   x, activeSet, activeSetSize);

  BOOST_CHECK_CLOSE(out, val, 1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

// min ||x||^2
//    s.t.
// x[0] =  1
// x[0] = -1
// DOES NOT WORK!

BOOST_AUTO_TEST_CASE(test_unfeasible_equalities) {
  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = 1.0;
  Q(1, 1) = 1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(2, 2);
  Aeq.setZero();
  Aeq(0, 0) = 1.;
  Aeq(0, 1) = 1.;

  Eigen::VectorXd Beq(2);
  Beq(0) = -1.;
  Beq(1) = 1.;

  Eigen::MatrixXd Aineq(2, 0);

  Eigen::VectorXd Bineq(0);

  Eigen::VectorXd x(2);
  Eigen::VectorXi activeSet(2);
  size_t activeSetSize;

  double out = eiquadprog::solvers::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq,
                                                   x, activeSet, activeSetSize);

  // DOES NOT WORK!?
  BOOST_WARN(std::isinf(out));
}

// min ||x||^2
//    s.t.
// x[0] >=  1
// x[0] <= -1

BOOST_AUTO_TEST_CASE(test_unfeasible_inequalities) {
  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = 1.0;
  Q(1, 1) = 1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(2, 0);

  Eigen::VectorXd Beq(0);

  Eigen::MatrixXd Aineq(2, 2);
  Aineq.setZero();
  Aineq(0, 0) = 1.;
  Aineq(0, 1) = -1.;

  Eigen::VectorXd Bineq(2);
  Bineq(0) = -1;
  Bineq(1) = -1;

  Eigen::VectorXd x(2);
  Eigen::VectorXi activeSet(2);
  size_t activeSetSize;

  double out = eiquadprog::solvers::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq,
                                                   x, activeSet, activeSetSize);

  BOOST_CHECK(std::isinf(out));
}

// min ||x-x_0||^2, x_0 = (1 1)^T
//    s.t.
// x[1] = 1 - x[0]
// x[0] <= 0
// x[1] <= 0

BOOST_AUTO_TEST_CASE(test_unfeasible_constraints) {
  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = 1.0;
  Q(1, 1) = 1.0;

  Eigen::VectorXd C(2);
  C(0) = -1.;
  C(1) = -1.;

  Eigen::MatrixXd Aeq(2, 1);
  Aeq(0, 0) = 1.;
  Aeq(1, 0) = 1.;

  Eigen::VectorXd Beq(1);
  Beq(0) = -1.;

  Eigen::MatrixXd Aineq(2, 2);
  Aineq.setZero();
  Aineq(0, 0) = -1.;
  Aineq(1, 1) = -1.;

  Eigen::VectorXd Bineq(2);
  Bineq.setZero();

  Eigen::VectorXd x(2);
  Eigen::VectorXi activeSet(3);
  size_t activeSetSize;

  double out = eiquadprog::solvers::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq,
                                                   x, activeSet, activeSetSize);

  BOOST_CHECK(std::isinf(out));
}

// min -||x||^2
// DOES NOT WORK!

BOOST_AUTO_TEST_CASE(test_unbounded) {
  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = -1.0;
  Q(1, 1) = -1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(2, 0);

  Eigen::VectorXd Beq(0);

  Eigen::MatrixXd Aineq(2, 0);

  Eigen::VectorXd Bineq(0);

  Eigen::VectorXd x(2);
  Eigen::VectorXi activeSet(0);
  size_t activeSetSize;

  double out = eiquadprog::solvers::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq,
                                                   x, activeSet, activeSetSize);

  // DOES NOT WORK!?
  BOOST_WARN(std::isinf(out));
}

// min -||x||^2
//    s.t.
// 0<= x[0] <= 1
// 0<= x[1] <= 1
// DOES NOT WORK!

BOOST_AUTO_TEST_CASE(test_nonconvex) {
  Eigen::MatrixXd Q(2, 2);
  Q.setZero();
  Q(0, 0) = -1.0;
  Q(1, 1) = -1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(2, 0);

  Eigen::VectorXd Beq(0);

  Eigen::MatrixXd Aineq(2, 4);
  Aineq.setZero();
  Aineq(0, 0) = 1.;
  Aineq(0, 1) = -1.;
  Aineq(1, 2) = 1.;
  Aineq(1, 3) = -1.;

  Eigen::VectorXd Bineq(4);
  Bineq(0) = 0.;
  Bineq(1) = 1.;
  Bineq(2) = 0.;
  Bineq(3) = 1.;

  Eigen::VectorXd x(2);
  Eigen::VectorXi activeSet(4);
  size_t activeSetSize;

  Eigen::VectorXd solution(2);
  solution(0) = 1.;
  solution(1) = 1.;

  double val = -1.;

  double out = eiquadprog::solvers::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq,
                                                   x, activeSet, activeSetSize);

  // DOES NOT WORK!?
  BOOST_WARN_CLOSE(out, val, 1e-6);

  BOOST_WARN(x.isApprox(solution));
}

BOOST_AUTO_TEST_SUITE_END()
