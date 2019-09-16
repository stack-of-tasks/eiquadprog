#include <iostream>

#include <Eigen/Core>

#include <boost/test/unit_test.hpp>

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

BOOST_AUTO_TEST_SUITE ( BOOST_TEST_MODULE )

// min ||x||^2

BOOST_AUTO_TEST_CASE ( test_unbiased )
{
  Eigen::MatrixXd Q(2,2);
  Q.setZero();
  Q(0,0) = 1.0;
  Q(1,1) = 1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(0,2);

  Eigen::VectorXd Beq(0);

  Eigen::MatrixXd Aineq(0,2);

  Eigen::VectorXd Bineq(0);

  Eigen::VectorXd x(2);
  Eigen::VectorXi activeSet(0);
  size_t activeSetSize;

  Eigen::VectorXd solution(2);
  solution.setZero();

  double res = 0.0;

  double val = Eigen::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x, activeSet, activeSetSize);

  BOOST_CHECK_CLOSE(val,res,1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

BOOST_AUTO_TEST_SUITE_END ()

