#include <iostream>

#include <Eigen/Core>

#include <boost/test/unit_test.hpp>

#include "eiquadprog/eiquadprog-fast.hpp"

using namespace eiquadprog::solvers;

/**
 * solves the problem
 * min. x' Hess x + 2 g0' x
 * s.t. CE x + ce0 = 0
 *      CI x + ci0 >= 0
 */

BOOST_AUTO_TEST_SUITE ( BOOST_TEST_MODULE )

// min ||x||^2

BOOST_AUTO_TEST_CASE ( test_unbiased )
{
  EiquadprogFast qp;
  qp.reset(2,0,0);

  Eigen::MatrixXd Q(2,2);
  Q.setZero();
  Q(0,0) = 1.0;
  Q(1,1) = 1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(2,0);

  Eigen::VectorXd Beq(0);

  Eigen::MatrixXd Aineq(2,0);

  Eigen::VectorXd Bineq(0);

  Eigen::VectorXd x(2);

  Eigen::VectorXd solution(2);
  solution.setZero();

  double val = 0.0;

  EiquadprogFast_status expected = EIQUADPROG_FAST_OPTIMAL;

  EiquadprogFast_status status = qp.solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x);

  BOOST_CHECK(status==expected);

  BOOST_CHECK_CLOSE(qp.getObjValue(),val,1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

// min ||x-x_0||^2, x_0 = (1 1)^T

BOOST_AUTO_TEST_CASE ( test_biased )
{
  EiquadprogFast qp;
  qp.reset(2,0,0);

  Eigen::MatrixXd Q(2,2);
  Q.setZero();
  Q(0,0) = 1.0;
  Q(1,1) = 1.0;

  Eigen::VectorXd C(2);
  C(0) = -1.;
  C(1) = -1.;

  Eigen::MatrixXd Aeq(2,0);

  Eigen::VectorXd Beq(0);

  Eigen::MatrixXd Aineq(2,0);

  Eigen::VectorXd Bineq(0);

  Eigen::VectorXd x(2);
  Eigen::VectorXi activeSet(0);
  size_t activeSetSize;

  Eigen::VectorXd solution(2);
  solution(0) = 1.;
  solution(1) = 1.;

  double val = -1.;

  EiquadprogFast_status expected = EIQUADPROG_FAST_OPTIMAL;

  EiquadprogFast_status status = qp.solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x);

  BOOST_CHECK(status==expected);

  BOOST_CHECK_CLOSE(qp.getObjValue(),val,1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

BOOST_AUTO_TEST_SUITE_END ()

