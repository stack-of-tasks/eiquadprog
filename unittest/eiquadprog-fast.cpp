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

// min ||x||^2
//    s.t.
// x[1] = 1 - x[0]

BOOST_AUTO_TEST_CASE ( test_equality_constraints )
{
  EiquadprogFast qp;
  qp.reset(2,1,0);

  Eigen::MatrixXd Q(2,2);
  Q.setZero();
  Q(0,0) = 1.0;
  Q(1,1) = 1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(1,2);
  Aeq(0,0) = 1.;
  Aeq(0,1) = 1.;

  Eigen::VectorXd Beq(1);
  Beq(0) = -1.;

  Eigen::MatrixXd Aineq(0,2);

  Eigen::VectorXd Bineq(0);

  Eigen::VectorXd x(2);

  Eigen::VectorXd solution(2);
  solution(0) = 0.5;
  solution(1) = 0.5;

  double val = 0.25;

  EiquadprogFast_status expected = EIQUADPROG_FAST_OPTIMAL;

  EiquadprogFast_status status = qp.solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x);

  BOOST_CHECK(status==expected);

  BOOST_CHECK_CLOSE(qp.getObjValue(),val,1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

// min ||x||^2
//    s.t.
// x[i] >= 1

BOOST_AUTO_TEST_CASE ( test_inequality_constraints )
{
  EiquadprogFast qp;
  qp.reset(2,0,2);

  Eigen::MatrixXd Q(2,2);
  Q.setZero();
  Q(0,0) = 1.0;
  Q(1,1) = 1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(0,2);

  Eigen::VectorXd Beq(0);

  Eigen::MatrixXd Aineq(2,2);
  Aineq.setZero();
  Aineq(0,0) = 1.;
  Aineq(1,1) = 1.;

  Eigen::VectorXd Bineq(2);
  Bineq(0) = -1.;
  Bineq(1) = -1.;

  Eigen::VectorXd x(2);

  Eigen::VectorXd solution(2);
  solution(0) = 1.;
  solution(1) = 1.;

  double val = 1.;

  EiquadprogFast_status expected = EIQUADPROG_FAST_OPTIMAL;

  EiquadprogFast_status status = qp.solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x);

  BOOST_CHECK(status==expected);

  BOOST_CHECK_CLOSE(qp.getObjValue(),val,1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

// min ||x-x_0||^2, x_0 = (1 1)^T
//    s.t.
// x[1] = 5 - x[0]
// x[1] >= 3

BOOST_AUTO_TEST_CASE ( test_full )
{
  EiquadprogFast qp;
  qp.reset(2,1,1);

  Eigen::MatrixXd Q(2,2);
  Q.setZero();
  Q(0,0) = 1.0;
  Q(1,1) = 1.0;

  Eigen::VectorXd C(2);
  C(0) = -1.;
  C(1) = -1.;

  Eigen::MatrixXd Aeq(1,2);
  Aeq(0,0) = 1.;
  Aeq(0,1) = 1.;

  Eigen::VectorXd Beq(1);
  Beq(0) = -5.;

  Eigen::MatrixXd Aineq(1,2);
  Aineq.setZero();
  Aineq(0,1) = 1.;

  Eigen::VectorXd Bineq(1);
  Bineq(0) = -3.;

  Eigen::VectorXd x(2);

  Eigen::VectorXd solution(1);
  solution(0) = 2.;
  solution(1) = 3.;

  double val = 1.5;

  EiquadprogFast_status expected = EIQUADPROG_FAST_OPTIMAL;

  EiquadprogFast_status status = qp.solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x);

  BOOST_CHECK(status==expected);

  BOOST_CHECK_CLOSE(qp.getObjValue(),val,1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

BOOST_AUTO_TEST_SUITE_END ()

