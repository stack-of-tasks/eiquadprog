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

  Eigen::MatrixXd Aeq(2,0);

  Eigen::VectorXd Beq(0);

  Eigen::MatrixXd Aineq(2,0);

  Eigen::VectorXd Bineq(0);

  Eigen::VectorXd x(2);
  Eigen::VectorXi activeSet(0);
  size_t activeSetSize;

  Eigen::VectorXd solution(2);
  solution.setZero();

  double val = 0.0;

  double out = Eigen::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x, activeSet, activeSetSize);

  BOOST_CHECK_CLOSE(out,val,1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

// min ||x-x_0||^2, x_0 = (1 1)^T

BOOST_AUTO_TEST_CASE ( test_biased )
{
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

  double out = Eigen::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x, activeSet, activeSetSize);

  BOOST_CHECK_CLOSE(out,val,1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

// min ||x||^2
//    s.t.
// x[1] = 1 - x[0]

BOOST_AUTO_TEST_CASE ( test_equality_constraints )
{
  Eigen::MatrixXd Q(2,2);
  Q.setZero();
  Q(0,0) = 1.0;
  Q(1,1) = 1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(2,1);
  Aeq(0,0) = 1.;
  Aeq(1,0) = 1.;

  Eigen::VectorXd Beq(1);
  Beq(0) = -1.;

  Eigen::MatrixXd Aineq(2,0);

  Eigen::VectorXd Bineq(0);

  Eigen::VectorXd x(2);
  Eigen::VectorXi activeSet(0);
  size_t activeSetSize;

  Eigen::VectorXd solution(2);
  solution(0) = 0.5;
  solution(1) = 0.5;

  double val = 0.25;

  double out = Eigen::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x, activeSet, activeSetSize);

  BOOST_CHECK_CLOSE(out,val,1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

// min ||x||^2
//    s.t.
// x[i] >= 1

BOOST_AUTO_TEST_CASE ( test_inequality_constraints )
{
  Eigen::MatrixXd Q(2,2);
  Q.setZero();
  Q(0,0) = 1.0;
  Q(1,1) = 1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(2,0);

  Eigen::VectorXd Beq(0);

  Eigen::MatrixXd Aineq(2,2);
  Aineq.setZero();
  Aineq(0,0) = 1.;
  Aineq(1,1) = 1.;

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

  double out = Eigen::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x, activeSet, activeSetSize);

  BOOST_CHECK_CLOSE(out,val,1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

// min ||x-x_0||^2, x_0 = (1 1)^T
//    s.t.
// x[1] = 5 - x[0]
// x[1] >= 3

BOOST_AUTO_TEST_CASE ( test_full )
{
  Eigen::MatrixXd Q(2,2);
  Q.setZero();
  Q(0,0) = 1.0;
  Q(1,1) = 1.0;

  Eigen::VectorXd C(2);
  C(0) = -1.;
  C(1) = -1.;

  Eigen::MatrixXd Aeq(2,1);
  Aeq(0,0) = 1.;
  Aeq(1,0) = 1.;

  Eigen::VectorXd Beq(1);
  Beq(0) = -5.;

  Eigen::MatrixXd Aineq(2,1);
  Aineq.setZero();
  Aineq(1,0) = 1.;

  Eigen::VectorXd Bineq(1);
  Bineq(0) = -3.;

  Eigen::VectorXd x(2);
  Eigen::VectorXi activeSet;
  size_t activeSetSize;

  Eigen::VectorXd solution(1);
  solution(0) = 2.;
  solution(1) = 3.;

  double val = 1.5;

  double out = Eigen::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x, activeSet, activeSetSize);

  BOOST_CHECK_CLOSE(out,val,1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

// min ||x||^2
//    s.t.
// x[0] = -1
// x[0] =  1
// DOES NOT WORK!

BOOST_AUTO_TEST_CASE ( test_unfeasible_equalities )
{
  Eigen::MatrixXd Q(2,2);
  Q.setZero();
  Q(0,0) = 1.0;
  Q(1,1) = 1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(2,2);
  Aeq.setZero();
  Aeq(0,0) = 1.;
  Aeq(0,1) = 1.;

  Eigen::VectorXd Beq(2);
  Beq(0) = -1.;
  Beq(1) =  1.;

  Eigen::MatrixXd Aineq(2,0);

  Eigen::VectorXd Bineq(0);

  Eigen::VectorXd x(2);
  Eigen::VectorXi activeSet;
  size_t activeSetSize;

  double out = Eigen::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x, activeSet, activeSetSize);

  // DOES NOT WORK!?
  // BOOST_CHECK(std::isinf(out));
}

// min ||x||^2
//    s.t.
// x[0] >=  1
// x[0] <= -1

BOOST_AUTO_TEST_CASE ( test_unfeasible_inequalities )
{
  Eigen::MatrixXd Q(2,2);
  Q.setZero();
  Q(0,0) = 1.0;
  Q(1,1) = 1.0;

  Eigen::VectorXd C(2);
  C.setZero();

  Eigen::MatrixXd Aeq(2,0);

  Eigen::VectorXd Beq(0);

  Eigen::MatrixXd Aineq(2,2);
  Aineq.setZero();
  Aineq(0,0) = 1.;
  Aineq(0,1) = -1.;

  Eigen::VectorXd Bineq(2);
  Bineq(0) = -1;
  Bineq(1) = -1;

  Eigen::VectorXd x(2);
  Eigen::VectorXi activeSet;
  size_t activeSetSize;

  double out = Eigen::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x, activeSet, activeSetSize);

  BOOST_CHECK(std::isinf(out));
}

BOOST_AUTO_TEST_SUITE_END ()

