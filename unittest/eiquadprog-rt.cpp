#include <iostream>

#include <Eigen/Core>

#include <boost/test/unit_test.hpp>

#include "eiquadprog/eiquadprog-rt.hpp"

using namespace eiquadprog::solvers;

/**
 * solves the problem
 * min. 0.5 * x' Hess x + g0' x
 * s.t. CE x + ce0 = 0
 *      CI x + ci0 >= 0
 */

BOOST_AUTO_TEST_SUITE ( BOOST_TEST_MODULE )

// min ||x-x_0||^2, x_0 = (1 1)^T
//    s.t.
// x[1] = 5 - x[0]
// x[1] >= 3

BOOST_AUTO_TEST_CASE ( test_full )
{
  RtEiquadprog<2,1,1> qp;

  RtMatrixX<2,2>::d Q;
  Q.setZero();
  Q(0,0) = 1.0;
  Q(1,1) = 1.0;

  RtVectorX<2>::d C;
  C(0) = -1.;
  C(1) = -1.;

  RtMatrixX<1,2>::d Aeq;
  Aeq(0,0) = 1.;
  Aeq(0,1) = 1.;

  RtVectorX<1>::d Beq;
  Beq(0) = -5.;

  RtMatrixX<1,2>::d Aineq;
  Aineq.setZero();
  Aineq(0,1) = 1.;

  RtVectorX<1>::d Bineq;
  Bineq(0) = -3.;

  RtVectorX<2>::d x;

  RtVectorX<2>::d solution;
  solution(0) = 2.;
  solution(1) = 3.;

  double val = 1.5;

  RtEiquadprog_status expected = RT_EIQUADPROG_OPTIMAL;

  RtEiquadprog_status status = qp.solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x);

  BOOST_CHECK_EQUAL(status,expected);

  BOOST_CHECK_CLOSE(qp.getObjValue(),val,1e-6);

  BOOST_CHECK(x.isApprox(solution));
}

BOOST_AUTO_TEST_SUITE_END ()

