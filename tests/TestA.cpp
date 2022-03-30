#include "TestA.hpp"

#include <Eigen/Core>
#include <iostream>

using namespace eiquadprog::solvers;
using namespace eiquadprog::tests;

A::A() : Q_(2, 2), C_(2), Aeq_(0, 2), Beq_(0), Aineq_(0, 2), Bineq_(0), QP_() {
  QP_.reset(2, 0, 0);

  Q_.setZero();
  Q_(0, 0) = 1.0;
  Q_(1, 1) = 1.0;

  C_.setZero();

  expected_ = EIQUADPROG_FAST_OPTIMAL;
}

EiquadprogFast_status A::solve(Eigen::VectorXd &x) {
  return QP_.solve_quadprog(Q_, C_, Aeq_, Beq_, Aineq_, Bineq_, x);
}
