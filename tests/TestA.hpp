#ifndef TEST_EIQUADPROG_CLASS_A_
#define TEST_EIQUADPROG_CLASS_A_

#include <Eigen/Core>
#include <eiquadprog/eiquadprog-fast.hpp>

namespace eiquadprog {
namespace tests {

class A {
 protected:
  eiquadprog::solvers::EiquadprogFast_status expected_;
  Eigen::MatrixXd Q_;
  Eigen::VectorXd C_;
  Eigen::MatrixXd Aeq_;
  Eigen::VectorXd Beq_;
  Eigen::MatrixXd Aineq_;
  Eigen::VectorXd Bineq_;

 public:
  eiquadprog::solvers::EiquadprogFast QP_;

  A();
  eiquadprog::solvers::EiquadprogFast_status solve(Eigen::VectorXd &x);
};

}  // namespace tests
} /* namespace eiquadprog */

#endif /* TEST_EIQUADPROG_CLASS_A_ */
