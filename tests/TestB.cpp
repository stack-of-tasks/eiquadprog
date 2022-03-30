#include "TestB.hpp"

#include <Eigen/Core>
#include <iostream>

using namespace eiquadprog::solvers;
namespace eiquadprog {
namespace tests {

B::B() : solution_(2) { solution_.setZero(); }

bool B::do_something() {
  eiquadprog::solvers::EiquadprogFast_status expected = EIQUADPROG_FAST_OPTIMAL;

  Eigen::VectorXd x(2);

  eiquadprog::solvers::EiquadprogFast_status status = A_.solve(x);

  bool rstatus = true;

  if (status != expected) {
    std::cerr << "Status not to true for A_" << expected << " " << status
              << std::endl;
    rstatus = false;
  }

  if (!x.isApprox(solution_)) {
    std::cerr << "x!=solution : " << x << "!=" << solution_ << std::endl;
    rstatus = false;
  }
  return rstatus;
}

}  // namespace tests
}  // namespace eiquadprog
