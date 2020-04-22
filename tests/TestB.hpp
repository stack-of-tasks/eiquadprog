#ifndef TEST_EIQUADPROG_CLASS_B_
#define TEST_EIQUADPROG_CLASS_B_

#include "TestA.hpp"

namespace eiquadprog {
namespace tests {

class B {

 protected:
  Eigen::VectorXd solution_;
  
 public:
  A A_;
  
  B();
  bool do_something();
};
}
}
#endif /* TEST_EIQUADPROG_CLASS_B_ */
