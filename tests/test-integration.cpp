#include <Eigen/Core>
#include <boost/test/unit_test.hpp>
#include <iostream>

#include "TestA.hpp"
#include "TestB.hpp"

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(test_class_A_and_class_B) {
  eiquadprog::tests::B aB;

  BOOST_CHECK(aB.do_something());
}

BOOST_AUTO_TEST_SUITE_END()
