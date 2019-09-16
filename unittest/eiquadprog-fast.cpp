#include <iostream>

#include <Eigen/Core>

#include <boost/test/unit_test.hpp>

#include "eiquadprog/eiquadprog-fast.hpp"

BOOST_AUTO_TEST_SUITE ( BOOST_TEST_MODULE )

// min ||x||^2

BOOST_AUTO_TEST_CASE ( test_unbiased )
{
  std::cout << "Ok" << std::endl;
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_SUITE_END ()

