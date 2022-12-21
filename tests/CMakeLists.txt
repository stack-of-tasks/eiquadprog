#
# This file is part of eiquadprog.
#
# eiquadprog is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# eiquadprog is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with eiquadprog.  If not, see <https://www.gnu.org/licenses/>.
#

add_definitions(-DBOOST_TEST_DYN_LINK -DBOOST_TEST_MAIN)

# test paths and names (without .cpp extension)
set(TESTS eiquadprog-basic eiquadprog-fast eiquadprog-rt eiquadprog-both
          test-integration)

foreach(test ${TESTS})
  add_unit_test(${test} ${test}.cpp)
  target_link_libraries(${test} ${PROJECT_NAME} Boost::unit_test_framework)
  target_compile_options(
    ${test} PRIVATE "-Wno-sign-conversion") # We have a lot of implicit size_t
                                            # to Eigen::Index conversions
endforeach(test ${TESTS})

add_library(testab SHARED TestA.cpp TestB.cpp)
target_link_libraries(testab ${PROJECT_NAME})
target_link_libraries(test-integration testab)
target_compile_options(
  testab PRIVATE "-Wno-sign-conversion") # We have a lot of implicit size_t to
                                         # Eigen::Index conversions
