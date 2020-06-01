#
# This file is part of eiquadprog.
#
# eiquadprog is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# eiquadprog is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with eiquadprog.  If not, see <https://www.gnu.org/licenses/>.
#

INCLUDE_DIRECTORIES(SYSTEM ${Boost_INCLUDE_DIRS})
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK -DBOOST_TEST_MAIN)

#test paths and names (without .cpp extension)
SET(TESTS
  eiquadprog-basic
  eiquadprog-fast
  eiquadprog-rt
  eiquadprog-both
  )

FOREACH(test ${TESTS})
  ADD_UNIT_TEST(${test} ${test}.cpp)
  TARGET_LINK_LIBRARIES(${test} ${PROJECT_NAME}
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
ENDFOREACH(test ${TESTS})

ADD_LIBRARY(testab SHARED TestA.cpp TestB.cpp)
TARGET_INCLUDE_DIRECTORIES(testab SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})
TARGET_LINK_LIBRARIES( testab ${PROJECT_NAME})

ADD_UNIT_TEST(test-integration test-integration.cpp )
TARGET_LINK_LIBRARIES(test-integration
  testab ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
