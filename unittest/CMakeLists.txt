ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK -DBOOST_TEST_MAIN)

#test paths and names (without .cpp extension)
SET (TESTS
  eiquadprog-basic
  eiquadprog-fast
  )

FOREACH(test ${TESTS})
  ADD_UNIT_TEST(${test} ${test})
  TARGET_LINK_LIBRARIES(${test} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
  PKG_CONFIG_USE_DEPENDENCY(${test} eigen3)
ENDFOREACH(test ${TESTS})
