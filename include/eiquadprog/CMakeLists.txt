SET(${PROJECT_NAME}_HEADERS
    eiquadprog.hpp
    eiquadprog-fast.hpp
    eiquadprog-fast.hxx
    eiquadprog-rt.hpp
    eiquadprog-rt.hxx
  )

INSTALL(FILES
  ${${PROJECT_NAME}_HEADERS}
  DESTINATION include/${PROJECT_NAME}
  )
