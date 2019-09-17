SET(${PROJECT_NAME}_HEADERS
    eiquadprog.hpp
    eiquadprog-fast.hpp
    eiquadprog-fast.hxx
  )

INSTALL(FILES
  ${${PROJECT_NAME}_HEADERS}
  DESTINATION include/${PROJECT_NAME}
  )
