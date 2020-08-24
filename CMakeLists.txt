#
# Copyright (c) 2019-2020 CNRS
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

CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

# Project properties
SET(PROJECT_ORG stack-of-tasks)
SET(PROJECT_NAME eiquadprog)
SET(PROJECT_DESCRIPTION "C++ reimplementation of eiquadprog")
SET(PROJECT_URL "https://github.com/${PROJECT_ORG}/${PROJECT_NAME}")

# Project options
OPTION(TRACE_SOLVER "trace solver on stderr" OFF)

# Project configuration
SET(PROJECT_USE_CMAKE_EXPORT TRUE)
SET(CXX_DISABLE_WERROR TRUE)
SET(CMAKE_VERBOSE_MAKEFILE TRUE)
SET(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")

# JRL-cmakemodule setup
INCLUDE(cmake/base.cmake)
INCLUDE(cmake/boost.cmake)

# Project definition
COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
PROJECT(${PROJECT_NAME} ${PROJECT_ARGS})

# Project dependencies
ADD_PROJECT_DEPENDENCY(Eigen3 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.2.92")
FIND_PACKAGE(Boost REQUIRED COMPONENTS unit_test_framework)

# Main Library
SET(${PROJECT_NAME}_HEADERS
  include/${PROJECT_NAME}/eiquadprog.hpp
  include/${PROJECT_NAME}/eiquadprog-fast.hpp
  include/${PROJECT_NAME}/eiquadprog-rt.hpp
  include/${PROJECT_NAME}/eiquadprog-rt.hxx
  include/${PROJECT_NAME}/eiquadprog-utils.hxx
  )

ADD_LIBRARY(${PROJECT_NAME} SHARED
  src/eiquadprog-fast.cpp
  src/eiquadprog.cpp
  )

IF(TRACE_SOLVER)
  TARGET_COMPILE_DEFINITIONS(${PROJECT_NAME} PRIVATE TRACE_SOLVER)
ENDIF(TRACE_SOLVER)

TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} INTERFACE $<INSTALL_INTERFACE:include>)
INSTALL(TARGETS ${PROJECT_NAME} EXPORT ${TARGETS_EXPORT_NAME} DESTINATION lib)

ADD_SUBDIRECTORY(tests)
