#
# Copyright (c) 2019-2020 CNRS
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

cmake_minimum_required(VERSION 3.1)

# Project properties
set(PROJECT_ORG stack-of-tasks)
set(PROJECT_NAME eiquadprog)
set(PROJECT_DESCRIPTION "C++ reimplementation of eiquadprog")
set(PROJECT_URL "https://github.com/${PROJECT_ORG}/${PROJECT_NAME}")

# Project options
option(TRACE_SOLVER "trace solver on stderr" OFF)

# Project configuration
set(PROJECT_USE_CMAKE_EXPORT TRUE)
set(CXX_DISABLE_WERROR TRUE)
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")

# JRL-cmakemodule setup
include(cmake/base.cmake)
include(cmake/boost.cmake)

# Project definition
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

# Project dependencies
add_project_dependency(Eigen3 REQUIRED)
if(BUILD_TESTING)
  find_package(Boost REQUIRED COMPONENTS unit_test_framework)
endif(BUILD_TESTING)

# Main Library
set(${PROJECT_NAME}_HEADERS
    include/${PROJECT_NAME}/eiquadprog.hpp
    include/${PROJECT_NAME}/eiquadprog-fast.hpp
    include/${PROJECT_NAME}/eiquadprog-rt.hpp
    include/${PROJECT_NAME}/eiquadprog-rt.hxx
    include/${PROJECT_NAME}/eiquadprog-utils.hxx)

add_library(${PROJECT_NAME} SHARED src/eiquadprog-fast.cpp src/eiquadprog.cpp)
target_compile_options(${PROJECT_NAME} PRIVATE "-Wno-sign-conversion"
)# We have a lot of implicit size_t to Eigen::Index conversions

if(TRACE_SOLVER)
  target_compile_definitions(${PROJECT_NAME} PRIVATE EIQGUADPROG_TRACE_SOLVER)
endif(TRACE_SOLVER)

modernize_target_link_libraries(
  ${PROJECT_NAME}
  SCOPE
  PUBLIC
  TARGETS
  Eigen3::Eigen
  INCLUDE_DIRS
  ${EIGEN3_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME}
                           INTERFACE $<INSTALL_INTERFACE:include>)
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${TARGETS_EXPORT_NAME}
  DESTINATION lib)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif(BUILD_TESTING)
