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

cmake_minimum_required(VERSION 3.10)

# Project properties
set(PROJECT_ORG stack-of-tasks)
set(PROJECT_NAME eiquadprog)
set(PROJECT_DESCRIPTION "C++ reimplementation of eiquadprog")
set(PROJECT_URL "https://github.com/${PROJECT_ORG}/${PROJECT_NAME}")

# Project options
option(TRACE_SOLVER "trace solver on stderr" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Project configuration
set(PROJECT_USE_CMAKE_EXPORT TRUE)
set(CXX_DISABLE_WERROR TRUE)
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  if(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    message(
      FATAL_ERROR
        "\nPlease run the following command first:\ngit submodule update --init\n"
    )
  else()
    message(STATUS "JRL cmakemodules not found. Let's fetch it.")
    include(FetchContent)
    FetchContent_Declare(
      "jrl-cmakemodules"
      GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git")
    FetchContent_MakeAvailable("jrl-cmakemodules")
    FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
  endif()
endif()

# JRL-cmakemodule setup
include("${JRL_CMAKE_MODULES}/base.cmake")
include("${JRL_CMAKE_MODULES}/boost.cmake")

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

add_library(${PROJECT_NAME} src/eiquadprog-fast.cpp src/eiquadprog.cpp)
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
