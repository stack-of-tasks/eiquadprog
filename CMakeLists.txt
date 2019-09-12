#
# Copyright (c) 2019 CNRS
#
# This file is part of eiquadprog
# pinocchio is free software: you can redistribute it
# and/or modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
# pinocchio is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Lesser Public License for more details. You should have
# received a copy of the GNU Lesser General Public License along with
# pinocchio If not, see
# <http://www.gnu.org/licenses/>.

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/boost.cmake)

SET(PROJECT_NAME eiquadprog)
SET(PROJECT_DESCRIPTION "C++ reimplementation of eiquadprog")
SET(PROJECT_URL "http://github.com/stack-of-tasks/eiquadprog")

# Disable -Werror on Unix for now.
SET(CXX_DISABLE_WERROR True)
SET(CMAKE_VERBOSE_MAKEFILE True)

# ----------------------------------------------------
# --- DEPENDENCIES -----------------------------------
# ----------------------------------------------------
# ADD_REQUIRED_DEPENDENCY("pinocchio >= 2.1.0")

# ----------------------------------------------------
# --- INCLUDE ----------------------------------------
# ----------------------------------------------------
# Add headers here if any
SET(${PROJECT_NAME}_HEADERS
#    include/eiquadprog/fwd.hpp
  )

# ----------------------------------------------------
# --- EXECUTABLE -------------------------------------
# ----------------------------------------------------
SET(${PROJECT_NAME}_SOURCES
#    src/main.cpp
  )

# Create the main executable
#ADD_EXECUTABLE(main ${${PROJECT_NAME}_SOURCES})
# Link main to its dependencies 
#PKG_CONFIG_USE_DEPENDENCY(main pinocchio)

# --- PACKAGING ----------------------------------------------------------------
PKG_CONFIG_APPEND_LIBS(${PROJECT_NAME})
