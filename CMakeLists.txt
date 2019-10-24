#
# Copyright (c) 2019 CNRS
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

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/boost.cmake)
INCLUDE(cmake/eigen.cmake)
INCLUDE(cmake/test.cmake)

SET(PROJECT_NAMESPACE stack-of-tasks)
SET(PROJECT_NAME eiquadprog)
SET(PROJECT_DESCRIPTION "C++ reimplementation of eiquadprog")
SET(PROJECT_URL "https://gepgitlab.laas.fr/${PROJECT_NAMESPACE}/${PROJECT_NAME}")
SET(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")

# Disable -Werror on Unix for now.
SET(CXX_DISABLE_WERROR True)
SET(CMAKE_VERBOSE_MAKEFILE True)

SETUP_PROJECT()

# ----------------------------------------------------
# --- DEPENDENCIES -----------------------------------
# ----------------------------------------------------
ADD_REQUIRED_DEPENDENCY("eigen3")

SET(BOOST_COMPONENTS unit_test_framework)
SEARCH_FOR_BOOST()
INCLUDE_DIRECTORIES(SYSTEM ${Boost_INCLUDE_DIRS})

# --- PACKAGING ----------------------------------------------------------------
PKG_CONFIG_APPEND_LIBS(${PROJECT_NAME})

ADD_SUBDIRECTORY(include/${PROJECT_NAME})
ADD_SUBDIRECTORY(unittest)

SETUP_PROJECT_FINALIZE()
