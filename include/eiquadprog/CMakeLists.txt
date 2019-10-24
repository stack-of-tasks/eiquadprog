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
