#ifndef EIQUADPROG_UTILS_HPP_
#define EIQUADPROG_UTILS_HPP_

#include <Eigen/Core>
#include <iostream>

namespace eiquadprog {
namespace utils {

/// Compute sqrt(a^2 + b^2)
template <typename Scalar>
inline Scalar distance(Scalar a, Scalar b) {
  Scalar a1, b1, t;
  a1 = std::abs(a);
  b1 = std::abs(b);
  if (a1 > b1) {
    t = (b1 / a1);
    return a1 * std::sqrt(1.0 + t * t);
  } else if (b1 > a1) {
    t = (a1 / b1);
    return b1 * std::sqrt(1.0 + t * t);
  }
  return a1 * std::sqrt(2.0);
}

template <class Derived>
void print_vector(const char *name, Eigen::MatrixBase<Derived> &x) {
  std::cerr << name << x.transpose() << std::endl;
}
template <class Derived>
void print_matrix(const char *name, Eigen::MatrixBase<Derived> &x) {
  std::cerr << name << std::endl << x << std::endl;
}

template <class Derived>
void print_vector(const char *name, Eigen::MatrixBase<Derived> &x, int /*n*/) {
  print_vector(name, x);
}
template <class Derived>
void print_matrix(const char *name, Eigen::MatrixBase<Derived> &x, int /*n*/) {
  print_matrix(name, x);
}

}  // namespace utils
}  // namespace eiquadprog

#endif
