#include <eiquadprog/eiquadprog.hpp>
namespace eiquadprog {
namespace solvers {

using namespace Eigen;

/* solve_quadprog is used for on-demand QP solving */

double solve_quadprog(MatrixXd &G, VectorXd &g0, const MatrixXd &CE,
                      const VectorXd &ce0, const MatrixXd &CI,
                      const VectorXd &ci0, VectorXd &x, VectorXi &activeSet,
                      size_t &activeSetSize) {
  Eigen::DenseIndex p = CE.cols();
  Eigen::DenseIndex m = CI.cols();

  VectorXd y(p + m);

  return solve_quadprog(G, g0, CE, ce0, CI, ci0, x, y, activeSet,
                        activeSetSize);
}

double solve_quadprog(MatrixXd &G, VectorXd &g0, const MatrixXd &CE,
                      const VectorXd &ce0, const MatrixXd &CI,
                      const VectorXd &ci0, VectorXd &x, VectorXd &y,
                      VectorXi &activeSet, size_t &activeSetSize) {
  LLT<MatrixXd, Lower> chol(G.cols());
  double c1;
  /* compute the trace of the original matrix G */
  c1 = G.trace();

  /* decompose the matrix G in the form LL^T */
  chol.compute(G);

  return solve_quadprog(chol, c1, g0, CE, ce0, CI, ci0, x, y, activeSet,
                        activeSetSize);
}

double solve_quadprog(LLT<MatrixXd, Lower> &chol, double c1, VectorXd &g0,
                      const MatrixXd &CE, const VectorXd &ce0,
                      const MatrixXd &CI, const VectorXd &ci0, VectorXd &x,
                      VectorXi &activeSet, size_t &activeSetSize) {
  Eigen::DenseIndex p = CE.cols();
  Eigen::DenseIndex m = CI.cols();

  VectorXd y(p + m);

  return solve_quadprog(chol, c1, g0, CE, ce0, CI, ci0, x, y, activeSet,
                        activeSetSize);
}

/* solve_quadprog2 is used for when the Cholesky decomposition of G is
 * pre-computed
 * @param A Output vector containing the indexes of the active constraints.
 * @param q Output value representing the size of the active set.
 */
double solve_quadprog(LLT<MatrixXd, Lower> &chol, double c1, VectorXd &g0,
                      const MatrixXd &CE, const VectorXd &ce0,
                      const MatrixXd &CI, const VectorXd &ci0, VectorXd &x,
                      VectorXd &u, VectorXi &A, size_t &q) {
  size_t i, k, l; /* indices */
  size_t ip, me, mi;
  size_t n = g0.size();
  size_t p = CE.cols();
  size_t m = CI.cols();
  MatrixXd R(g0.size(), g0.size()), J(g0.size(), g0.size());

  VectorXd s(m + p), z(n), r(m + p), d(n), np(n);
  VectorXd x_old(n), u_old(m + p);
  double f_value, psi, c2, sum, ss, R_norm;
  const double inf = std::numeric_limits<double>::infinity();
  double t, t1, t2; /* t is the step length, which is the minimum of the partial
                     * step length t1 and the full step length t2 */
  //        VectorXi A(m + p); // Del Prete: active set is now an output
  //        parameter
  if (static_cast<size_t>(A.size()) != m + p) A.resize(m + p);
  VectorXi A_old(m + p), iai(m + p), iaexcl(m + p);
  //        int q;
  size_t iq, iter = 0;

  me = p; /* number of equality constraints */
  mi = m; /* number of inequality constraints */
  q = 0;  /* size of the active set A (containing the indices of the active
             constraints) */

  /*
   * Preprocessing phase
   */

  /* initialize the matrix R */
  d.setZero();
  R.setZero();
  R_norm = 1.0; /* this variable will hold the norm of the matrix R */

  /* compute the inverse of the factorized matrix G^-1, this is the initial
   * value for H */
  // J = L^-T
  J.setIdentity();
  chol.matrixU().solveInPlace(J);
  c2 = J.trace();
#ifdef EIQGUADPROG_TRACE_SOLVER
  utils::print_matrix("J", J);
#endif

  /* c1 * c2 is an estimate for cond(G) */

  /*
   * Find the unconstrained minimizer of the quadratic form 0.5 * x G x + g0 x
   * this is a feasible point in the dual space
   * x = G^-1 * g0
   */
  x = -g0;
  chol.solveInPlace(x);
  /* and compute the current solution value */
  f_value = 0.5 * g0.dot(x);
#ifdef EIQGUADPROG_TRACE_SOLVER
  std::cerr << "Unconstrained solution: " << f_value << std::endl;
  utils::print_vector("x", x);
#endif

  /* Add equality constraints to the working set A */
  iq = 0;
  for (i = 0; i < me; i++) {
    np = CE.col(i);
    compute_d(d, J, np);
    update_z(z, J, d, iq);
    update_r(R, r, d, iq);
#ifdef EIQGUADPROG_TRACE_SOLVER
    utils::print_matrix("R", R);
    utils::print_vector("z", z);
    utils::print_vector("r", r);
    utils::print_vector("d", d);
#endif

    /* compute full step length t2: i.e., the minimum step in primal space s.t.
       the contraint becomes feasible */
    t2 = 0.0;
    if (std::abs(z.dot(z)) >
        std::numeric_limits<double>::epsilon())  // i.e. z != 0
      t2 = (-np.dot(x) - ce0(i)) / z.dot(np);

    x += t2 * z;

    /* set u = u+ */
    u(iq) = t2;
    u.head(iq) -= t2 * r.head(iq);

    /* compute the new solution value */
    f_value += 0.5 * (t2 * t2) * z.dot(np);
    A(i) = static_cast<VectorXi::Scalar>(-i - 1);

    if (!add_constraint(R, J, d, iq, R_norm)) {
      // FIXME: it should raise an error
      // Equality constraints are linearly dependent
      return f_value;
    }
  }

  /* set iai = K \ A */
  for (i = 0; i < mi; i++) iai(i) = static_cast<VectorXi::Scalar>(i);

l1:
  iter++;
#ifdef EIQGUADPROG_TRACE_SOLVER
  utils::print_vector("x", x);
#endif
  /* step 1: choose a violated constraint */
  for (i = me; i < iq; i++) {
    ip = A(i);
    iai(ip) = -1;
  }

  /* compute s(x) = ci^T * x + ci0 for all elements of K \ A */
  ss = 0.0;
  psi = 0.0; /* this value will contain the sum of all infeasibilities */
  ip = 0;    /* ip will be the index of the chosen violated constraint */
  for (i = 0; i < mi; i++) {
    iaexcl(i) = 1;
    sum = CI.col(i).dot(x) + ci0(i);
    s(i) = sum;
    psi += std::min(0.0, sum);
  }
#ifdef EIQGUADPROG_TRACE_SOLVER
  utils::print_vector("s", s);
#endif

  if (std::abs(psi) <= static_cast<double>(mi) *
                           std::numeric_limits<double>::epsilon() * c1 * c2 *
                           100.0) {
    /* numerically there are not infeasibilities anymore */
    q = iq;
    return f_value;
  }

  /* save old values for u, x and A */
  u_old.head(iq) = u.head(iq);
  A_old.head(iq) = A.head(iq);
  x_old = x;

l2: /* Step 2: check for feasibility and determine a new S-pair */
  for (i = 0; i < mi; i++) {
    if (s(i) < ss && iai(i) != -1 && iaexcl(i)) {
      ss = s(i);
      ip = i;
    }
  }
  if (ss >= 0.0) {
    q = iq;
    return f_value;
  }

  /* set np = n(ip) */
  np = CI.col(ip);
  /* set u = (u 0)^T */
  u(iq) = 0.0;
  /* add ip to the active set A */
  A(iq) = static_cast<VectorXi::Scalar>(ip);

#ifdef EIQGUADPROG_TRACE_SOLVER
  std::cerr << "Trying with constraint " << ip << std::endl;
  utils::print_vector("np", np);
#endif

l2a: /* Step 2a: determine step direction */
  /* compute z = H np: the step direction in the primal space (through J, see
   * the paper) */
  compute_d(d, J, np);
  update_z(z, J, d, iq);
  /* compute N* np (if q > 0): the negative of the step direction in the dual
   * space */
  update_r(R, r, d, iq);
#ifdef EIQGUADPROG_TRACE_SOLVER
  std::cerr << "Step direction z" << std::endl;
  utils::print_vector("z", z);
  utils::print_vector("r", r);
  utils::print_vector("u", u);
  utils::print_vector("d", d);
  utils::print_vector("A", A);
#endif

  /* Step 2b: compute step length */
  l = 0;
  /* Compute t1: partial step length (maximum step in dual space without
   * violating dual feasibility */
  t1 = inf; /* +inf */
  /* find the index l s.t. it reaches the minimum of u+(x) / r */
  for (k = me; k < iq; k++) {
    double tmp;
    if (r(k) > 0.0 && ((tmp = u(k) / r(k)) < t1)) {
      t1 = tmp;
      l = A(k);
    }
  }
  /* Compute t2: full step length (minimum step in primal space such that the
   * constraint ip becomes feasible */
  if (std::abs(z.dot(z)) >
      std::numeric_limits<double>::epsilon())  // i.e. z != 0
    t2 = -s(ip) / z.dot(np);
  else
    t2 = inf; /* +inf */

  /* the step is chosen as the minimum of t1 and t2 */
  t = std::min(t1, t2);
#ifdef EIQGUADPROG_TRACE_SOLVER
  std::cerr << "Step sizes: " << t << " (t1 = " << t1 << ", t2 = " << t2
            << ") ";
#endif

  /* Step 2c: determine new S-pair and take step: */

  /* case (i): no step in primal or dual space */
  if (t >= inf) {
    /* QPP is infeasible */
    // FIXME: unbounded to raise
    q = iq;
    return inf;
  }
  /* case (ii): step in dual space */
  if (t2 >= inf) {
    /* set u = u +  t * [-r 1) and drop constraint l from the active set A */
    u.head(iq) -= t * r.head(iq);
    u(iq) += t;
    iai(l) = static_cast<VectorXi::Scalar>(l);
    delete_constraint(R, J, A, u, p, iq, l);
#ifdef EIQGUADPROG_TRACE_SOLVER
    std::cerr << " in dual space: " << f_value << std::endl;
    utils::print_vector("x", x);
    utils::print_vector("z", z);
    utils::print_vector("A", A);
#endif
    goto l2a;
  }

  /* case (iii): step in primal and dual space */

  x += t * z;
  /* update the solution value */
  f_value += t * z.dot(np) * (0.5 * t + u(iq));

  u.head(iq) -= t * r.head(iq);
  u(iq) += t;
#ifdef EIQGUADPROG_TRACE_SOLVER
  std::cerr << " in both spaces: " << f_value << std::endl;
  utils::print_vector("x", x);
  utils::print_vector("u", u);
  utils::print_vector("r", r);
  utils::print_vector("A", A);
#endif

  if (t == t2) {
#ifdef EIQGUADPROG_TRACE_SOLVER
    std::cerr << "Full step has taken " << t << std::endl;
    utils::print_vector("x", x);
#endif
    /* full step has taken */
    /* add constraint ip to the active set*/
    if (!add_constraint(R, J, d, iq, R_norm)) {
      iaexcl(ip) = 0;
      delete_constraint(R, J, A, u, p, iq, ip);
#ifdef EIQGUADPROG_TRACE_SOLVER
      utils::print_matrix("R", R);
      utils::print_vector("A", A);
#endif
      for (i = 0; i < m; i++) iai(i) = static_cast<VectorXi::Scalar>(i);
      for (i = 0; i < iq; i++) {
        A(i) = A_old(i);
        iai(A(i)) = -1;
        u(i) = u_old(i);
      }
      x = x_old;
      goto l2; /* go to step 2 */
    } else
      iai(ip) = -1;
#ifdef EIQGUADPROG_TRACE_SOLVER
    utils::print_matrix("R", R);
    utils::print_vector("A", A);
#endif
    goto l1;
  }

  /* a patial step has taken */
#ifdef EIQGUADPROG_TRACE_SOLVER
  std::cerr << "Partial step has taken " << t << std::endl;
  utils::print_vector("x", x);
#endif
  /* drop constraint l */
  iai(l) = static_cast<VectorXi::Scalar>(l);
  delete_constraint(R, J, A, u, p, iq, l);
#ifdef EIQGUADPROG_TRACE_SOLVER
  utils::print_matrix("R", R);
  utils::print_vector("A", A);
#endif

  s(ip) = CI.col(ip).dot(x) + ci0(ip);

#ifdef EIQGUADPROG_TRACE_SOLVER
  utils::print_vector("s", s);
#endif
  goto l2a;
}

bool add_constraint(MatrixXd &R, MatrixXd &J, VectorXd &d, size_t &iq,
                    double &R_norm) {
  size_t n = J.rows();
#ifdef EIQGUADPROG_TRACE_SOLVER
  std::cerr << "Add constraint " << iq << '/';
#endif
  size_t j, k;
  double cc, ss, h, t1, t2, xny;

  /* we have to find the Givens rotation which will reduce the element
     d(j) to zero.
     if it is already zero we don't have to do anything, except of
     decreasing j */
  for (j = n - 1; j >= iq + 1; j--) {
    /* The Givens rotation is done with the matrix (cc cs, cs -cc).
       If cc is one, then element (j) of d is zero compared with element
       (j - 1). Hence we don't have to do anything.
       If cc is zero, then we just have to switch column (j) and column (j - 1)
       of J. Since we only switch columns in J, we have to be careful how we
       update d depending on the sign of gs.
       Otherwise we have to apply the Givens rotation to these columns.
       The i - 1 element of d has to be updated to h. */
    cc = d(j - 1);
    ss = d(j);
    h = utils::distance(cc, ss);
    if (h == 0.0) continue;
    d(j) = 0.0;
    ss = ss / h;
    cc = cc / h;
    if (cc < 0.0) {
      cc = -cc;
      ss = -ss;
      d(j - 1) = -h;
    } else
      d(j - 1) = h;
    xny = ss / (1.0 + cc);
    for (k = 0; k < n; k++) {
      t1 = J(k, j - 1);
      t2 = J(k, j);
      J(k, j - 1) = t1 * cc + t2 * ss;
      J(k, j) = xny * (t1 + J(k, j - 1)) - t2;
    }
  }
  /* update the number of constraints added*/
  iq++;
  /* To update R we have to put the iq components of the d vector
     into column iq - 1 of R
  */
  R.col(iq - 1).head(iq) = d.head(iq);
#ifdef EIQGUADPROG_TRACE_SOLVER
  std::cerr << iq << std::endl;
#endif

  if (std::abs(d(iq - 1)) <= std::numeric_limits<double>::epsilon() * R_norm)
    // problem degenerate
    return false;
  R_norm = std::max<double>(R_norm, std::abs(d(iq - 1)));
  return true;
}

void delete_constraint(MatrixXd &R, MatrixXd &J, VectorXi &A, VectorXd &u,
                       size_t p, size_t &iq, size_t l) {
  size_t n = R.rows();
#ifdef EIQGUADPROG_TRACE_SOLVER
  std::cerr << "Delete constraint " << l << ' ' << iq;
#endif
  size_t i, j, k, qq = 0;
  double cc, ss, h, xny, t1, t2;

  /* Find the index qq for active constraint l to be removed */
  for (i = p; i < iq; i++)
    if (static_cast<size_t>(A(i)) == l) {
      qq = i;
      break;
    }

  /* remove the constraint from the active set and the duals */
  for (i = qq; i < iq - 1; i++) {
    A(i) = A(i + 1);
    u(i) = u(i + 1);
    R.col(i) = R.col(i + 1);
  }

  A(iq - 1) = A(iq);
  u(iq - 1) = u(iq);
  A(iq) = 0;
  u(iq) = 0.0;
  for (j = 0; j < iq; j++) R(j, iq - 1) = 0.0;
  /* constraint has been fully removed */
  iq--;
#ifdef EIQGUADPROG_TRACE_SOLVER
  std::cerr << '/' << iq << std::endl;
#endif

  if (iq == 0) return;

  for (j = qq; j < iq; j++) {
    cc = R(j, j);
    ss = R(j + 1, j);
    h = utils::distance(cc, ss);
    if (h == 0.0) continue;
    cc = cc / h;
    ss = ss / h;
    R(j + 1, j) = 0.0;
    if (cc < 0.0) {
      R(j, j) = -h;
      cc = -cc;
      ss = -ss;
    } else
      R(j, j) = h;

    xny = ss / (1.0 + cc);
    for (k = j + 1; k < iq; k++) {
      t1 = R(j, k);
      t2 = R(j + 1, k);
      R(j, k) = t1 * cc + t2 * ss;
      R(j + 1, k) = xny * (t1 + R(j, k)) - t2;
    }
    for (k = 0; k < n; k++) {
      t1 = J(k, j);
      t2 = J(k, j + 1);
      J(k, j) = t1 * cc + t2 * ss;
      J(k, j + 1) = xny * (J(k, j) + t1) - t2;
    }
  }
}

}  // namespace solvers
}  // namespace eiquadprog
