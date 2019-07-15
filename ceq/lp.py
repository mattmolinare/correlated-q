# -*- coding: utf-8 -*-

from cvxopt import matrix, solvers
import numpy as np

__all__ = ['minimax', 'ceq']


def set_options(tol):
    solvers.options['abstol'] = tol
    solvers.options['reltol'] = tol
    solvers.options['feastol'] = tol
    solvers.options['show_progress'] = False
    solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF', 'LPX_K_MSGLEV': 0}


def reset_options():
    solvers.options = {}


class options:

    def __init__(self, tol):
        self.tol = tol

    def __enter__(self):
        set_options(self.tol)

    def __exit__(self, type, value, traceback):
        reset_options()


def rationality_constraint_matrix(Qs):

    m = np.empty((5, 4, 5))
    for i in range(5):
        np.concatenate((Qs[:i], Qs[(i + 1):]), out=m[i])

    m -= Qs[:, np.newaxis]

    return m


def ceq(Qsa, Qsb):

    set_options(1e-4)

    # objective coeffecients
    c = np.zeros((26, 1))
    c[25, 0] = -1.0
    c = matrix(c)

    # inequality constraints...
    G = np.zeros((65, 26))
    h = np.zeros((65, 1))

    # ... p >= 0
    np.fill_diagonal(G[:25, :25], -1.0)

    # ... player A rationality constraint
    ma = rationality_constraint_matrix(Qsa)

    for i in range(5):

        r1 = 25 + 4 * i
        r2 = r1 + 4

        c1 = 5 * i
        c2 = c1 + 5

        G[r1:r2, c1:c2] = ma[i]

    # ... player B rationality constraint
    mb = rationality_constraint_matrix(Qsb.T)

    for i in range(5):

        r1 = 45 + 4 * i
        r2 = r1 + 4

        G[r1:r2, i:25:5] = mb[i]

    G = matrix(G)
    h = matrix(h)

    # equality constraints...
    A = np.zeros((2, 26))
    b = np.zeros((2, 1))

    # ... sum(p) == 1
    A[0, :25] = 1.0
    b[0, 0] = 1.0

    # ... sum(p * Qsa) + sum(p * Qsb) == v
    A[1, :25] = (Qsa + Qsb).flat
    A[1, 25] = -1.0

    A = matrix(A)
    b = matrix(b)

    sol = solvers.lp(c, G, h, A=A, b=b, solver='glpk')
    if sol['x'] is None:
        # try default solver
        sol = solvers.lp(c, G, h, A=A, b=b)
    success = sol['status'] == 'optimal'

    p = np.array(sol['x'])[:25].reshape((5, 5))

    Vsa = (p * Qsa).sum()
    Vsb = (p * Qsb).sum()

    reset_options()

    return Vsa, Vsb, success


def minimax_lp(Qs):

    set_options(1e-4)

    # objective coeffecients
    c = np.zeros((6, 1))
    c[5, 0] = -1.0
    c = matrix(c)

    # inequality constraints...
    G = np.zeros((10, 6))
    h = np.zeros((10, 1))

    # ... p >= 0
    np.fill_diagonal(G[:5, :5], -1.0)

    # ... sum(p * Qs.T) >= v
    np.negative(Qs.T, out=G[5:, :5])
    G[5:, 5] = 1.0

    G = matrix(G)
    h = matrix(h)

    # equality constraints...
    A = np.zeros((1, 6))
    b = np.zeros((1, 1))

    # ... sum(p) == 1
    A[0, :5] = 1.0
    b[0, 0] = 1.0

    A = matrix(A)
    b = matrix(b)

    sol = solvers.lp(c, G, h, A=A, b=b, solver='glpk')
    if sol['x'] is None:
        # try default solver
        sol = solvers.lp(c, G, h, A=A, b=b)
    success = sol['status'] == 'optimal'

    Vs = sol['x'][5]

    reset_options()

    return Vs, success


def minimax_op(Qs):

    from cvxopt.modeling import dot, op, variable

    solvers.options['show_progress'] = False

    v = variable()
    p = variable(5)

    c1 = p >= 0
    c2 = sum(p) == 1
    c3 = dot(matrix(Qs), p) >= v

    lp = op(-v, [c1, c2, c3])
    lp.solve()
    success = lp.status == 'optimal'

    Vs = v.value[0]

    return Vs, success


# alias
minimax = minimax_lp
