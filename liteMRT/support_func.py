"""
Support Functions for Radiative Transfer Calculations

A collection of low-level, Numba-compiled utility functions used in the liteMRT
radiative transfer model. These functions provide essential mathematical and
numerical operations for angular integration, Legendre/Schmidt polynomial
evaluation, phase function computation, and Gauss quadrature.

"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def gauss_zeroes_weights(x1, x2, n):
    """
    Task:
        To compute 'n' Gauss nodes and weights within (x1, x2).
    In:
        x1, x2   d       interval
        n        i       number of nodes
    Out:
        x, w     d[n]    zeros and weights
    Tree:
        -
    Notes:
        Tested only for x1 < x2. To test run, e.g.,
        ng = 64
        x, w = gauszw(-1.0, 1.0, ng)
        and compare vs [1].
    Refs:
        1. https://pomax.github.io/bezierinfo/legendre-gauss.html
    """
    const_yeps = 3.0e-14
    x = np.zeros(n)
    w = np.zeros(n)
    m = int((n + 1) / 2)
    yxm = 0.5 * (x2 + x1)
    yxl = 0.5 * (x2 - x1)
    for i in range(m):
        yz = np.cos(np.pi * (i + 0.75) / (n + 0.5))
        while True:
            yp1 = 1.0
            yp2 = 0.0
            for j in range(n):
                yp3 = yp2
                yp2 = yp1
                yp1 = ((2.0 * j + 1.0) * yz * yp2 - j * yp3) / (j + 1)
            ypp = n * (yz * yp1 - yp2) / (yz * yz - 1.0)
            yz1 = yz
            yz = yz1 - yp1 / ypp
            if np.abs(yz - yz1) < const_yeps:
                break  # exit while loop
        x[i] = yxm - yz * yxl
        x[n - 1 - i] = yxm + yxl * yz
        w[i] = 2.0 * yxl / ((1.0 - yz * yz) * ypp * ypp)
        w[n - 1 - i] = w[i]
    return x, w


# ==============================================================================


@jit(nopython=True, cache=True)
def legendre_polynomial(x, kmax):
    """
    Task:
        To compute the Legendre polynomials, Pk(x), for all orders k=0:kmax and a
        single point 'x' within [-1:+1]
    In:
        x      f   abscissa
        kmax   i   maximum order, k = 0,1,2...kmax
    Out:
        pk    [kmax+1]   Legendre polynomials
    Tree:
        -
    Notes:
        The Bonnet recursion formula [1, 2]:

        (k+1)P{k+1}(x) = (2k+1)*P{k}(x) - k*P{k-1}(x),                      (1)

        where k = 0:K, P{0}(x) = 1.0, P{1}(x) = x.
        For fast summation over k, this index changes first.
    Refs:
        1. https://en.wikipedia.org/wiki/Legendre_polynomials
        2. http://mathworld.wolfram.com/LegendrePolynomial.html
        3. https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.legendre.html
        4. https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_legendre.html
    """
    nk = kmax + 1
    pk = np.zeros(nk)
    if kmax == 0:
        pk[0] = 1.0
    elif kmax == 1:
        pk[0] = 1.0
        pk[1] = x
    else:
        pk[0] = 1.0
        pk[1] = x
        for ik in range(2, nk):
            pk[ik] = (2.0 - 1.0 / ik) * x * pk[ik - 1] - (1.0 - 1.0 / ik) * pk[ik - 2]
    return pk


@jit(nopython=True, cache=True)
def get_phase_function(xk: np.array, thetas: np.array) -> np.array:
    """Return the phase function for xk at thetas

    Parameters
    ----------
    xk : np.array
        Expansion moments
    thetas : np.array
        Angles

    Returns
    -------
    p: np.array
        p at different thetas
    """
    nk = len(xk)
    # Thetas = np.linspace(-np.pi, 0, 100)
    nu = np.cos(thetas)
    p = np.zeros_like(nu)
    for inu, nui in enumerate(nu):
        pk = legendre_polynomial(nui, nk - 1)
        p[inu] = np.dot(xk, pk)
    return p


@jit(nopython=True, cache=True)
def schmidt_polynomial(m, x, kmax):
    """
    Task:
        To compute the Qkm(x) plynomials for all k = m:kmax & Fourier order
        m > 0. Qkm(x) = 0 is returned for all k < m.
    In:
        m      i   Fourier order (as in theory cos(m*phi)): m = 1,2,3....
        x      f   abscissa
        kmax   i   maximum order, k = 0,1,2...kmax
    Out:
        pk    [kmax+1]   polynomials
    Tree:
        -
    Notes:
        Think me: provide only non-zero polynomials, k>=m, on output.
        Definition:

            Qkm(x) = sqrt[(k-m)!/(k+m)!]*Pkm,                               (1)
            Pkm(x) = (1-x2)^(m/2)*(dPk(x)/dx)^m,                            (2)

        where Pk(x) are the Legendre polynomials. Note, unlike in [2] (-1)^m is
        omitted in Qkm(x). Refer to [1-4] for details.

        Qkm(x) for a few initial values of m > 0 and k for testing:
        m = 1:
            Q01 = 0.0                                                // k = 0
            Q11 = sqrt( 0.5*(1.0 - x2) )                             // k = 1
            Q21 = 3.0*x*sqrt( (1.0 - x2)/6.0 )                       // k = 2
            Q31 = (3.0/4.0)*(5.0*x2 - 1.0)*sqrt( (1.0 - x2)/3.0 )    // k = 3
        m = 2:
            Q02 = 0.0                                                // k = 0
            Q12 = 0.0                                                // k = 1
            Q22 = 3.0/(2.0*sqrt(6.0))*(1.0 - x2);	                 // k = 2
            Q32 = 15.0/sqrt(120.0)*x*(1.0 - x2);                     // k = 3
            Q42 = 15.0/(2.0*sqrt(360.0))*(7.0*x2 - 1.0)*(1.0 - x2)   // k = 4
        m = 3:
            Q03 = 0.0                                                // k = 0
            Q13 = 0.0                                                // k = 1
            Q23 = 0.0                                                // k = 2
            Q33 = 15.0/sqrt(720.0)*(1.0 - x2)*sqrt(1.0 - x2);        // k = 3
            Q43 = 105.0/sqrt(5040.0)*(1.0 - x2)*x*sqrt(1.0 - x2)     // k = 4

       Data for stress test: POLQKM.f90 (agrees with polqkm.cpp)
            k = 512 (in Fortran 513), m = 256
                       x        POLQKM.f90               def polqkm              |err|
            -1.00       0.000000000000000E+000  -0.0000000000000000e+00   0.0
            -0.50 (!)  -2.601822304856592E-002  -2.6018223048565915e-02   3.5e-18
             0.00       3.786666189291950E-002   3.7866661892919498e-02   0.0
             0.25       9.592316443679009E-003   9.5923164436790085e-03   0.0
             0.50 (!)  -2.601822304856592E-002  -2.6018223048565915e-02   3.5e-18
             0.75      -2.785756308806302E-002  -2.7857563088063021e-02   0.0
             1.00       0.000000000000000E+000   0.0000000000000000e+00   0.0
    Refs:
        1. Gelfand IM et al., 1963: Representations of the rotation and Lorentz
           groups and their applications. Oxford: Pergamon Press.
        2. Hovenier JW et al., 2004: Transfer of Polarized Light in Planetary
           Atmosphere. Basic Concepts and Practical Methods, Dordrecht: Kluwer
           Academic Publishers.
        3. http://mathworld.wolfram.com/AssociatedLegendrePolynomial.html
        4. http://www.mathworks.com/help/matlab/ref/legendre.html
    """
    #
    nk = kmax + 1
    qk = np.zeros(nk)
    #
    #   k=m: Qmm(x)=c0*[sqrt(1-x2)]^m
    c0 = 1.0
    for ik in range(2, 2 * m + 1, 2):
        c0 = c0 - c0 / ik
    qk[m] = np.sqrt(c0) * np.power(np.sqrt(1.0 - x * x), m)
    #
    # Q{k-1}m(x), Q{k-2}m(x) -> Qkm(x)
    m1 = m * m - 1.0
    m4 = m * m - 4.0
    for ik in range(m + 1, nk):
        c1 = 2.0 * ik - 1.0
        c2 = np.sqrt((ik + 1.0) * (ik - 3.0) - m4)
        c3 = 1.0 / np.sqrt((ik + 1.0) * (ik - 1.0) - m1)
        qk[ik] = (c1 * x * qk[ik - 1] - c2 * qk[ik - 2]) * c3
    return qk
