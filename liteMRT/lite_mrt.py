"""
liteMRT: Lightweight Multi-Resolution Radiative Transfer for Atmospheric and
Surface Reflectance

A high-performance, Numba-accelerated radiative transfer solver for computing
top-of-atmosphere (TOA) and bottom-of-atmosphere (BOA) radiances in a
plane-parallel atmosphere with non-homogeneous scattering and bidirectional
surface reflectance (BRDF). This module efficiently handles multiple scattering,
anisotropic phase functions, and complex BRDF models using Fourier-mode
decomposition and Gauss-Seidel iterative methods.
"""

import numpy as np
from numba import jit
from .support_func import gauss_zeroes_weights, legendre_polynomial, schmidt_polynomial
from .brdf import rho, expand_brdf_mu0_mu, expand_brdf_mu0_mup, expand_brdf_mup_mu, expand_brdf_mup_mup

TINY = 1e-8


@jit(nopython=True, cache=True)
def single_scattering_up(muup: float, mu0: float, azr: np.ndarray, 
    dtau: float, nlr: int, xk: np.ndarray, srfa_azrs: np.ndarray) -> np.ndarray:
    """
    Calculate the single-scattering upward intensity for a non-homogeneous BRDF
    using Gaussian quadrature.

    Parameters
    ----------
    muup : float
        Cosine of the Viewing Zenith Angle (VZA), where cos(VZA) < 0.
    mu0 : float
        Cosine of the Solar Zenith Angle (SZA), where cos(SZA) > 0.
    azr : ndarray of shape (naz,)
        Azimuthal angles in radians.
    dtau : float
        Optical thickness of each layer (integration step over tau).
    nlr : int
        Number of layers in the atmosphere.
    xk : ndarray of shape (nlr, nk)
        Expansion moments of the scattering phase function, scaled by the
        single-scattering albedo (ssa) and a factor of 2.
    srfa_azrs : ndarray of shape (naz,)
        BRDF surface reflectance from mu0 to muup for different azimuthal
        directions.

    Returns
    -------
    intensity_1up : ndarray of shape (naz,)
        Single-scattering upward intensity at the Top of Atmosphere (TOA) for
        each azimuthal angle.

    Notes
    -----
    This function implements the single-scattering upward intensity calculation
    by iterating over atmospheric layers, using Gaussian quadrature for
    azimuthal angles.
    intensity_t handles different azimuthal directions and computes the intensity by
    integrating over the optical depth. intensity_f the minimum value of `srfa` exceeds
    a small threshold,
    it assumes non-zero reflectance.

    - `legendre_polynomial()` calculates the Legendre polynomial for a given
    order.
    - The computation includes a special case for `mu == mu0` to avoid
    numerical instability.
    - Assumes a Lambertian surface if the surface albedo `srfa` has
    non-negligible values.

    References
    ----------
    - Additional references on single-scattering calculations and
    non-homogeneous BRDF models may be added here.
    """
    nb, nk = nlr + 1, xk.shape[1]
    smu, smu0 = np.sqrt(1.0 - muup * muup), np.sqrt(1.0 - mu0 * mu0)
    nu = muup * mu0 + smu * smu0 * np.cos(azr)
    naz, tau0 = len(azr), nlr * dtau
    tau = np.linspace(0.0, tau0, nb)
    p = np.zeros((nlr, naz))
    for inu, nui in enumerate(nu):
        pk = legendre_polynomial(nui, nk - 1)
        for ilr in range(nlr):
            p[ilr, inu] = np.dot(xk[ilr], pk)
    intensity_11up = np.zeros((nlr, naz))
    for ilr in range(nlr):
        intensity_11up[ilr, :] = (p[ilr, 0:naz] * mu0 / (mu0 - muup) * 
                                  (1.0 - np.exp(dtau / muup - dtau / mu0)))
    intensity_1up = np.zeros((nb, naz))
    for ib in range(nb - 2, -1, -1):
        intensity_1up[ib, :] = (intensity_1up[ib + 1, :] * np.exp(dtau / muup) + 
                               intensity_11up[ib] * np.exp(-tau[ib] / mu0))
    return (intensity_1up[0, :] + 2.0 * srfa_azrs * mu0 * 
            np.exp(-nlr * dtau / mu0) * np.exp(nlr * dtau / muup))


@jit(nopython=True, cache=True)
def single_scattering_down(mudn: float, mu0: float, azr: np.ndarray, 
    dtau: float, nlr: int, xk: np.ndarray) -> np.ndarray:
    """
    Calculate the single-scattering downward intensity for a non-homogeneous
    BRDF using Gaussian quadrature.

    Parameters
    ----------
    mudn : float
        Cosine of the Viewing Zenith Angle (VZA), where cos(VZA) > 0.
    mu0 : float
        Cosine of the Solar Zenith Angle (SZA), where cos(SZA) > 0.
    azr : ndarray of shape (naz,)
        Azimuthal angles in radians.
    dtau : float
        Optical thickness of each atmospheric layer (integration step over
        tau).
    nlr : int
        Number of layers in the atmosphere.
    xk : ndarray of shape (nlr, nk)
        Expansion moments of the scattering phase function, scaled by the
        single-scattering albedo (ssa) and a factor of 2.

    Returns
    -------
    intensity_1dn : ndarray of shape (naz,)
        Single-scattering downward intensity at the bottom-most layer for each
        azimuthal angle.

    Notes
    -----
    This function calculates the single-scattering downward intensity by
    iterating over atmospheric layers using Gaussian quadrature for azimuthal
    angles.
    intensity_t considers both cases where `mu == mu0` and `mu != mu0` to handle
    numerical stability.

    - `legendre_polynomial()` computes the Legendre polynomial for the
    specified order.
    - This method uses a series of integrals to accumulate the downward
    intensity at each layer.

    References
    ----------
    - Additional references or literature on single-scattering radiative
    transfer and BRDF models can be added here.
    """
    nb, nk = nlr + 1, xk.shape[1]
    smu, smu0 = np.sqrt(1.0 - mudn * mudn), np.sqrt(1.0 - mu0 * mu0)
    nu = mudn * mu0 + smu * smu0 * np.cos(azr)
    naz, tau0 = len(azr), nlr * dtau
    tau = np.linspace(0.0, tau0, nb)
    p = np.zeros((nlr, naz))
    for inu, nui in enumerate(nu):
        pk = legendre_polynomial(nui, nk - 1)
        for ilr in range(nlr):
            p[ilr, inu] = np.dot(xk[ilr], pk)
    intensity_11dn = np.zeros((nlr, naz))
    for ilr in range(nlr):
        for iaz in range(naz):
            if np.abs(mu0 - mudn) < TINY:
                intensity_11dn[ilr, iaz] = p[ilr, iaz] * dtau * np.exp(-dtau / mu0) / mu0
            else:
                intensity_11dn[ilr, iaz] = (p[ilr, iaz] * mu0 / (mu0 - mudn) * 
                                           (np.exp(-dtau / mu0) - np.exp(-dtau / mudn)))
    intensity_1dn = np.zeros((nb, naz))
    intensity_1dn[1, :] = intensity_11dn[0]
    for ib in range(2, nb):
        intensity_1dn[ib, :] = (intensity_1dn[ib - 1, :] * np.exp(-dtau / mudn) + 
                               intensity_11dn[ib - 1] * np.exp(-tau[ib - 1] / mu0))
    return intensity_1dn[-1, :]


@jit(nopython=True, cache=True)
def gauss_seidel_iterations_m(m: int, mu0: float, srfa_mu0_mup: np.ndarray, 
    srfa_mup_mup: np.ndarray, nit: int, ng1: int, nlr: int, 
    dtau: float, xk: np.ndarray, tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the Radiative Transfer Equation (RTE) using Gauss-Seidel (GS)
    iterations for a non-homogeneous BRDF.

    Parameters
    ----------
    m : int
        Fourier moment.
    mu0 : float
        Cosine of the Solar Zenith Angle (SZA), where cos(SZA) > 0.
    srfa_mu0_mup : ndarray of shape (ng1,)
        BRDF surface reflectance, from `mu0` to `mup`.
    srfa_mup_mup : ndarray of shape (ng1, ng1)
        BRDF surface reflectance, from `mup` to `mup`.
    nit : int
        Maximum number of Gauss-Seidel iterations, where nit > 0.
    ng1 : int
        Number of Gauss nodes per hemisphere.
    nlr : int
        Number of layers (or elements) for `dtau`, with total optical
        thickness tau0 = dtau * nlr.
    dtau : float
        Thickness of each layer element (integration step over tau).
    xk : ndarray of shape (nlr, nk)
        Expansion moments of the scattering phase function, scaled by the
        single-scattering albedo (ssa) and a factor of 2.

    Returns
    -------
    mug : ndarray of shape (ng1*2,)
        Gauss nodes for both hemispheres.
    wg : ndarray of shape (ng1*2,)
        Gauss weights for both hemispheres.
    intensity_up : ndarray of shape (nlr+1, ng1)
        Upward intensity at each layer and Gauss node.
    intensity_dn : ndarray of shape (nlr+1, ng1)
        Downward intensity at each layer and Gauss node.

    Notes
    -----
    - The Top-Of-Atmosphere (TOA) scaling factor is 2π.
    - For xk with the shape of (nlr, nk), we can only take the (nlr, ng1) part
    into consideration.
    """
    nb, nk, ng2 = nlr + 1, xk.shape[1], ng1 * 2
    tau0, tau = nlr * dtau, np.linspace(0.0, nlr * dtau, nb)
    mup, w = gauss_zeroes_weights(0.0, 1.0, ng1)
    mug, wg = np.concatenate((-mup, mup)), np.concatenate((w, w))

    nk = min(nk, ng2)
    xk = np.ascontiguousarray(xk[:, :nk])
    pk, p = np.zeros((ng2, nk)), np.zeros((nlr, ng2))
    if m == 0:
        pk0 = legendre_polynomial(mu0, nk - 1)
        for ig in range(ng2):
            pk[ig, :] = legendre_polynomial(mug[ig], nk - 1)
            for ilr in range(nlr):
                p[ilr, ig] = np.dot(xk[ilr], pk[ig, :] * pk0)
    else:
        pk0 = schmidt_polynomial(m, mu0, nk - 1)
        for ig in range(ng2):
            pk[ig, :] = schmidt_polynomial(m, mug[ig], nk - 1)
            for ilr in range(nlr):
                p[ilr, ig] = np.dot(xk[ilr], pk[ig, :] * pk0)
    intensity_11dn = np.zeros((nlr, ng1))
    for ilr in range(nlr):
        for ig in range(ng1):
            mu = mup[ig]
            if np.abs(mu0 - mu) < TINY:
                intensity_11dn[ilr, ig] = p[ilr, ng1 + ig] * dtau * np.exp(-dtau / mu0) / mu0
            else:
                intensity_11dn[ilr, ig] = (p[ilr, ng1 + ig] * mu0 / (mu0 - mu) * 
                                          (np.exp(-dtau / mu0) - np.exp(-dtau / mu)))
    intensity_1dn = np.zeros((nb, ng1))
    intensity_1dn[1, :] = intensity_11dn[0]
    for ib in range(2, nb):
        intensity_1dn[ib, :] = (intensity_1dn[ib - 1, :] * np.exp(-dtau / mup) + 
                               intensity_11dn[ib - 1] * np.exp(-tau[ib - 1] / mu0))
    intensity_11up = np.zeros((nlr, ng1))
    for ilr in range(nlr):
        intensity_11up[ilr, :] = (p[ilr, 0:ng1] * mu0 / (mu0 + mup) * 
                                  (1.0 - np.exp(-dtau / mup - dtau / mu0)))
    intensity_1up = np.zeros_like(intensity_1dn)
    intensity_1up[nb - 1, :] = 2.0 * srfa_mu0_mup * mu0 * np.exp(-tau0 / mu0)
    intensity_1up[nb - 2, :] = (intensity_1up[nb - 1, :] * np.exp(-dtau / mup) + 
                                intensity_11up[-1] * np.exp(-tau[nb - 2] / mu0))
    for ib in range(nb - 3, -1, -1):
        intensity_1up[ib, :] = (intensity_1up[ib + 1, :] * np.exp(-dtau / mup) + 
                                intensity_11up[ib] * np.exp(-tau[ib] / mu0))
    wpij = np.zeros((nlr, ng2, ng2))
    for ilr in range(nlr):
        for ig in range(ng2):
            for jg in range(ng2):
                wpij[ilr, ig, jg] = wg[jg] * np.dot(xk[ilr], pk[ig, :] * pk[jg, :])
    transmittance_matrix = wpij[:, 0:ng1, 0:ng1].copy()
    reflectance_matrix = wpij[:, 0:ng1, ng1:ng2].copy()
    intensity_up, intensity_dn = np.copy(intensity_1up), np.copy(intensity_1dn)
    tol, conv_check = 1e-4, 0
    prev_intensity_up, prev_intensity_dn = np.copy(intensity_up), np.copy(intensity_dn)

    for _ in range(nit):
        #       Down:
        # intensity_up05 = 0.5 * (intensity_up[0, :] + intensity_up[1, :])
        # intensity_dn05 = 0.5 * (
        #     intensity_dn[0, :] + intensity_dn[1, :]
        # )  # intensity_dn[0, :] = 0.0
        # scattering_integral = np.dot(
        #     reflectance_matrix[0], intensity_up05
        # ) + np.dot(transmittance_matrix[0], intensity_dn05)
        # intensity_dn[1, :] = (
        #     intensity_11dn[0]
        #     + (1.0 - np.exp(-dtau / mup)) * scattering_integral
        # )
        for ib in range(1, nb):
            intensity_up05 = 0.5 * (intensity_up[ib - 1, :] + intensity_up[ib, :])
            intensity_dn05 = 0.5 * (intensity_dn[ib - 1, :] + intensity_dn[ib, :])
            scattering_integral = (np.dot(reflectance_matrix[ib - 1], intensity_up05) + 
                                   np.dot(transmittance_matrix[ib - 1], intensity_dn05))
            intensity_dn[ib, :] = (intensity_dn[ib - 1, :] * np.exp(-dtau / mup) + 
                                   intensity_11dn[ib - 1] * np.exp(-tau[ib - 1] / mu0) + 
                                   (1.0 - np.exp(-dtau / mup)) * scattering_integral)

        for ig in range(ng1):
            intensity_up[nb - 1, ig] = 2.0 * np.dot(intensity_dn[nb - 1, :] *\
                srfa_mup_mup[ig], mup * w) + \
                2.0 * srfa_mu0_mup[ig] * mu0 * np.exp(-tau0 / mu0)
        # intensity_up05 = 0.5 * (
        #     intensity_up[nb - 2, :] + intensity_up[nb - 1, :]
        # )  # intensity_up[nb-1, :] = 0.0
        # intensity_dn05 = 0.5 * (
        #     intensity_dn[nb - 2, :] + intensity_dn[nb - 1, :]
        # )
        # scattering_integral = np.dot(
        #     transmittance_matrix[-1], intensity_up05
        # ) + np.dot(reflectance_matrix[-1], intensity_dn05)
        # intensity_up[nb - 2, :] = (
        #     intensity_up[nb - 1, :] * np.exp(-dtau / mup)
        #     + intensity_11up[nb - 2] * np.exp(-tau[nb - 2] / mu0)
        #     + (1.0 - np.exp(-dtau / mup)) * scattering_integral
        # )
        for ib in range(nb - 2, -1, -1):
            intensity_up05 = 0.5 * (intensity_up[ib, :] + intensity_up[ib + 1, :])
            intensity_dn05 = 0.5 * (intensity_dn[ib, :] + intensity_dn[ib + 1, :])
            scattering_integral = (np.dot(transmittance_matrix[ib], intensity_up05) + 
                                   np.dot(reflectance_matrix[ib], intensity_dn05))
            intensity_up[ib, :] = (intensity_up[ib + 1, :] * np.exp(-dtau / mup) + 
                                   intensity_11up[ib] * np.exp(-tau[ib] / mu0) + 
                                   (1.0 - np.exp(-dtau / mup)) * scattering_integral)

        if (np.allclose(intensity_up, prev_intensity_up, rtol=tol, atol=tol) and 
            np.allclose(intensity_dn, prev_intensity_dn, rtol=tol, atol=tol)):
            conv_check += 1
        else:
            conv_check = 0
        prev_intensity_up, prev_intensity_dn = np.copy(intensity_up), np.copy(intensity_dn)
        if conv_check >= 2:
            break

    if conv_check < 2:
        print(f"Gauss-Seidel iterations did not converge within {nit} iterations. "
              f"Final results returned with residual error.", UserWarning)
    return mug, wg, intensity_up[:, :], intensity_dn[:, :]


@jit(nopython=True, cache=True)
def source_function_integrate_down(m: int, mudn: float, mu0: float, nlr: int,
    dtau: float, xk: np.ndarray, mug: np.ndarray, wg: np.ndarray,
    intensity_g05: np.ndarray) -> float:
    """
    Perform source function integration for downward radiative transfer in a
    non-homogeneous BRDF.

    Parameters
    ----------
    m : int
        Fourier moment, where m = 0, 1, 2, ...
    mudn : float
        Cosine of the viewing zenith angle (VZA) in the downward line-of-sight
        (LOS), where cos(VZA) > 0.
    mu0 : float
        Cosine of the Solar Zenith Angle (SZA), where cos(SZA) > 0.
    nlr : int
        Number of layers in the atmosphere, with the total optical thickness
        tau0 = dtau * nlr.
    dtau : float
        Thickness of each layer element, used as the integration step over tau.
    xk : ndarray of shape (nlr, nk)
        Expansion moments of the scattering phase function, scaled by
        single-scattering albedo (ssa) and a factor of 2. The term (2k+1) is
        included, where `nk = len(xk)`.
    mug : ndarray of shape (ng2,)
        Gauss nodes used in quadrature for integration.
    wg : ndarray of shape (ng2,)
        Gauss weights corresponding to `mug`.
    intensity_g05 : ndarray of shape (nlr, ng2)
        Radiative transfer equation (RTE) solution at Gauss nodes, evaluated
        at the midpoint of each layer `dtau`.

    Returns
    -------
    intensity_boa : float
        Top-of-atmosphere (BOA) radiance, `intensity_toa`, as a function of `mudn`.

    Notes
    -----
    - This function performs source function integration for the downward
    radiative transfer direction using Gauss quadrature over each layer.
    - A scaling factor of 2π is applied at the BOA.
    """
    ng2, nk, nb = len(wg), xk.shape[1], nlr + 1
    tau0, tau = nlr * dtau, np.linspace(0.0, nlr * dtau, nb)
    pk = np.zeros((ng2, nk))
    if m == 0:
        pk0 = legendre_polynomial(mu0, nk - 1)
        pku = legendre_polynomial(mudn, nk - 1)
        for ig in range(ng2):
            pk[ig, :] = legendre_polynomial(mug[ig], nk - 1)
    else:
        pk0 = schmidt_polynomial(m, mu0, nk - 1)
        pku = schmidt_polynomial(m, mudn, nk - 1)
        for ig in range(ng2):
            pk[ig, :] = schmidt_polynomial(m, mug[ig], nk - 1)

    intensity_11dn = np.zeros(nlr)
    for ilr in range(nlr):
        p = np.dot(xk[ilr], pku * pk0)
        if np.abs(mudn - mu0) < TINY:
            intensity_11dn[ilr] = p * dtau * np.exp(-dtau / mu0) / mu0
        else:
            intensity_11dn[ilr] = (p * mu0 / (mu0 - mudn) * 
                                   (np.exp(-dtau / mu0) - np.exp(-dtau / mudn)))
    intensity_1dn = np.zeros(nb)
    for ib in range(1, nb):
        intensity_1dn[ib] = (intensity_1dn[ib - 1] * np.exp(-dtau / mudn) + 
                             intensity_11dn[ib - 1] * np.exp(-tau[ib - 1] / mu0))
    wpij = np.zeros((nlr, ng2))
    for ilr in range(nlr):
        for jg in range(ng2):
            wpij[ilr, jg] = wg[jg] * np.dot(xk[ilr], pku[:] * pk[jg, :])
    intensity_dn = np.copy(intensity_1dn)
    for ib in range(1, nb):
        scattering_integral = np.dot(wpij[ib - 1], intensity_g05[ib - 1, :])
        intensity_dn[ib] = (intensity_dn[ib - 1] * np.exp(-dtau / mudn) + 
                           intensity_11dn[ib - 1] * np.exp(-tau[ib - 1] / mu0) + 
                           (1.0 - np.exp(-dtau / mudn)) * scattering_integral)
    intensity_ms = intensity_dn - intensity_1dn
    return intensity_ms[nb - 1]


@jit(nopython=True, cache=True)
def source_function_integrate_up(m: int, muup: float, mu0: float, 
    srfa_mu0_mu: float, srfa_mup_mu: np.ndarray, nlr: int, dtau: float,
    xk: np.ndarray, mug: np.ndarray, wg: np.ndarray, 
    intensity_g05: np.ndarray, intensity_gboa: np.ndarray) -> float:
    """
    Perform source function integration for upward radiative transfer in a
    non-homogeneous BRDF.

    Parameters
    ----------
    m : int
        Fourier moment, where m = 0, 1, 2, ...
    muup : float
        Cosine of the viewing zenith angle (VZA) in the upward line-of-sight
        (LOS), where cos(VZA) < 0.
    mu0 : float
        Cosine of the Solar Zenith Angle (SZA), where cos(SZA) > 0.
    srfa_mu0_mu : float
        BRDF surface reflectance from `mu0` to `mu`.
    srfa_mup_mu : ndarray of shape (ng1,)
        BRDF surface reflectance from `mup` to `mu`.
    nlr : int
        Number of layers in the atmosphere, with the total optical thickness
        tau0 = dtau * nlr.
    dtau : float
        Thickness of each layer element, used as the integration step over tau.
    xk : ndarray of shape (nlr, nk)
        Expansion moments of the scattering phase function, scaled by the
        single-scattering albedo (ssa) and a factor of 2. The term (2k+1) is
        included, where `nk = len(xk)`.
    mug : ndarray of shape (ng2,)
        Gauss nodes used in quadrature for integration.
    wg : ndarray of shape (ng2,)
        Gauss weights corresponding to `mug`.
    intensity_g05 : ndarray of shape (nlr, ng2)
        Radiative transfer equation (RTE) solution at Gauss nodes, evaluated
        at the midpoint of each layer `dtau`.
    intensity_gboa : ndarray of shape (ng1,)
        Downward intensity at the bottom of the atmosphere (BOA) at each Gauss
        node.

    Returns
    -------
    intensity_toa : float
        Top-of-atmosphere (TOA) radiance as a function of `mu`.

    Notes
    -----
    - This function performs source function integration for the upward
    radiative transfer direction using Gauss quadrature over each layer.
    - A scaling factor of 2π is applied at the TOA.
    - The function applies different polynomials (Legendre or Schmidt)
    depending on the Fourier moment `m`.
    - The function calculates the TOA radiance by subtracting single-scattering
    contributions and surface reflectance effects.
    """
    ng2, ng1, nk = len(wg), len(wg) // 2, xk.shape[1]
    mup, nb = -muup, nlr + 1
    tau0, tau = nlr * dtau, np.linspace(0.0, nlr * dtau, nb)
    pk = np.zeros((ng2, nk))
    if m == 0:
        pk0, pku = legendre_polynomial(mu0, nk - 1), legendre_polynomial(muup, nk - 1)
        for ig in range(ng2):
            pk[ig, :] = legendre_polynomial(mug[ig], nk - 1)
    else:
        pk0, pku = schmidt_polynomial(m, mu0, nk - 1), schmidt_polynomial(m, muup, nk - 1)
        for ig in range(ng2):
            pk[ig, :] = schmidt_polynomial(m, mug[ig], nk - 1)

    intensity_11up = np.zeros(nlr)
    for ilr in range(nlr):
        p = np.dot(xk[ilr], pku * pk0)
        intensity_11up[ilr] = p * mu0 / (mu0 + mup) * (1.0 - np.exp(-dtau / mup - dtau / mu0))
    intensity_1up = np.zeros(nb)
    intensity_1up[nb - 1] = 2.0 * srfa_mu0_mu * mu0 * np.exp(-tau0 / mu0)
    for ib in range(nb - 2, -1, -1):
        intensity_1up[ib] = (intensity_1up[ib + 1] * np.exp(-dtau / mup) + 
                            intensity_11up[ib] * np.exp(-tau[ib] / mu0))
    wpij = np.zeros((nlr, ng2))
    for ilr in range(nlr):
        for jg in range(ng2):
            wpij[ilr, jg] = wg[jg] * np.dot(xk[ilr], pku[:] * pk[jg, :])
    intensity_up = np.copy(intensity_1up)
    intensity_up[nb - 1] = (2.0 * np.dot(intensity_gboa * srfa_mup_mu, mug[ng1:ng2] * wg[ng1:ng2]) + 
                           2.0 * srfa_mu0_mu * mu0 * np.exp(-tau0 / mu0))
    for ib in range(nb - 2, -1, -1):
        scattering_integral = np.dot(wpij[ib], intensity_g05[ib, :])
        intensity_up[ib] = (intensity_up[ib + 1] * np.exp(-dtau / mup) + 
                           intensity_11up[ib] * np.exp(-tau[ib] / mu0) + 
                           (1.0 - np.exp(-dtau / mup)) * scattering_integral)
    return (intensity_up - intensity_1up)[0]


# ==============================================================================


@jit(nopython=True, cache=True)
def solve_lattice(nit: int, ng1: int, nm: int, szds: np.array, vzds: np.array, 
    azds: np.array, dtau: float, nlr: int, xk: np.array, ssa: np.array, srfa: float, 
    brdf_type: str, brdf_parameters: np.array) -> np.ndarray:
    """
    Solve the radiative transfer problem for a non-homogeneous atmosphere with
    a bidirectional surface reflectance distribution function (BRDF) using a
    Fourier-mode decomposition and Gauss-Seidel iterations.

    This function computes the top-of-atmosphere (TOA) and bottom-of-atmosphere
    (BOA) radiances for multiple solar zenith angles (SZAs), viewing zenith
    angles (VZAs), and azimuthal angles (AZs), accounting for multiple
    scattering and surface anisotropy.

    Parameters
    ----------
    nit : int
        Maximum number of Gauss-Seidel iterations for convergence in the
        radiative transfer solution. Must be > 0.
    ng1 : int
        Number of Gauss quadrature nodes per hemisphere (i.e., in the upper
        hemisphere). Total Gauss nodes = 2 * ng1.
    nm : int
        Number of Fourier moments to include in the angular expansion (m = 0,
        1, ..., nm-1). Higher values improve accuracy for highly anisotropic
        cases.
    szds : array-like of shape (nsz,)
        Solar zenith angles in degrees, where 0° = overhead sun.
    vzds : array-like of shape (nvzd,)
        Viewing zenith angles in degrees, where 0° = nadir view.
    azds : array-like of shape (naz,)
        Azimuthal angles in degrees.
    dtau : float
        Optical thickness of each atmospheric layer (integration step over
        optical depth τ).
    nlr : int
        Number of atmospheric layers.
    xk : array-like of shape (nlr, nk)
        Expansion coefficients of the scattering phase function.
    ssa : array-like of shape (nlr,)
        Single-scattering albedo for each atmospheric layer.
    srfa : float
        Surface albedo (scalar), used to scale the BRDF.
    brdf_type : str
    brdf_parameters : array-like
    Returns
    -------
    liteMRT_ans : ndarray of shape (nsz, nvzd, naz, 2)
        Array containing the TOA and BOA radiances for each combination of
        SZA, VZA, and AZ:
        - `liteMRT_ans[i_szd, i_vzd, i_azd, 0]`: Normalized TOA radiance at
        the given angles.
        - `liteMRT_ans[i_szd, i_vzd, i_azd, 1]`: Normalized BOA radiance at
        the given angles.

    Notes
    -----
    - The algorithm uses a Fourier-mode expansion (m = 0 to nm-1) to handle
    azimuthal dependence efficiently.
    - The Gauss-Seidel iterative method is applied for each Fourier moment to
    solve the radiative transfer equation (RTE) with non-homogeneous scattering
    and BRDF.
    - The single-scattering contributions are calculated separately and then
    corrected via source function integration for multiple scattering.
    - The BRDF is expanded into Fourier components using `expand_brdf_*`
    functions, which are model-specific.
    - The final radiances are scaled by `0.5 / np.pi` to convert from radiance
    to unit flux (i.e., to represent the radiance expected under a unit flux of
    incoming solar irradiance).
    - This implementation assumes:
        - A plane-parallel atmosphere.
        - No absorption (if ssa < 1.0, absorption is included via ssa).
        - The surface reflectance is scaled by `srfa` and modulated by the BRDF
        model.
    - The function internally uses Numba's `@jit` for performance, so all inputs
    must be NumPy arrays and types must be compatible.

    Examples
    --------
    >>> szds = np.array([0, 30, 60])
    >>> vzds = np.array([0, 30, 60])
    >>> azds = np.array([0, 90, 180])
    >>> dtau = 0.1
    >>> nlr = 10
    >>> xk = np.random.rand(nlr, 10)  # Example phase function moments
    >>> ssa = np.ones(nlr) * 0.8
    >>> srfa = 0.2
    >>> brdf_type = 'Ross-Li'
    >>> brdf_params = [0.5, 0.3]
    >>> result = solve_lattice(nit=50, ng1=8, nm=4, szds=szds, vzds=vzds, azds=azds,
    ...                       dtau=dtau, nlr=nlr, xk=xk, ssa=ssa, srfa=srfa,
    ...                       brdf_type=brdf_type, brdf_parameters=brdf_params)
    >>> print(result.shape)  # (3, 3, 3, 2)
    """
    # ... [function body remains unchanged]

    lite_mrt_ans = np.zeros((len(szds), len(vzds), len(azds), 2))
    mup, _ = gauss_zeroes_weights(0.0, 1.0, ng1)
    srfa_mup_mup = np.zeros((nm, ng1, ng1))
    for m in range(nm):
        srfa_mup_mup[m] = srfa * expand_brdf_mup_mup(mup, m, brdf_type, brdf_parameters)

    for i_szd, szd in enumerate(szds):
        mu0 = np.cos(np.radians(szd))

        srfa_mu0_mup = np.zeros((nm, ng1))
        for m in range(nm):
            srfa_mu0_mup[m] = srfa * expand_brdf_mu0_mup(mup, mu0, m, brdf_type, brdf_parameters)
        intensity_gup_precompute = np.zeros((nm, nlr + 1, ng1))
        intensity_gdn_precompute = np.zeros_like(intensity_gup_precompute)
        for m in range(nm):
            mug, wg, intensity_gup_precompute[m], intensity_gdn_precompute[m] = \
                gauss_seidel_iterations_m(m, mu0, srfa_mu0_mup[m], 
                        srfa_mup_mup[m], nit, ng1, nlr, dtau, 0.5 * ssa * xk)

        for i_vzd, vzd in enumerate(vzds):
            srfa_azrs = srfa * np.array([rho(np.radians(szd), np.radians(vzd), np.radians(azd), 
                                             brdf_type, brdf_parameters) for azd in azds])
            mudn, muup = np.cos(np.radians(vzd)), -np.cos(np.radians(vzd))
            azrs = np.radians(azds)
            srfa_mu0_mu, srfa_mup_mu = np.zeros((nm)), np.zeros((nm, ng1))
            for m in range(nm):
                srfa_mu0_mu[m] = srfa * expand_brdf_mu0_mu(mu0, muup, m, brdf_type, brdf_parameters)
                srfa_mup_mu[m] = srfa * expand_brdf_mup_mu(mup, muup, m, brdf_type, brdf_parameters)

            intensity_toa = single_scattering_up(muup, mu0, azrs, dtau, nlr, 0.5 * ssa * xk, srfa_azrs)
            intensity_boa = single_scattering_down(mudn, mu0, azrs, dtau, nlr, 0.5 * ssa * xk)
            deltm0 = 1.0
            for m in range(nm):
                intensity_gup, intensity_gdn = intensity_gup_precompute[m], intensity_gdn_precompute[m]
                intensity_up05 = 0.5 * (intensity_gup[:-1, :] + intensity_gup[1:, :])
                intensity_dn05 = 0.5 * (intensity_gdn[:-1, :] + intensity_gdn[1:, :])
                intensity_g05 = np.hstack((intensity_up05, intensity_dn05))
                cma = deltm0 * np.cos(m * azrs)
                intensity_ms_toa = source_function_integrate_up(
                    m, muup, mu0, srfa_mu0_mu[m], srfa_mup_mu[m], nlr, dtau, 0.5 * ssa * xk, 
                    mug, wg, intensity_g05, intensity_gdn[nlr, :])
                intensity_ms_boa = source_function_integrate_down(
                    m, mudn, mu0, nlr, dtau, 0.5 * ssa * xk, mug, wg, intensity_g05)
                intensity_toa += intensity_ms_toa * cma
                intensity_boa += intensity_ms_boa * cma
                deltm0 = 2.0

            intensity_toa *= 0.5 / np.pi
            intensity_boa *= 0.5 / np.pi
            lite_mrt_ans[i_szd, i_vzd, :, 0] = intensity_toa
            lite_mrt_ans[i_szd, i_vzd, :, 1] = intensity_boa
    return lite_mrt_ans
