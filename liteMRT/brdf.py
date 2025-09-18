"""
BRDF Module
"""

from math import erfc
import numpy as np
from numba import jit
from .support_func import gauss_zeroes_weights


# def coxmunk_function(pars, xj, sxj, xi, sxi, phi, cphi, skphi):
@jit(nopython=True, cache=True)
def coxmunk(theta_i: float, theta_r: float, phi: float, parameters: np.array):
    """
    Cox-Munk bidirectional reflectance distribution function (BRDF).

    This function computes the Cox-Munk ocean surface BRDF kernel with optional
    shadow effect correction.

    Parameters
    ----------
    pars : array-like
        Parameters for the Cox-Munk model:
        - pars[0]: Surface roughness parameter (sigma^2)
        - pars[1]: Refractive index squared minus 1
        - pars[2]: Shadow effect flag (0 = no shadow, non-zero = include shadow)
    xj, sxj : float
        Cosine and sine of the outgoing zenith angle
    xi, sxi : float
        Cosine and sine of the incident zenith angle
    phi, cphi, skphi : float
        Azimuth angle and its cosine/sine

    Returns
    -------
    float
        Cox-Munk BRDF kernel value

    Notes
    -----
    Shadow effect is controlled by pars[2]. If pars[2] == 0, no shadow
    effect is included. The function expects 3 parameters in pars.
    """

    # phi = np.pi - phi

    pars = parameters
    xj = np.cos(theta_r)
    sxj = np.sin(theta_r)
    xi = np.cos(theta_i)
    sxi = np.sin(theta_i)
    cphi = np.cos(phi)
    # skphi = np.sin(phi)

    # Constants
    critical_exp = 88.0

    # Initialize
    coxmunk_kernel = 0.0
    # xphi = np.pi - phi
    ckphi = -cphi

    # Scatter angles
    z = xi * xj + sxi * sxj * ckphi
    z = min(z, 1.0)  # Clamp to avoid numerical issues
    z1 = np.arccos(z)
    z2 = np.cos(z1 * 0.5)

    # Fresnel coefficients
    z2_sq_m1 = z2 * z2 - 1.0
    h1 = pars[1] * z2
    h2 = np.sqrt(pars[1] + z2_sq_m1)
    rp = (h1 - h2) / (h1 + h2)
    rl = (z2 - h2) / (z2 + h2)
    xmp = 0.5 * (rp * rp + rl * rl)

    # Cox-Munk function calculation
    a = 2.0 * z2
    b = (xi + xj) / a
    b = min(b, 1.0)  # Clamp to valid range

    a = np.pi / 2 - np.arcsin(b)
    ta = np.tan(a)
    argument = ta * ta / pars[0]

    if argument < critical_exp:
        prob = np.exp(-argument)
        fac1 = prob / pars[0]
        fac2 = 0.25 / xi / (b**4)
        coxmunk_kernel = xmp * fac1 * fac2 / xj

    # Return early if no shadow effect
    if pars[2] == 0.0:
        return coxmunk_kernel

    # Shadow effect calculation
    s1 = np.sqrt(pars[0] / np.pi)
    s3 = 1.0 / np.sqrt(pars[0])
    s2 = s3 * s3

    # Shadow for incident direction
    xxi = xi * xi
    dcot = xi / np.sqrt(1.0 - xxi)
    t1 = np.exp(-dcot * dcot * s2)
    t2 = erfc(dcot * s3)
    shadowi = 0.5 * (s1 * t1 / dcot - t2)

    # Shadow for outgoing direction
    xxj = xj * xj
    dcot = xj / np.sqrt(1.0 - xxj)
    t1 = np.exp(-dcot * dcot * s2)
    t2 = erfc(dcot * s3)
    shadowr = 0.5 * (s1 * t1 / dcot - t2)

    # Apply shadow correction
    shadow = 1.0 / (1.0 + shadowi + shadowr)
    coxmunk_kernel *= shadow

    return coxmunk_kernel


# pylint: disable=invalid-name
@jit(nopython=True, cache=True)
def rahman(theta_i: float, theta_r: float, phi: float, parameters: np.array):  # disable: C0103
    """Rahman BRDF function

    Parameters
    ----------
    theta_i : float
        incident angle, in radiance
    theta_r : float
        reflected angle, in radiance
    phi : float
        relative azimuth angle, in radiance
    parameters : np.array
        rho_0, Theta, k

    Returns
    -------
    _type_
        _description_
    """
    rho_0, Theta, k = parameters
    phi = np.pi - phi
    
    # Precompute trigonometric functions
    cos_theta_i = np.cos(theta_i)
    cos_theta_r = np.cos(theta_r)
    sin_theta_i = np.sin(theta_i)
    sin_theta_r = np.sin(theta_r)
    tan_theta_i = np.tan(theta_i)
    tan_theta_r = np.tan(theta_r)
    cos_phi = np.cos(phi)
    
    term1 = (cos_theta_i * cos_theta_r) ** (k - 1) / (
        cos_theta_i + cos_theta_r
    ) ** (1 - k)
    cos_g = cos_theta_i * cos_theta_r + sin_theta_i * sin_theta_r * cos_phi
    F_g = (1 - Theta**2) / (1 + Theta**2 + 2 * Theta * cos_g) ** (1.5)
    G = (
        tan_theta_i ** 2
        + tan_theta_r ** 2
        - 2 * tan_theta_i * tan_theta_r * cos_phi
    ) ** (0.5)
    R_G = (1 - rho_0) / (1 + G)
    return rho_0 * term1 * F_g * (1 + R_G)


@jit(nopython=True, cache=True)
def rho(
    theta_i: float,
    theta_r: float,
    phi: float,
    brdf_type: str,
    brdf_parameters: np.array,
) -> float:
    """Calculate the surface reflectance of a BRDF ground surface

    Args:
        theta_i (float): Incident zenith angle (in radians), which is the angle
            between the surface normal and the incoming light direction.
        theta_r (float): Reflected zenith angle (in radians), which is the angle
            between the surface normal and the outgoing light direction.
        phi (float): Relative azimuth angle (in radians) between the incident
            and reflected directions.
        nu (float): Wavelength or frequency of the incident light. Used to
            adjust reflectance with respect to spectral shifts.
        nu_0 (float): Reference wavelength or frequency around which spectral
            adjustments are made.
        r_s (float): Scale factor for surface reflectance. Controls the overall
            reflectance of the surface.
        rho_0 (float): Base reflectance coefficient, representing the
            fundamental reflectance of the surface under standard conditions.
        Theta (float): Roughness parameter, which adjusts the sharpness of the
            angular distribution of the surface reflectance.
        k (float): Exponent parameter in the reflectance model, controlling
            the angular dependence of reflectance. Affects how quickly
            reflectance changes with angle.
        w (float): Weight parameter for the linear term in the spectral
            adjustment. Modifies reflectance based on the difference between
            `nu` and `nu_0`.
        s (float): Weight parameter for the first-order (linear) term in the
            spectral adjustment function.
        q (float): Weight parameter for the second-order (quadratic) term in
            the spectral adjustment function.

    Returns:
        float: Calculated surface reflectance value based on the BRDF model,
            taking into account incident and reflected angles, azimuthal
            dependence, and spectral adjustments.
    """
    brdf = 0.0
    if brdf_type == "rahman":
        brdf = rahman(theta_i, theta_r, phi, brdf_parameters)
    elif brdf_type == "coxmunk":
        brdf = coxmunk(theta_i, theta_r, phi, brdf_parameters)
    return brdf


@jit(nopython=True, cache=True)
def gauss_integrate_internal(
    x1: float,
    x2: float,
    n: int,
    m: int,
    theta_i: float,
    theta_r: float,
    brdf_type: str,
    brdf_parameters: np.array,
) -> float:
    """
    Internal Gaussian integration function specifically for BRDF integration.
    This avoids the dynamic function passing that causes caching issues.
    """
    x, w = gauss_zeroes_weights(x1, x2, n)
    val = 0.0
    for i in range(n):
        phi = x[i]
        integrand_val = (
            rho(theta_i, theta_r, phi, brdf_type, brdf_parameters) * np.cos(m * phi) / (2 * np.pi)
        )
        val += w[i] * integrand_val
    return val


@jit(nopython=True, cache=True)
def brdf_expand(
    m: int,
    theta_i: float,
    theta_r: float,
    brdf_type: str,
    brdf_parameters: np.array,
) -> float:
    """
    Compute the m-th Fourier coefficient of the BRDF.

    Args:
        m: Fourier mode number
        theta_i: Incident zenith angle (radians)
        theta_r: Reflected zenith angle (radians)
        brdf_type: Type of BRDF model
        brdf_parameters: Parameters for the BRDF model

    Returns:
        float: The m-th Fourier coefficient
    """
    return gauss_integrate_internal(
        0, 2 * np.pi, 100, m, theta_i, theta_r, brdf_type, brdf_parameters
    )


@jit(nopython=True, cache=True)
def expand_brdf_mup_mup(
    mup: np.ndarray, m: int, brdf_type: str, brdf_parameters: np.array
) -> np.ndarray:
    """
    Compute the (ng1 * ng1) matrix srfa_mup_mup.

    Parameters
    ----------
    ng1 : int
        Number of discrete angles.
    mup : np.ndarray
        Array of cos(θ) values for the outgoing directions.
    m   : int
        Angular order of the expansion.
    r_s : float
        Surface reflectance coefficient.
    rho_0, Theta, k : float
        Parameters required by brdf_expand.
    """
    ng1 = len(mup)

    srfa = np.empty((ng1, ng1), dtype=float)

    for igx in range(ng1):
        theta1 = np.arccos(mup[igx])
        for igy in range(ng1):
            theta2 = np.arccos(mup[igy])
            srfa[igx, igy] = brdf_expand(m, theta1, theta2, brdf_type, brdf_parameters)
    return srfa


# ------------------------------------------------------------------
#  2. srfa_mu0_mup  –  vector of BRDF values for (μ0, μ′) pairs
# ------------------------------------------------------------------
@jit(nopython=True, cache=True)
def expand_brdf_mu0_mup(
    mup: np.ndarray,
    mu0: float,
    m: int,
    brdf_type: str,
    brdf_parameters: np.array,
) -> np.ndarray:
    """
    Compute the (ng1,) vector srfa_mu0_mup.

    Parameters are analogous to the previous function.
    """
    ng1 = len(mup)
    theta1 = np.arccos(mu0)
    srfa = np.empty(ng1, dtype=float)

    for ig in range(ng1):
        theta2 = np.arccos(mup[ig])
        srfa[ig] = brdf_expand(m, theta1, theta2, brdf_type, brdf_parameters)
    return srfa


# ------------------------------------------------------------------
#  3. srfa_mu0_mu  –  single BRDF value for (μ0, μup)
# ------------------------------------------------------------------
@jit(nopython=True, cache=True)
def expand_brdf_mu0_mu(
    mu0: float, muup: float, m: int, brdf_type: str, brdf_parameters: np.array
) -> float:
    """
    Compute the scalar srfa_mu0_mu.

    Note that the incoming direction is *downward* (-muup) in the
    original code, hence the negative sign.
    """

    theta1 = np.arccos(mu0)
    theta2 = np.arccos(-muup)

    return brdf_expand(m, theta1, theta2, brdf_type, brdf_parameters)


# ------------------------------------------------------------------
#  4. srfa_mup_mu  –  vector of BRDF values for (μ′, μup) pairs
# ------------------------------------------------------------------
@jit(nopython=True, cache=True)
def expand_brdf_mup_mu(
    mup: np.ndarray,
    muup: float,
    m: int,
    brdf_type: str,
    brdf_parameters: np.array,
) -> np.ndarray:
    """
    Compute the (ng1,) vector srfa_mup_mu.
    """
    ng1 = len(mup)

    theta2 = np.arccos(-muup)
    srfa = np.empty(ng1, dtype=float)

    for ig in range(ng1):
        theta1 = np.arccos(mup[ig])
        srfa[ig] = brdf_expand(m, theta1, theta2, brdf_type, brdf_parameters)
    return srfa


# ------------------------------------------------------------------
#  Top-level convenience wrapper (keeps the original API)
# ------------------------------------------------------------------
@jit(nopython=True, cache=True)
def expand_brdf_full(
    mup: np.ndarray,
    mu0: float,
    m: int,
    muup: float,
    brdf_type: str,
    brdf_parameters: np.array,
):
    """
    Return all four BRDF-expansion results, mimicking the original
    single-function interface.
    """
    return (
        expand_brdf_mup_mup(mup, m, brdf_type, brdf_parameters),
        expand_brdf_mu0_mup(mup, mu0, m, brdf_type, brdf_parameters),
        expand_brdf_mu0_mu(mu0, muup, m, brdf_type, brdf_parameters),
        expand_brdf_mup_mu(mup, muup, m, brdf_type, brdf_parameters),
    )
