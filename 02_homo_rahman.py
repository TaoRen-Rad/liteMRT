# pylint: disable=invalid-name
"""
Computation script: Calculate liteMRT and LIDORT results and save to file.
"""
import numpy as np
from liteMRT.lite_mrt import solve_lattice
from testutils.load_lidort import parse_lidort_all
from testutils.plot_results import plot_results

BASE_NAME = "02_homo_rahman"

def main():
    """Main computation function."""
    # BRDF parameters
    rho_0 = 0.05
    Theta = -0.1
    k = 0.75
    
    srfa = 0.4
    brdf_type = "rahman"
    brdf_parameters = np.array([rho_0, Theta, k])
    
    # brdf_type = "lambertian"
    # brdf_parameters = np.array([])
    # Atmospheric parameters
    nlr = 25
    dtau = 0.01
    
    # Single scattering albedo
    layer_scales = np.linspace(0, 1, nlr)
    # ssa = (0.9 + (0.99 - 0.9) * layer_scales).reshape(-1, 1)
    ssa = np.ones_like(layer_scales).reshape(-1, 1) * 0.9
    
    # Phase function moments
    nmoments_input = 80
    gaer = 0.8
    waer = 0.95
    taer = 0.5
    aermoms = np.zeros(nmoments_input + 1)  # 包含 l=0 到 l=nmoments_input
    aermoms[0] = 1.0
    
    L = np.arange(1, nmoments_input + 1)
    aermoms[1:] = (2 * L + 1) * (gaer**L)
    
    raymoms = np.zeros(nmoments_input + 1)
    raymoms[0] = 1.0
    raymoms[1] = 0.0
    raymoms[2] = 4.920062061e-01
    
    xk = np.zeros((nlr, len(aermoms)))
    xk[:] = aermoms # * layer_scales[:, np.newaxis] + raymoms * (1.0 - layer_scales[:, np.newaxis])
    
    # Fine grid refinement
    N_FINE = 2
    
    dtau /= N_FINE
    nlr *= N_FINE
    ssa = np.ascontiguousarray(np.repeat(ssa, N_FINE, axis=0))  # shape: (nlr * N_FINE, 1)
    xk = np.ascontiguousarray(np.repeat(xk, N_FINE, axis=0))  # shape: (nlr * N_FINE, nk)
    
    # Load LIDORT reference data
    szds, vzds, azds, lidort_ans = parse_lidort_all(f"benchmark/{BASE_NAME}", 0)
    
    # liteMRT computation parameters
    nit = 10  # number of iterations
    ng1 = 16  # number of Gauss nodes per hemisphere
    nm = 24
    
    print("Computing liteMRT solution...")
    liteMRT_ans = solve_lattice(
        nit,
        ng1,
        nm,
        szds,
        vzds,
        azds,
        dtau,
        nlr,
        xk,
        ssa,
        srfa,
        brdf_type,
        brdf_parameters,
    )
    
    # Print comparison statistics
    max_rel_error = np.max(np.abs(liteMRT_ans - lidort_ans) / lidort_ans) * 100.0
    is_close = np.allclose(liteMRT_ans, lidort_ans, rtol=5e-4)
    
    print(f"Maximum relative error: {max_rel_error:.6f}%")
    print(f"Solutions are close (rtol=4e-5): {is_close}")
    
    # Save results to npz file
    print("Saving results to computation_results.npz...")
    np.savez_compressed(
        f"results/{BASE_NAME}.npz",
        szds=szds,
        vzds=vzds,
        azds=azds,
        lidort_ans=lidort_ans,
        liteMRT_ans=liteMRT_ans,
        max_rel_error=max_rel_error,
        is_close=is_close
    )
    
    print("Computation completed and results saved!")
    plot_results(BASE_NAME)

if __name__ == "__main__":
    main()
