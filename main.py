# pylint: disable=invalid-name
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from testutils.plot_config import set_plot_config, enable_jupyter_inline_backend
from liteMRT.lite_mrt import (
    gauss_seidel_iterations_m,
    single_scattering_up,
    single_scattering_down,
    source_function_integrate_up,
    source_function_integrate_down,
)
from liteMRT.support_func import gauss_zeroes_weights
from liteMRT.brdf import (
    rho,
    expand_brdf_mu0_mu,
    expand_brdf_mu0_mup,
    expand_brdf_mup_mu,
    expand_brdf_mup_mup,
)
from testutils.load_lidort import parse_lidort_all

# Timing storage
timing_results = {
    'gauss_seidel_iterations_m': [],
    'single_scattering_up': [],
    'single_scattering_down': [],
    'source_function_integrate_up': [],
    'source_function_integrate_down': [],
    'expand_brdf_mup_mup': [],
    'expand_brdf_mu0_mup': [],
    'expand_brdf_mu0_mu': [],
    'expand_brdf_mup_mu': [],
    'rho': []
}

def solve_lattice_with_timing(
    nit: int,
    ng1: int,
    nm: int,
    szds: np.array,
    vzds: np.array,
    azds: np.array,
    dtau: float,
    nlr: int,
    xk: np.array,
    ssa: np.array,
    srfa: float,
    brdf_type: str,
    brdf_parameters: np.array,
):
    """
    Same as solve_lattice but with timing measurements for the target functions.
    """
    lite_mrt_ans = np.zeros((len(szds), len(vzds), len(azds), 2))
    mup, _ = gauss_zeroes_weights(0.0, 1.0, ng1)

    srfa_mup_mup = np.zeros((nm, ng1, ng1))
    for m in range(nm):
        # Time expand_brdf_mup_mup
        start_time = time.perf_counter()
        srfa_mup_mup[m] = srfa * expand_brdf_mup_mup(
            mup, m, brdf_type, brdf_parameters
        )
        end_time = time.perf_counter()
        timing_results['expand_brdf_mup_mup'].append(end_time - start_time)

    for i_szd, szd in enumerate(szds):
        mu0 = np.cos(np.radians(szd))

        srfa_mu0_mup = np.zeros((nm, ng1))
        for m in range(nm):
            # Time expand_brdf_mu0_mup
            start_time = time.perf_counter()
            srfa_mu0_mup[m] = srfa * expand_brdf_mu0_mup(
                mup, mu0, m, brdf_type, brdf_parameters
            )
            end_time = time.perf_counter()
            timing_results['expand_brdf_mu0_mup'].append(end_time - start_time)

        intensity_gup_precompute = np.zeros((nm, nlr + 1, ng1))
        intensity_gdn_precompute = np.zeros_like(intensity_gup_precompute)
        for m in range(nm):
            # Time gauss_seidel_iterations_m
            start_time = time.perf_counter()
            (
                mug,
                wg,
                intensity_gup_precompute[m],
                intensity_gdn_precompute[m],
            ) = gauss_seidel_iterations_m(
                m,
                mu0,
                srfa_mu0_mup[m],
                srfa_mup_mup[m],
                nit,
                ng1,
                nlr,
                dtau,
                0.5 * ssa * xk,
            )
            end_time = time.perf_counter()
            timing_results['gauss_seidel_iterations_m'].append(end_time - start_time)

        for i_vzd, vzd in enumerate(vzds):
            # Time rho function calls
            rho_values = []
            for azd in azds:
                start_time = time.perf_counter()
                rho_val = rho(
                    np.radians(szd),
                    np.radians(vzd),
                    np.radians(azd),
                    brdf_type,
                    brdf_parameters,
                )
                end_time = time.perf_counter()
                timing_results['rho'].append(end_time - start_time)
                rho_values.append(rho_val)
            srfa_azrs = srfa * np.array(rho_values)

            mudn = np.cos(np.radians(vzd))
            muup = -mudn

            azrs = np.radians(azds)

            # Calculating BRDF
            srfa_mu0_mu = np.zeros((nm))
            srfa_mup_mu = np.zeros((nm, ng1))
            for m in range(nm):
                # Time expand_brdf_mu0_mu
                start_time = time.perf_counter()
                srfa_mu0_mu[m] = srfa * expand_brdf_mu0_mu(
                    mu0, muup, m, brdf_type, brdf_parameters
                )
                end_time = time.perf_counter()
                timing_results['expand_brdf_mu0_mu'].append(end_time - start_time)
                
                # Time expand_brdf_mup_mu
                start_time = time.perf_counter()
                srfa_mup_mu[m] = srfa * expand_brdf_mup_mu(
                    mup, muup, m, brdf_type, brdf_parameters
                )
                end_time = time.perf_counter()
                timing_results['expand_brdf_mup_mu'].append(end_time - start_time)

            # Time single_scattering_up
            start_time = time.perf_counter()
            intensity_toa = single_scattering_up(
                muup, mu0, azrs, dtau, nlr, 0.5 * ssa * xk, srfa_azrs
            )
            end_time = time.perf_counter()
            timing_results['single_scattering_up'].append(end_time - start_time)

            # Time single_scattering_down
            start_time = time.perf_counter()
            intensity_boa = single_scattering_down(
                mudn, mu0, azrs, dtau, nlr, 0.5 * ssa * xk
            )
            end_time = time.perf_counter()
            timing_results['single_scattering_down'].append(end_time - start_time)

            deltm0 = 1.0

            for m in range(nm):
                intensity_gup, intensity_gdn = (
                    intensity_gup_precompute[m],
                    intensity_gdn_precompute[m],
                )
                intensity_g05 = np.zeros((nlr, 2 * ng1))
                intensity_up05 = 0.5 * (
                    intensity_gup[:-1, :] + intensity_gup[1:, :]
                )
                intensity_dn05 = 0.5 * (
                    intensity_gdn[:-1, :] + intensity_gdn[1:, :]
                )
                intensity_g05 = np.hstack((intensity_up05, intensity_dn05))
                cma = deltm0 * np.cos(m * azrs)

                # Time source_function_integrate_up
                start_time = time.perf_counter()
                intensity_ms_toa = source_function_integrate_up(
                    m,
                    muup,
                    mu0,
                    srfa_mu0_mu[m],
                    srfa_mup_mu[m],
                    nlr,
                    dtau,
                    0.5 * ssa * xk,
                    mug,
                    wg,
                    intensity_g05,
                    intensity_gdn[nlr, :],
                )
                end_time = time.perf_counter()
                timing_results['source_function_integrate_up'].append(end_time - start_time)

                # Time source_function_integrate_down
                start_time = time.perf_counter()
                intensity_ms_boa = source_function_integrate_down(
                    m,
                    mudn,
                    mu0,
                    nlr,
                    dtau,
                    0.5 * ssa * xk,
                    mug,
                    wg,
                    intensity_g05,
                )
                end_time = time.perf_counter()
                timing_results['source_function_integrate_down'].append(end_time - start_time)

                intensity_toa += intensity_ms_toa * cma
                intensity_boa += intensity_ms_boa * cma

                deltm0 = 2.0  # Kronecker delta = 2 for m > 0

            # print(intensity_toa, intensity_boa)
            intensity_toa *= 0.5 / np.pi  # Scale to unit flux on TOA
            intensity_boa *= 0.5 / np.pi  # Scale to unit flux on TOA

            lite_mrt_ans[i_szd, i_vzd, :, 0] = intensity_toa
            lite_mrt_ans[i_szd, i_vzd, :, 1] = intensity_boa
    return lite_mrt_ans

rho_0 = 0.05
Theta = -0.1
k = 0.75

srfa = 0.4
brdf_type = "rahman"
brdf_parameters = np.array([rho_0, Theta, k])


# N_FINE = 5
nlr = 25
dtau = 0.01

# ssa = waer   # single scattering albedo
layer_scales = np.linspace(0, 1, nlr)
ssa = (0.9 + (0.99 - 0.9) * layer_scales).reshape(-1, 1)
# ssa = 0.01

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
xk = aermoms * layer_scales[:, np.newaxis] + raymoms * (1.0 - layer_scales[:, np.newaxis])


N_FINE = 5

dtau /= N_FINE
nlr *= N_FINE
ssa = np.ascontiguousarray(np.repeat(ssa, N_FINE, axis=0))  # shape: (nlr * N_FINE, 1)
xk = np.ascontiguousarray(np.repeat(xk, N_FINE, axis=0))  # shape: (nlr * N_FINE, nk)


szds, vzds, azds, lidort_ans = parse_lidort_all("benchmark/3p8p3_005_test_plot", 0)


nit = 10  # number of iterations
ng1 = 8 * 3  # number of Gauss nodes per hemisphere
nm = 32

liteMRT_ans = solve_lattice_with_timing(
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


print(np.max(np.abs(liteMRT_ans - lidort_ans) / lidort_ans) * 100.0)
print(np.allclose(liteMRT_ans, lidort_ans, rtol=5e-5))

# Print timing results
print("\n" + "="*60)
print("TIMING ANALYSIS")
print("="*60)

for func_name, times in timing_results.items():
    if times:
        total_time = sum(times)
        avg_time = total_time / len(times)
        min_time = min(times)
        max_time = max(times)
        num_calls = len(times)
        
        print(f"\n{func_name}:")
        print(f"  Number of calls: {num_calls}")
        print(f"  Total time:      {total_time:.6f} s")
        print(f"  Average time:    {avg_time:.6f} s")
        print(f"  Min time:        {min_time:.6f} s")
        print(f"  Max time:        {max_time:.6f} s")
    else:
        print(f"\n{func_name}: No calls recorded")

# Calculate total time for all functions
total_measured_time = sum(sum(times) for times in timing_results.values() if times)
print(f"\nTotal measured time: {total_measured_time:.6f} s")


set_plot_config(small_size=8, usetex=True)
enable_jupyter_inline_backend()


nszd = len(szds)


plt.close("all")
fig = plt.figure(figsize=[6, 2.5])

# 主图区域范围（去掉左右 colorbar 的区域）
main_left = 0.15
main_right = 0.8
main_bottom = 0.1
main_top = 0.9

main_gs = GridSpec(
    2,
    nszd,
    left=main_left,
    right=main_right,
    bottom=main_bottom,
    top=main_top,
    wspace=0.1,
    hspace=0.2,
)

az_grid, vz_grid = np.meshgrid(azds, vzds)

plot_vz = np.sin(np.radians(vz_grid))


titlenames = ["TOA", "BOA"]

error_bound = 0.004

# 绘制主图
for id_szd in range(nszd):
    for i in range(2):
        ax = fig.add_subplot(main_gs[i, id_szd], projection="polar")
        if i == 0:
            ax.set_title(f"$\\theta_i$ = {szds[id_szd]:.1f}°")
        plot_lidort = lidort_ans[id_szd, :, :, i]
        plot_liteMRT = liteMRT_ans[id_szd, :, :, i]

        rel_error = (plot_liteMRT - plot_lidort) / plot_lidort * 100.0
        cf_intens = ax.contourf(np.radians(az_grid), plot_vz, plot_liteMRT, cmap="jet", levels=101)
        cf_err = ax.contourf(
            2 * np.pi - np.radians(az_grid),
            plot_vz,
            rel_error,
            levels=np.linspace(-error_bound, error_bound, 101),
            cmap="seismic",
        )

        theta_ticks = np.arange(0, 2 * np.pi, np.pi / 3)  # 每45度一个刻度
        theta_labels = ["0°", "60°", "120°", "180°", "240°", "300°"]

        ax.set_thetagrids(theta_ticks * 180 / np.pi, labels=theta_labels)  # 设置角度刻度和标签
        ax.set_rlabel_position(-90)

        rticks = np.array([15, 30, 45, 60])
        ax.set_rticks(np.sin(np.radians(rticks)))
        if i == 0 and id_szd == 0:
            yticklabels = [f"$\\theta_r=${rtick}°" for rtick in rticks]
        else:
            yticklabels = [f"{rtick}°" for rtick in rticks]
        ax.set_yticklabels(yticklabels, ha="center", va="bottom")
        # ax.yaxis.set_tick_params(pad=-2)  # 默认大约是10，数值越大越远

        ax.set_axisbelow(False)
        ax.grid(color="white")
        ax.xaxis.set_tick_params(pad=-2)  # 默认大约是10，数值越大越远

        ax.text(
            x=-0.3,
            y=0.9,
            s=f"{titlenames[i]}-{id_szd + 1}",
            transform=ax.transAxes,
        )
        # ax.annotate(
        #     '',
        #     xy=(-np.pi/2, 60.0),      # arrow tip
        #     xytext=(0.0, 0.0),  # arrow base
        #     arrowprops=dict(color='black', arrowstyle='->', lw=1)
        # )


# 左 colorbar（Normalized Intensity）
cbar_width = 0.02  # colorbar 宽度
cax_left = fig.add_axes([main_left - 0.07, main_bottom, cbar_width, main_top - main_bottom])
norm = Normalize(vmin=0, vmax=1)
sm = ScalarMappable(cmap="jet", norm=norm)
sm.set_array([])
cb1 = fig.colorbar(sm, cax=cax_left, ticklocation="left")
cb1.set_label("Normalized Intensity [-]")

# 右 colorbar（Relative Error）
cax_right = fig.add_axes([main_right + 0.05, main_bottom, cbar_width, main_top - main_bottom])
norm_err = Normalize(vmin=-error_bound, vmax=error_bound)
sm_err = ScalarMappable(cmap="seismic", norm=norm_err)
sm_err.set_array([])
cb2 = fig.colorbar(sm_err, cax=cax_right)
# cb2.set_ticks(np.linspace(-0.5, 0.5, 5))
cb2.set_label(r"Relative Error [\%]")

fig.savefig("output.png")
