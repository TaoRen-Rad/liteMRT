# pylint: disable=invalid-name
"""
Plotting script: Load computed results and create visualization.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from .plot_config import set_plot_config, enable_jupyter_inline_backend

# BASE_NAME = "homo_black"

def plot_results(BASE_NAME):
    """Main plotting function."""
    # Load computed results
    print("Loading computation results...")
    try:
        data = np.load(f"results/{BASE_NAME}.npz")
        szds = data['szds']
        vzds = data['vzds']
        azds = data['azds']
        lidort_ans = data['lidort_ans']
        liteMRT_ans = data['liteMRT_ans']
        max_rel_error = data['max_rel_error']
        is_close = data['is_close']
        
        print(f"Loaded results - Max relative error: {max_rel_error:.6f}%")
        print(f"Solutions are close: {is_close}")
        
    except FileNotFoundError:
        print("Error: computation_results.npz not found!")
        print("Please run compute_results.py first to generate the data.")
        return
    
    # Set up plotting configuration
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
    error_bound = 0.04
    
    print("Creating polar plots...")
    
    # 绘制主图
    for id_szd in range(nszd):
        for i in range(2):
            ax = fig.add_subplot(main_gs[i, id_szd], projection="polar")
            # if i == 0:
            #     ax.set_title(f"$\\theta_i$ = {szds[id_szd]:.1f}°")
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
                
                # Add arc annotation outside the plot area
                from matplotlib.patches import Arc
                import matplotlib.transforms as transforms
                
                # Create transform that allows drawing outside axes
                trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
                
                # Draw curved arrow using FancyArrowPatch for phi
                from matplotlib.patches import FancyArrowPatch
                
                # Calculate start and end points for the arc
                start_angle = np.radians(5)
                end_angle = np.radians(55)
                arc_radius = 0.65  # radius in axes coordinates
                
                start_x = 0.5 + arc_radius * np.cos(start_angle)
                start_y = 0.5 + arc_radius * np.sin(start_angle)
                end_x = 0.5 + arc_radius * np.cos(end_angle)
                end_y = 0.5 + arc_radius * np.sin(end_angle)
                
                # Create curved arrow for phi
                fancy_arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                            connectionstyle="arc3,rad=0.3",  # curved connection
                                            arrowstyle='->', 
                                            mutation_scale=5,  # arrow head size
                                            linewidth=1.0, 
                                            color='black',
                                            clip_on=False)
                fancy_arrow.set_transform(ax.transAxes)
                ax.add_patch(fancy_arrow)
                
                # Add phi label outside the plot area
                ax.annotate(r'$\varphi$', xy=(1.0, 0.8), xycoords='axes fraction',
                           ha='center', va='center', fontsize=10, clip_on=False)
                
            else:
                yticklabels = [f"{rtick}°" for rtick in rticks]
            ax.set_yticklabels(yticklabels, ha="center", va="bottom")
            # ax.yaxis.set_tick_params(pad=-2)  # 默认大约是10，数值越大越远
    
            ax.set_axisbelow(False)
            ax.grid(color="white")
            ax.xaxis.set_tick_params(pad=-2)  # 默认大约是10，数值越大越远
    
            ax.text(
                x=-0.35,
                y=0.85,
                s=f"{titlenames[i]}-{id_szd + 1}",
                transform=ax.transAxes,
            )
            if i == 0:
                ax.text(
                    x = 0.5,
                    y = 1.12,
                    s=f"$\\theta_i$ = {szds[id_szd]:.1f}°",
                    transform=ax.transAxes,
                    ha="center",
                    va = "bottom",
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
    
    print("Saving plot to output.png...")
    fig.savefig(f"results/{BASE_NAME}.png")
    print("Plot saved successfully!")

if __name__ == "__main__":
    plot_results()
