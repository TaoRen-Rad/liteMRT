import matplotlib.pyplot as plt


def set_plot_config(small_size=10, medium_size=10, big_size=10, usetex=False):
    rc = {
        "xtick.major.size": 2,
        "xtick.major.width": 0.5,
        "ytick.major.size": 2,
        "ytick.major.width": 0.5,
        "xtick.bottom": True,
        "ytick.left": True,
        "font.size": medium_size,
        "axes.titlesize": medium_size,
        "axes.labelsize": medium_size,
        "xtick.labelsize": small_size,
        "ytick.labelsize": small_size,
        "legend.fontsize": small_size,
        "figure.titlesize": big_size,
        "savefig.dpi": 300,
        "figure.dpi": 300,
        "text.usetex": usetex,
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif", "Nimbus Roman No9 L", "Times"],
    }

    plt.rcParams.update(rc)


def enable_jupyter_inline_backend():
    """
    启用 Jupyter inline 后端显示图像，并设置 bbox_inches 参数。
    仅在 Jupyter 环境下有效。
    """
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is not None:
            ipython.run_line_magic("matplotlib", "inline")
            ipython.run_line_magic(
                "config", "InlineBackend.print_figure_kwargs = {'bbox_inches': None}"
            )
    except Exception as e:
        print("无法启用 Jupyter inline backend。错误信息: ", e)
