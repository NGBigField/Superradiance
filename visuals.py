# ==================================================================================== #
# |                                 Imports                                          | #
# ==================================================================================== #

# For typing hints
import typing as typ

# For plotting:
import matplotlib as mpl
import matplotlib.pyplot as plt
import qutip
from qiskit.visualization import plot_state_city
from qutip.matplotlib_utilities import complex_phase_cmap
from mpl_toolkits.mplot3d import Axes3D

# Everyone needs numpy in their life:
import numpy as np

# For tools and helpers:
from utils import decorators, visuals

# For type hints:
from typing import (
    Optional,
    Union,
)
_QobjType = qutip.qobj.Qobj
from qiskit.quantum_info.states.densitymatrix import DensityMatrix

# For function version detection:
from packaging.version import parse as parse_version


# ==================================================================================== #
# |                             Inner Functions                                      | #
# ==================================================================================== #

if parse_version(mpl.__version__) >= parse_version('3.4'):
    def _axes3D(fig, *args, **kwargs):
        ax = Axes3D(fig, *args, auto_add_to_figure=False, **kwargs)
        return fig.add_axes(ax)
else:
    def _axes3D(*args, **kwargs):
        return Axes3D(*args, **kwargs)


# ==================================================================================== #
# |                            Declared Functions                                    | #
# ==================================================================================== #


def plot_city(M:np.matrix, title:Optional[str]=None, ax=None):

    # Define common symbols:
    pi = np.pi

    n = np.size(M)
    xpos, ypos = np.meshgrid(range(M.shape[0]), range(M.shape[1]))
    xpos = xpos.T.flatten() - 0.5
    ypos = ypos.T.flatten() - 0.5
    zpos = np.zeros(n)
    dx = dy = 0.8 * np.ones(n)
    Mvec = M.flatten()
    dz = abs(Mvec).tolist()[0]

    # make small numbers real, to avoid random colors
    idx, _ = np.where(abs(Mvec) < 0.001)
    Mvec[idx] = abs(Mvec[idx])

    phase_min = -pi
    phase_max = pi

    norm = mpl.colors.Normalize(phase_min, phase_max)
    cmap = complex_phase_cmap()

    colors = cmap(norm(np.angle(Mvec)))
    colors_l = colors[0]

    if ax is None:
        fig = plt.figure()
        ax = _axes3D(fig, azim=-35, elev=35)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)
    # ax.bar3d(xpos, ypos, zpos, dx, dy, dz)

    if title:
        ax.set_title(title)

    # x axis
    xtics = -0.5 + np.arange(M.shape[0])
    ax.axes.w_xaxis.set_major_locator(plt.FixedLocator(xtics))
    if xlabels:
        nxlabels = len(xlabels)
        if nxlabels != len(xtics):
            raise ValueError(f"got {nxlabels} xlabels but needed {len(xtics)}")
        ax.set_xticklabels(xlabels)
    ax.tick_params(axis='x', labelsize=12)

    # y axis
    ytics = -0.5 + np.arange(M.shape[1])
    ax.axes.w_yaxis.set_major_locator(plt.FixedLocator(ytics))
    if ylabels:
        nylabels = len(ylabels)
        if nylabels != len(ytics):
            raise ValueError(f"got {nylabels} ylabels but needed {len(ytics)}")
        ax.set_yticklabels(ylabels)
    ax.tick_params(axis='y', labelsize=12)

    # z axis
    if limits and isinstance(limits, list):
        ax.set_zlim3d(limits)
    else:
        ax.set_zlim3d([0, 1])  # use min/max
    # ax.set_zlabel('abs')

    # color axis
    if colorbar:
        cax, kw = mpl.colorbar.make_axes(ax, shrink=.75, pad=.0)
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb.set_ticks([-pi, -pi / 2, 0, pi / 2, pi])
        cb.set_ticklabels(
            (r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'))
        cb.set_label('arg')

    return fig, ax

def _plot_city(state:np.matrix, title:Optional[str]=None):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    rho = DensityMatrix(state)

    # get the real and imag parts of rho
    datareal = np.real(rho.data)
    dataimag = np.imag(rho.data)

    # get the labels
    column_names = [bin(i)[2:].zfill(num) for i in range(2**num)]
    row_names = [bin(i)[2:].zfill(num) for i in range(2**num)]

    lx = len(datareal[0])  # Work out matrix dimensions
    ly = len(datareal[:, 0])
    xpos = np.arange(0, lx, 1)  # Set up a mesh of positions
    ypos = np.arange(0, ly, 1)
    xpos, ypos = np.meshgrid(xpos + 0.25, ypos + 0.25)

    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(lx * ly)

    dx = 0.5 * np.ones_like(zpos)  # width of bars
    dy = dx.copy()
    dzr = datareal.flatten()
    dzi = dataimag.flatten()

    if color is None:
        color = ["#648fff", "#648fff"]
    else:
        if len(color) != 2:
            raise ValueError("'color' must be a list of len=2.")
        if color[0] is None:
            color[0] = "#648fff"
        if color[1] is None:
            color[1] = "#648fff"
    if ax_real is None and ax_imag is None:
        # set default figure size
        if figsize is None:
            figsize = (15, 5)

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    elif ax_real is not None:
        fig = ax_real.get_figure()
        ax1 = ax_real
        ax2 = ax_imag
    else:
        fig = ax_imag.get_figure()
        ax1 = None
        ax2 = ax_imag

    max_dzr = max(dzr)
    min_dzr = min(dzr)
    min_dzi = np.min(dzi)
    max_dzi = np.max(dzi)

    if ax1 is not None:
        fc1 = generate_facecolors(xpos, ypos, zpos, dx, dy, dzr, color[0])
        for idx, cur_zpos in enumerate(zpos):
            if dzr[idx] > 0:
                zorder = 2
            else:
                zorder = 0
            b1 = ax1.bar3d(
                xpos[idx],
                ypos[idx],
                cur_zpos,
                dx[idx],
                dy[idx],
                dzr[idx],
                alpha=alpha,
                zorder=zorder,
            )
            b1.set_facecolors(fc1[6 * idx : 6 * idx + 6])

        xlim, ylim = ax1.get_xlim(), ax1.get_ylim()
        x = [xlim[0], xlim[1], xlim[1], xlim[0]]
        y = [ylim[0], ylim[0], ylim[1], ylim[1]]
        z = [0, 0, 0, 0]
        verts = [list(zip(x, y, z))]

        pc1 = Poly3DCollection(verts, alpha=0.15, facecolor="k", linewidths=1, zorder=1)

        if min(dzr) < 0 < max(dzr):
            ax1.add_collection3d(pc1)
        ax1.set_xticks(np.arange(0.5, lx + 0.5, 1))
        ax1.set_yticks(np.arange(0.5, ly + 0.5, 1))
        if max_dzr != min_dzr:
            ax1.axes.set_zlim3d(np.min(dzr), max(np.max(dzr) + 1e-9, max_dzi))
        else:
            if min_dzr == 0:
                ax1.axes.set_zlim3d(np.min(dzr), max(np.max(dzr) + 1e-9, np.max(dzi)))
            else:
                ax1.axes.set_zlim3d(auto=True)
        ax1.get_autoscalez_on()
        ax1.xaxis.set_ticklabels(row_names, fontsize=14, rotation=45, ha="right", va="top")
        ax1.yaxis.set_ticklabels(column_names, fontsize=14, rotation=-22.5, ha="left", va="center")
        ax1.set_zlabel("Re[$\\rho$]", fontsize=14)
        for tick in ax1.zaxis.get_major_ticks():
            tick.label1.set_fontsize(14)

    if ax2 is not None:
        fc2 = generate_facecolors(xpos, ypos, zpos, dx, dy, dzi, color[1])
        for idx, cur_zpos in enumerate(zpos):
            if dzi[idx] > 0:
                zorder = 2
            else:
                zorder = 0
            b2 = ax2.bar3d(
                xpos[idx],
                ypos[idx],
                cur_zpos,
                dx[idx],
                dy[idx],
                dzi[idx],
                alpha=alpha,
                zorder=zorder,
            )
            b2.set_facecolors(fc2[6 * idx : 6 * idx + 6])

        xlim, ylim = ax2.get_xlim(), ax2.get_ylim()
        x = [xlim[0], xlim[1], xlim[1], xlim[0]]
        y = [ylim[0], ylim[0], ylim[1], ylim[1]]
        z = [0, 0, 0, 0]
        verts = [list(zip(x, y, z))]

        pc2 = Poly3DCollection(verts, alpha=0.2, facecolor="k", linewidths=1, zorder=1)

        if min(dzi) < 0 < max(dzi):
            ax2.add_collection3d(pc2)
        ax2.set_xticks(np.arange(0.5, lx + 0.5, 1))
        ax2.set_yticks(np.arange(0.5, ly + 0.5, 1))
        if min_dzi != max_dzi:
            eps = 0
            ax2.axes.set_zlim3d(np.min(dzi), max(np.max(dzr) + 1e-9, np.max(dzi) + eps))
        else:
            if min_dzi == 0:
                ax2.set_zticks([0])
                eps = 1e-9
                ax2.axes.set_zlim3d(np.min(dzi), max(np.max(dzr) + 1e-9, np.max(dzi) + eps))
            else:
                ax2.axes.set_zlim3d(auto=True)

        ax2.xaxis.set_ticklabels(row_names, fontsize=14, rotation=45, ha="right", va="top")
        ax2.yaxis.set_ticklabels(column_names, fontsize=14, rotation=-22.5, ha="left", va="center")
        ax2.set_zlabel("Im[$\\rho$]", fontsize=14)
        for tick in ax2.zaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
        ax2.get_autoscalez_on()

    fig.suptitle(title, fontsize=16)
    if ax_real is None and ax_imag is None:
        matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)

def plot_superradiance_evolution(times, energies, intensities):
    # Plot:
    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)

    axes[0].plot(times[0:-1], energies[0:-1])    
    axes[0].grid(which='major')
    axes[0].set_xlabel('Time [sec] '    , fontdict=dict(size=16) )
    axes[0].set_ylabel('Energy     '    , fontdict=dict(size=16) )
    axes[0].set_title('Evolution   '    , fontdict=dict(size=16) )

    axes[1].plot(times[0:-1], intensities)    
    axes[1].grid(which='major')
    axes[1].set_xlabel('Time [sec] '    , fontdict=dict(size=16) )
    axes[1].set_ylabel('Intensity  '    , fontdict=dict(size=16) )
    visuals.save_figure()
    plt.show()



# ==================================================================================== #
# |                                  Tests                                           | #
# ==================================================================================== #

def tests():
    from coherentcontrol import _test_pi_pulse
    _test_pi_pulse(MAX_ITER=2, N=2)

if __name__ == "__main__":
    tests()