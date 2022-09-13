# For typing hints
import typing as typ
# For plotting:
import matplotlib.pyplot as plt
import qutip
# For tools and helpers:
from utils import decorators, visuals


def plot_city(state):
    mat = state.data
    labels = all_qubits_strings(num_qubits=2)
    title = f"Tomography on \n'{run_type.name}' \nfor circuit '{circ_func.__name__}'"
    fig, ax = qutip.matrix_histogram_complex(
        mat, 
        xlabels=labels, 
        ylabels=labels,
        title=title
    )
    x, y = 0.40, 0.85
    plt.figtext(x, y, title, fontsize=16) 

    return fig

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