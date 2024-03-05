
if __name__ == "__main__":
    import pathlib, sys
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent.parent)
    )

import numpy as np
from physics.gkp import gkp_state
import matplotlib.pyplot as plt
from utils.visuals import plot_plain_wigner, plot_wigner_bloch_sphere
from algo.metrics import fidelity
from utils.strings import ProgressBar




def project_to_plain(gkp_on_sphere:np.matrix, plain_hilbert_space_size:int)->np.ndarray:
    n = gkp_on_sphere.shape[0]
    N = plain_hilbert_space_size+1

    gkp_on_plain = np.zeros((N, N))
    gkp_on_plain[:n,:n] = gkp_on_sphere
    return gkp_on_plain


def main(
    max_N:int = 40,
    plain_hilbert_space_size:int = 100
):

    # compare to this:
    perfect_plain_gkp = project_to_plain(gkp_state(plain_hilbert_space_size), plain_hilbert_space_size)

    fidelities1 = []
    fidelities2 = []
    ns = []

    prog_bar = ProgressBar(max_N)
    for n in range(1, max_N+1, 1):
        prog_bar.next()

        # Create GKP:
        gkp_on_sphere = gkp_state(n)
        gkp_on_plain = project_to_plain(gkp_on_sphere, plain_hilbert_space_size)

        gkp2 = perfect_plain_gkp.copy()
        gkp2[n+2:, :] = 0.0
        gkp2[:, n+2:] = 0.0

        # Compare to perfect:
        fidelity1 = fidelity(gkp_on_plain, perfect_plain_gkp) # type: ignore
        fidelity2 = fidelity(gkp2, perfect_plain_gkp) # type: ignore

        fidelities1.append(fidelity1)
        fidelities2.append(fidelity2)
        ns.append(n)

        # plot_plain_wigner(gkp_on_plain)
        # plot_plain_wigner(gkp_on_sphere)
        # plot_wigner_bloch_sphere(gkp_on_sphere, num_points=50)

    prog_bar.clear()

    plt.plot(ns, fidelities1, '--')
    plt.plot(ns, fidelities2)


    print("Done")



if __name__ == "__main__":
    main()
    print("Done")