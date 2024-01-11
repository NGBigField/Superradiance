
if __name__ == "__main__":
    import pathlib, sys
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent.parent)
    )

from physics.gkp import gkp_state
import matplotlib.pyplot as plt
from utils.visuals import plot_plain_wigner
from algo.metrics import fidelity

N_ideal=100
rho_ideal=gkp_state(N_ideal, form="square")


#transform for bigger size of N_ideal+1
rho_mod = [[0 for col in range(N_ideal+1)] for row in range(N_ideal+1)]

# general loop
for m in range(N_ideal):
    N=m+1
    rho=gkp_state(N, form="square")
    for i in range(N+1):
        for j in range(N+1):
            rho_mod[i][j]=rho[i][j]

    fid = fidelity(rho_ideal,rho_mod)
    print(fid)
