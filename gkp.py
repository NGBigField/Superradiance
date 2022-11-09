import qutip
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from coherentcontrol import CoherentControl


def goal_gkp_state(num_moments:int):
    # Get un-rotated state:
    psi = _goal_gkp_state_ket(num_moments)
    rho = qutip.ket2dm(psi).full()
    # Rotate the state:
    coherent_control = CoherentControl(num_moments=num_moments)
    gkp = coherent_control.pulse_on_state(rho, x=pi/2)
    return gkp
    
def _goal_gkp_state_ket(num_moments:int):
    n = num_moments + 1 
    alpha =  np.sqrt(2*pi)

    S = qutip.squeeze(n, 1)

    D_alpha = qutip.displace(n, alpha) + qutip.displace(n, -alpha)

    psi = qutip.basis(n, 0)
    psi = S * psi
    psi = psi.unit()

    psi = D_alpha * psi
    psi = psi.unit()
    psi = D_alpha * psi

    psi = psi.unit()

    return psi

def main():
    num_moments = 40

    psi = _goal_gkp_state_ket(num_moments)
    rho = goal_gkp_state(num_moments)

    qutip.plot_wigner(psi)
    f = plt.figure(2)
    ax = f.add_subplot(111)
    print(rho)
    ax.pcolormesh(np.real(rho), cmap='bwr')
    ax.set_aspect('equal')
    ax.set_ylabel('m')
    ax.set_xlabel('n')
    ax.set_title('real($\\rho$)')
    np.save('goal_gkp',rho)
    m = ax.collections[0]
    m.set_clim(-np.max(np.abs(np.real(rho))), np.max(np.abs(np.real(rho))))
    plt.colorbar(m)

    plt.show()


if __name__ == '__main__':
    main()
    print("Done.")

