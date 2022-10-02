import numpy as np
from scipy.integrate import solve_ivp
import scipy

try:
    from light_wigner.Operators import Operators
except ImportError:
    from Operators import Operators

OMEGA_0 = 1
OMEGA_J = 0

def find_dSi_dt(t, y, N, gamma):
    '''

    :return:
    '''
    if len(y) == 1:
        y = y[0]
    Sx = np.reshape(y[0:N ** 2], (N, N))
    Sy = np.reshape(y[N ** 2:2 * N ** 2], (N, N))
    Sz = np.reshape(y[2 * N ** 2:3 * N ** 2], (N, N))
    rho = np.reshape(y[3 * N ** 2:4 * N ** 2], (N, N))

    op = Operators(N)
    Sx0, Sy0, Sz0 = op.create_spin_matrices(N - 1)

    Splus0 = Sx0 + 1j * Sy0
    Sminus0 = Sx0.T - 1j * np.conj(Sy0.T)

    # dSx_dt = 1j*OMEGA_0/2 * (Sz@Sx - Sx@Sz)- 2 * OMEGA_J / N * (Sz @ Sy + Sy @ Sz) + \
    #          GAMMA * (S_plus @ Sx @ S_minus - 1 / 2 * (Sx @ S_plus @ S_minus + S_plus @ S_minus @ Sx))
    # dSy_dt = 1j*OMEGA_0/2 * (Sz@Sy - Sy@Sz) - 2 * OMEGA_J / N * (Sz @ Sx + Sx @ Sz) + \
    #          GAMMA * (S_plus @ Sy @ S_minus - 1 / 2 * (Sy @ S_plus @ S_minus + S_plus @ S_minus @ Sy))
    # dSz_dt = GAMMA * (S_plus @ Sz @ S_minus - 1 / 2 * (Sz @ S_plus @ S_minus + S_plus @ S_minus @ Sz))
    dSx_dt = 1j * OMEGA_0 * (Sz0 @ Sx - Sx @ Sz0) - 2 * OMEGA_J / N * (Sz @ Sy + Sy @ Sz) + \
             gamma * (Splus0 @ Sx @ Sminus0 - 1 / 2 * (Sx @ Splus0 @ Sminus0 + Splus0 @ Sminus0 @ Sx))
    dSy_dt = 1j * OMEGA_0 * (Sz0 @ Sy - Sy @ Sz0) - 2 * OMEGA_J / N * (Sz @ Sx + Sx @ Sz) + \
             gamma * (Splus0 @ Sy @ Sminus0 - 1 / 2 * (Sy @ Splus0 @ Sminus0 + Splus0 @ Sminus0 @ Sy))
    dSz_dt = gamma * (Splus0 @ Sz @ Sminus0 - 1 / 2 * (Sz @ Splus0 @ Sminus0 + Splus0 @ Sminus0 @ Sz))
    # S_minus = np.conj(S_plus.T)
    #
    # dSplus_dt = 1j * OMEGA_0 * S_plus + GAMMA * S_plus @ Sz
    # dSz_dt = -GAMMA * (S_plus @ S_minus)
    drho_dt = -1j * OMEGA_0 * (Sz0 @ rho - rho @ Sz0) + gamma * (
            Sminus0 @ rho @ Splus0 - 1 / 2 * (rho @ Splus0 @ Sminus0 + Splus0 @ Sminus0 @ rho))
    dy_dt = np.concatenate(
        [np.reshape(dSx_dt, (N ** 2, 1)), np.reshape(dSy_dt, (N ** 2, 1)), np.reshape(dSz_dt, (N ** 2, 1)),
         np.reshape(drho_dt, (N ** 2, 1))]).T
    return dy_dt


class Superradiance_no_cavity:
    def __init__(self, op, N, rho, delta_t:float, gamma:float, num_times:int=10001):
        self.Sx, self.Sy, self.Sz = op.create_spin_matrices(N)
        self.N = N + 1
        self.rho = rho
        self.t = np.linspace(0, delta_t, num_times)
        self.gamma = gamma

    def solve_ode(self):
        N = self.N
        op = Operators(N - 1)
        Sx0, Sy0, Sz0 = op.create_spin_matrices(N - 1)

        y0 = np.concatenate((np.reshape((Sx0), (1, self.N ** 2)), np.reshape(Sy0, (1, self.N ** 2)),
                             np.reshape(Sz0, (1, self.N ** 2)), np.reshape(self.rho, (1, self.N ** 2))), axis=1)[0]
        sol = solve_ivp(find_dSi_dt, (self.t[0], self.t[-1]), y0, args=[self.N, self.gamma], t_eval=self.t, max_step=0.1)
        Sx_t = np.reshape(sol.y[0:N ** 2, :], (N, N, np.size(sol.y) // N ** 2 // 4))
        Sy_t = np.reshape(sol.y[N ** 2:2 * N ** 2, :], (N, N, np.size(sol.y) // N ** 2 // 4))
        Sz_t = np.reshape(sol.y[2 * N ** 2:3 * N ** 2, :], (N, N, np.size(sol.y) // N ** 2 // 4))
        rho_t = np.reshape(sol.y[3 * N ** 2:4 * N ** 2, :], (N, N, np.size(sol.y) // N ** 2 // 4))
        return sol.t, Sx_t, Sy_t, Sz_t, rho_t
