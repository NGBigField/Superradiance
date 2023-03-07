import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from scipy.integrate import solve_ivp
from Operators import Operators
import matplotlib
import scipy

OMEGA_0 = 0
OMEGA_1 = 0
OMEGA_J = 0
g = 1j
kappa = 1 / np.sqrt(10)
kappa_s = 0.


def get_two_time_corralation_funciton(rho_init, light_dim, atom_dim, kappa):
    '''

    :param rho_init:
    :return:
    '''
    rho_init = Qobj(rho_init)
    S_plus = jmat((atom_dim - 1) / 2, '+')
    S_minus = jmat((atom_dim - 1) / 2, '-')
    S_z = jmat((atom_dim - 1) / 2, 'z')
    times = np.linspace(0.0, 10.0, 200)
    a = tensor(qeye(atom_dim), destroy(light_dim))
    sp = tensor(S_plus, qeye(light_dim))
    sm = tensor(S_minus, qeye(light_dim))
    sz = tensor(S_z, qeye(light_dim))

    g = 1j
    OMEGA_0 = 0
    H = OMEGA_0 * a.dag() * a + g * sm * a.dag() + np.conj(g) * sp * a
    opts = Options()
    opts.store_states = True
    cmap = matplotlib.cm.bwr
    result = mesolve(H, rho_init, times,
                     [kappa ** 0.5 * a],
                     [a.dag() * a, sp * sm], options=opts)
    rho_t = result.states[50]


def find_dheisenberg_dt(t, y, S_plus_f0, S_minus_f0, a_dagger_f0, a_f0, Sz_f0, N, M):
    '''

    :return:
    '''

    if len(y) == 1:
        y = y[0]
    a_f = np.reshape(y, ((N + 1) * (M + 1), (N + 1) * (M + 1)))
    H_AF = (g * S_minus_f0 @ a_dagger_f0 + np.conj(g) * S_plus_f0 @ a_f0)

    da_dt = -(OMEGA_1 * 1j * (a_f @ a_dagger_f0 @ a_f0 - a_dagger_f0 @ a_f0 @ a_f) + 1j * OMEGA_0 * (
            a_f @ Sz_f0 - Sz_f0 @ a_f) \
              + 1j * (a_f @ H_AF - H_AF @ a_f)) + kappa * (
                    2 * a_dagger_f0 @ a_f @ a_f0 - a_dagger_f0 @ a_f0 @ a_f - a_f @ a_dagger_f0 @ a_f0)

    # plt.show()
    dy_dt = np.reshape(da_dt, ((M + 1) ** 2 * (N + 1) ** 2, 1)).T[0]
    return dy_dt


def find_drho_dt(t, y, S_plus_f, S_minus_f, a_dagger_f, a_f, Sz_f, N, M):
    '''

    :return:
    '''
    if len(y) == 1:
        y = y[0]

    rho = np.reshape(y, ((N + 1) * (M + 1), (N + 1) * (M + 1)))

    H_AF = (g * S_minus_f @ a_dagger_f + np.conj(g) * S_plus_f @ a_f)
    drho_dt = 1j * OMEGA_1 * (rho @ a_dagger_f @ a_f - a_dagger_f @ a_f @ rho) + 1j * OMEGA_0 * (
            rho @ Sz_f - Sz_f @ rho) \
              + 1j * (rho @ H_AF - H_AF @ rho) + kappa * (
                      2 * a_f @ rho @ a_dagger_f - a_dagger_f @ a_f @ rho - rho @ a_dagger_f @ a_f)

    dy_dt = np.reshape(drho_dt, ((N + 1) ** 2 * (M + 1) ** 2, 1)).T[0]
    # print(dy_dt[dy_dt>0])
    return dy_dt


def H1_coeff(t, args):
    mode = args['mode']
    times = args['times']
    index = np.where(np.abs(times - t) == np.min(np.abs(t - times)))[0]
    # plt.plot(mode)
    # plt.show()
    return mode[index]


def H2_coeff(t, args):
    mode = args['mode']
    times = args['times']

    index = np.where(np.abs(times - t) == np.min(np.abs(t - times)))[0]
    # print(mode[index])
    return np.conj(mode[index])


def calc_spectrum(times, corr):
    N = times.shape[0]
    dt = times[1] - times[0]
    # calculate the frequencies for the components in F
    w = 2 * np.pi * np.linspace(-1 / (2 * dt), 1 / (2 * dt), N)
    S = np.zeros(np.shape(times), dtype=complex)
    I = np.sum([corr[i, i] * dt for i in range(len(corr))])
    for i in range(len(S)):
        S[i] =1/2/np.pi* np.sum(
            [np.sum([corr[j, k] * np.exp(1j * w[i] * (times[k] - times[j])) for j in range(len(times))])*dt**2
             for k in range(len(times))])
        print(i)

    return w, np.abs(S)


class W_t_no_markov:
    def __init__(self, op, N, M, rho_atoms, rho_light, kappa, kappa_s):
        self.Sx, self.Sy, self.Sz = op.create_spin_matrices(N)
        self.kappa = kappa
        self.kappa_s = kappa_s

        self.N = N
        self.M = M
        self.rho_atoms = rho_atoms
        self.rho_light = rho_light
        self.t = np.linspace(0, 3, 201)

    def solve_cavity_sr(self, rho_init, light_dim, atom_dim, times, spec=False):
        rho_init = Qobj(rho_init)
        kappa = self.kappa
        kappa_s = self.kappa_s

        S_plus = jmat((atom_dim - 1) / 2, '+')
        S_minus = jmat((atom_dim - 1) / 2, '-')
        S_z = jmat((atom_dim - 1) / 2, 'z')
        a = tensor(qeye(atom_dim), destroy(light_dim))
        sp = tensor(S_plus, qeye(light_dim))
        sm = tensor(S_minus, qeye(light_dim))
        sz = tensor(S_z, qeye(light_dim))

        g = 1j
        OMEGA_0 = 0
        H = OMEGA_0 * a.dag() * a + OMEGA_0 * sz + g * sm * a.dag() + np.conj(g) * sp * a
        opts = Options()
        opts.store_states = True
        cmap = matplotlib.cm.bwr
        dt = times[2] - times[1]

        result = mesolve(H, rho_init, times,
                         [kappa ** 0.5 * a + kappa_s ** 0.5 * sm],
                         [a.dag() * a, sp * sm], options=opts)
        rho_t = result.states
        negativity_state = 0  # np.array([negativity(rho_t[i], 0) for i in range(len(times))])

        # negativity_state = np.array([negativity(rho_t[i], 0) for i in range(len(times))])
        corr = correlation_2op_2t(H, rho_init, times, times, [kappa ** 0.5 * a], a.dag(), a)

        corr_cor = np.zeros_like(corr)
        for i in range(np.size(corr, 0)):
            corr_cor[i, i + 1:] = corr[i, :-i - 1]
        corr_cor = corr_cor + np.conj(corr_cor.T)
        for i in range(np.size(corr, 0)):
            corr_cor[i, i] = corr[i, 0]
        w, v = np.linalg.eig(kappa * corr_cor * dt)
        # print(v[:, np.abs(w) == max(np.abs(w))])

        vec = v[:, w == max(w)]
        #
        # # print(sum(abs(v) ** 2))
        # f1 = plt.figure()
        # ax2 = f1.add_subplot()
        #
        # ax2.pcolormesh(times, times, np.real(corr_cor), cmap='bwr')
        # m = ax2.collections[0]
        # m.set_clim(-np.max(np.real(corr_cor)), np.max(np.real(corr_cor)))
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # g = plt.colorbar(m)
        # g.ax.tick_params(labelsize=16)
        # f3 = plt.figure()
        # ax2 = f3.add_subplot()
        # for big_w in np.flip(np.sort(w)[-3:]):
        #     plt.plot(times, v[:, w == big_w], linewidth=2)
        # plt.legend(['      ', '      ', '      '], fontsize=16)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # f2 = plt.figure()
        # ax2 = f2.add_subplot()
        # ax2.bar(np.linspace(1, len(w), len(w)), np.abs(w))
        # ax2.set_xlim([0, 10])
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.show()
        if spec:
            w, S = calc_spectrum(times, kappa*corr_cor)
            return times, rho_t, negativity_state, vec, kappa * result.expect[0], w, S
        else:
            return times, rho_t, negativity_state, vec, kappa * result.expect[0]

    def get_light_outside(self, rho_init, mode, light_dim, atom_dim, times):
        '''

        :param rho_init:
        :param light_dim:
        :param atom_dim:
        :param times:
        :return:
        '''
        kappa = self.kappa
        kappa_s = self.kappa_s
        S_plus = jmat((atom_dim - 1) / 2, '+')
        S_minus = jmat((atom_dim - 1) / 2, '-')
        dt = times[2] - times[1]

        S_z = jmat((atom_dim - 1) / 2, 'z')
        a_c = tensor(qeye(atom_dim), destroy(light_dim), qeye(light_dim))
        a_eta = tensor(qeye(atom_dim), qeye(light_dim), destroy(light_dim))
        sp = tensor(S_plus, qeye(light_dim), qeye(light_dim))
        sm = tensor(S_minus, qeye(light_dim), qeye(light_dim))
        sz = tensor(S_z, qeye(light_dim), qeye(light_dim))

        g = 1j
        OMEGA_0 = 0
        H0 = OMEGA_0 * a_c.dag() * a_c + OMEGA_0 * sz + g * sm * a_c.dag() + np.conj(g) * sp * a_c
        H1 = 1j / 2 * a_c.dag() * a_eta * kappa ** 0.5
        H2 = -1j / 2 * a_c * a_eta.dag() * kappa ** 0.5
        mode_new = np.zeros_like(mode)
        for i in range(len(mode)):
            # print(mode[i])
            S = np.sum(np.abs(mode[:i + 1]) ** 2 * dt) ** 0.5
            if S > 0:
                mode_new[i] = -np.conj(mode[i]) / S
            # print(S)
        mode = mode_new
        opts = Options()
        opts.store_states = True
        cmap = matplotlib.cm.bwr
        args = {'mode': mode, 'times': times}
        result = mesolve([H0, [H1, H1_coeff], [H2, H2_coeff]], rho_init, times,
                         [[kappa ** 0.5 * a_c + kappa_s ** 0.5 * sm, [a_eta, H2_coeff]]],
                         [a_eta.dag() * a_eta], args=args, options=opts)
        rho_t = result.states
        negativity_state = 0
        # plt.plot(result.expect[0])
        # plt.show()
        return times, rho_t

    #
    # def solve_cavity_sr(self, rho_initial_atoms, rho_initial_light):
    #
    #     N = np.size(rho_initial_light, 1) - 1
    #     M = np.size(rho_initial_atoms, 1) - 1
    #     print(M)
    #     I_light = np.eye(N + 1, dtype=complex)
    #     I_atoms = np.eye(M + 1, dtype=complex)
    #     op = Operators(M)
    #     rho_tot = np.kron(rho_initial_atoms, rho_initial_light)
    #     a, a_dagger f= op.create_a_and_a_dagger(N)
    #     Sx, Sy, Sz = op.create_spin_matrices(M)
    #     S_plus_f = np.kron(Sx + 1j * Sy, I_light)
    #     S_minus_f = np.conj(S_plus_f.T)
    #     Sz_f = np.kron(Sz, I_light)
    #     a_f = np.kron(I_atoms, a)
    #     a_dagger_f = np.kron(I_atoms, a_dagger)
    #     op = Operators(N)
    #     y0 = np.reshape(rho_tot, (1, (N + 1) ** 2 * (M + 1) ** 2))[0]
    #     sol = solve_ivp(find_drho_dt, (self.t[0], self.t[-1]), y0,
    #                     args=[S_plus_f, S_minus_f, a_dagger_f, a_f, Sz_f, N, M],
    #                     t_eval=self.t, max_step=0.1)
    #     rho_t = np.reshape(sol.y,
    #                        ((N + 1) * (M + 1), (N + 1) * (M + 1), np.size(sol.y) // (N + 1) ** 2 // (M + 1) ** 2))
    #     rho_light_t = np.zeros([N + 1, N + 1, len(sol.t)], dtype=complex)
    #     rho_atoms_t = np.zeros([M + 1, M + 1, len(sol.t)], dtype=complex)
    #     negativity = np.zeros(len(sol.t))
    #     for k in range(len(sol.t)):
    #         rho_light_t[:, :, k] = np.trace(np.reshape(rho_t[:, :, k], (M + 1, N + 1, M + 1, N + 1)), axis1=0, axis2=2)
    #         rho_atoms_t[:, :, k] = np.trace(np.reshape(rho_t[:, :, k], (M + 1, N + 1, M + 1, N + 1)), axis1=1, axis2=3)
    #         negativity[k] = self.negativity(rho_t[:, :, k], N + 1, N + 1)
    #     return sol.t, rho_atoms_t, rho_light_t, negativity

    def negativity(self, rho, M, N):
        rho_par_trans = self.partial_transpose(rho, M, N)
        l = qutip.Qobj(rho_par_trans).eigenenergies()
        return ((abs(l) - l) / 2).sum()

    def partial_transpose(self, rho, M, N):
        '''

        :param rho:
        :return:
        '''
        k = np.zeros_like(rho)
        for i in range(M):
            for j in range(N):
                k[i * M:(i + 1) * M, j * N: (j + 1) * N] = np.conj(rho[i * M:(i + 1) * M, j * N:(j + 1) * N]).T
        return k

    def solve_heisenberg(self):
        N = self.N
        M = self.M
        I_light = np.eye(N + 1, dtype=complex)
        I_atoms = np.eye(M + 1, dtype=complex)
        op = Operators(N)

        a, a_dagger = op.create_a_and_a_dagger(N)
        Sx, Sy, Sz = op.create_spin_matrices(M)
        Sz_f = np.kron(Sz, I_light)

        S_plus_f = np.kron(Sx + 1j * Sy, I_light)
        S_minus_f = np.conj(S_plus_f.T)
        a_f = np.kron(I_atoms, a)
        a_dagger_f = np.conj(a_f.T)

        y0 = np.reshape(a_f, (1, (N + 1) ** 2 * (M + 1) ** 2))[0]
        sol = solve_ivp(find_dheisenberg_dt, (self.t[0], self.t[-1]), y0,
                        args=[S_plus_f, S_minus_f, a_dagger_f, a_f, Sz_f, N, M],
                        t_eval=self.t)
        a_f_t = np.reshape(sol.y[0:(N + 1) ** 2 * (M + 1) ** 2],
                           ((N + 1) * (M + 1), (N + 1) * (M + 1), np.size(sol.y) // (N + 1) ** 2 // (M + 1) ** 2))
        return sol.t, a_f_t
