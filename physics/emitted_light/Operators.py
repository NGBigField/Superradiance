import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from scipy.special import factorial
from decimal import *
import scipy.fftpack
# from qutip import partial_transpose


def make_3d_exp(N, time, omega):
    d3_exp = np.ones([N + 1, N + 1, len(time)], dtype=complex)
    for i in range(len(time)):
        d3_exp[:, :, i] = d3_exp[:, :, i] * np.exp(1j * omega * time[i])
    return d3_exp


# import QuTiP
def find_correlation_matrix(a_t, rho_initial, kappa):
    Nt = np.size(a_t, 2)
    indices = np.array(range(Nt), dtype=int)
    print('total photon number:')
    print(2 * kappa * sum([np.trace(np.conj(a_t[:, :, i].T) @ a_t[:, :, i] @ rho_initial) for i in range(Nt)]))
    return 2 * kappa * np.array(
        [[np.trace(np.conj(a_t[:, :, i].T) @ a_t[:, :, j] @ rho_initial) for i in range(Nt)] for j in range(Nt)],
        dtype=complex)


def find_modes(a_t, rho_initial, kappa, dt):
    '''

    :param a_t:
    :param rho_initial:
    :param kappa:
    :return:
    '''
    correlation_mat = find_correlation_matrix(a_t * dt ** 0.5, rho_initial, kappa)
    w, v = np.linalg.eig(correlation_mat)
    print(sum(w))
    vec = v[:, w == max(w)]
    # a = np.array([np.trace(np.conj(a_t[:, :, i].T) @ a_t[:, :, i] @ rho_initial) for i in range(len(vec))])
    # plt.plot(np.sqrt(a)/np.linalg.norm(np.sqrt(a)))

    # plt.plot(np.abs(vec),linestyle = 'dashed')
    # vec = v[:, w == np.sort(w)[-2]]
    # f = plt.figure()
    # ax = f.add_subplot(111)
    # ax.bar(np.linspace(1, len(w), len(w)), -np.sort(-w)/sum(w))
    # plt.plot(np.abs(vec),linestyle = 'dashed')
    # plt.legend(['$\sqrt{intensity}$','Most populated mode','Second most populated mode'])
    # plt.xlabel('Mode',fontsize=20)
    # plt.ylabel('Occupation',fontsize=20)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    #
    # print(vec)
    # plt.show()
    print(sum(np.abs(vec) ** 2))
    return vec / dt ** 0.5


class Operators:
    '''
    '''

    def __init__(self, N):
        self.a, self.a_dagger = self.create_a_and_a_dagger(N)

    def operator_moments_b(self, NUM_OF_MOMENTS, rho, a, a_dagger):
        '''

        :param NUM_OF_MOMENTS:
        :param rho:
        :param a:
        :param a_dagger:
        :return:
        '''
        N = np.size(a, 1)
        # norm = np.trace(np.abs(a@a_dagger - a_dagger@a))
        # print(norm)
        # if norm !=0:
        #     a = a/norm*N
        #     a_dagger = a_dagger/norm**(0.5/N)

        a_avg = np.trace(rho @ a)
        moments = np.zeros([NUM_OF_MOMENTS, NUM_OF_MOMENTS], dtype=complex)
        for n in range(0, NUM_OF_MOMENTS):
            for m in range(0, NUM_OF_MOMENTS):
                moments[n, m] = np.trace(
                    rho @ np.linalg.matrix_power(a_dagger - np.eye(N, dtype=complex) * np.conj(a_avg),
                                                 n) @ np.linalg.matrix_power(
                        a - np.eye(N, dtype=complex) * a_avg, m))
        return moments, a_avg

    def find_a_x(self, a_k, k):
        '''

        :param Sminus_t:
        :param k:

        :param t:
        :return:
        '''
        NUM_P = len(k)
        dk = k[1] - k[0]
        N = np.size(a_k, 0)
        x = np.linspace(-1 / 2 / dk, 1 / 2 / dk, NUM_P)
        E_xplus = np.zeros(np.shape(a_k), dtype=complex)
        E_xminus = np.zeros(np.shape(a_k), dtype=complex)

        for i, x_0 in enumerate(x):
            e_ikr = make_3d_exp(N - 1, k, x_0)
            a_k_dagger = self.dagger_3d(a_k)
            E_xplus[:, :, i] = 1j * np.sum(a_k * e_ikr * dk, axis=2)
            E_xminus[:, :, i] = -1j * np.sum(a_k_dagger * np.conj(e_ikr) * dk, axis=2)

        return E_xplus, E_xminus, x

    def find_intensity(self, E_xplus, rho, x):
        '''

        :param a_x:
        :param atomic_rho:
        :param x:
        :return:
        '''
        intensity_x = np.zeros(np.shape(x), dtype=complex)
        for i, x_0 in enumerate(x):
            E_xplus0 = E_xplus[:, :, i]
            intensity_x[i] = np.trace(rho @ E_xplus0 @ np.conj(E_xplus0.T))
        return x, intensity_x

    def find_a_k(self, Sminus_t, k, t, t_0, g_w, c=0.1):
        '''
        :param Sminus_t:
        :param k:
        :param t:
        :return:
        '''
        dt = t[1] - t[0]
        N = np.size(Sminus_t, 0)
        a_k = np.zeros([N, N, len(k)], dtype=complex)
        for i, k_0 in enumerate(k):
            e_iwt = make_3d_exp(N - 1, t, np.abs(k_0) * c)
            Sminus_cumsum = np.cumsum(Sminus_t * e_iwt * dt, axis=2)
            a_k[:, :, i] = np.conj(e_iwt[:, :, t_0]) * Sminus_cumsum[:, :, t_0] * np.conj(g_w)
        return a_k

    def get_a_xsi_t(self, a_t, rho_f_t, a_f0, a_dagger_f0, dt, kappa, omega, vec):
        flux_t = 2 * kappa * np.array(
            [np.trace(a_dagger_f0 @ a_f0 @ rho_f_t[:, :, i]) for i in range(np.size(rho_f_t, 2))])
        N_tot = np.cumsum(flux_t * dt)
        print('Total photons emitted' + str(N_tot[-1]))
        M = np.size(a_t, 1)
        # a_t = np.real(a_t)
        a_cumsum = np.zeros_like(a_t)  # np.cumsum(np.abs(a_t) * dt, axis=2)

        # for i in range(1, np.size(a_t, 2)):

        for i in range(1, np.size(a_t, 2)):
            a_cumsum[:, :, i] = a_cumsum[:, :, i - 1] + np.sqrt(2 * kappa) * a_t[:, :, i] * np.conj(vec[i]) * dt

        return a_cumsum, N_tot

    def operator_moments(self, NUM_OF_MOMENTS, rho, a, a_dagger):
        '''

        :param NUM_OF_MOMENTS:
        :param rho:
        :param a:
        :param a_dagger:
        :return:
        '''
        N = np.size(a, 1)
        moments = np.zeros([NUM_OF_MOMENTS, NUM_OF_MOMENTS], dtype=complex)
        for n in range(0, NUM_OF_MOMENTS):
            for m in range(0, NUM_OF_MOMENTS):
                moments[n, m] = np.trace(
                    rho @ np.linalg.matrix_power(a_dagger, n) @ np.linalg.matrix_power(a, m))
        return moments

    def get_negativity(self, rho, subsys, method='tracenorm'):
        """
        Compute the negativity for a multipartite quantum system described
        by the density matrix rho. The subsys argument is an index that
        indicates which system to compute the negativity for.

        .. note::

            Experimental.
        """
        mask = [idx == subsys for idx, n in enumerate(rho.dims[0])]
        rho_pt = partial_transpose(rho, mask)

        if method == 'tracenorm':
            N = ((rho_pt.dag() * rho_pt).sqrtm().tr().real - 1) / 2.0
        elif method == 'eigenvalues':
            l = rho_pt.eigenenergies()
            N = ((abs(l) - l) / 2).sum()
        else:
            raise ValueError("Unknown method %s" % method)
        return N

    def create_initial_atomic_state(self, N, m):
        '''

        :param N:
        :param m:
        :return:
        '''
        rho = np.zeros([N + 1, N + 1], dtype=complex)
        rho[m, m] = 1
        return rho

    def generate_indices(self, N, k):
        if k == 2:
            return np.array(range(0, N + 1), dtype=int)
        elif k > 3:
            print('Only upto k=3 for now')
            return
        ind = np.concatenate([[(i, j) for i in range(N - j + 1)] for j in range(N + 1)])
        print(ind)
        return ind

    def create_spin_matrices_general(self, N, k):
        '''

        :param N:
        :return:
        '''
        if k == 2:
            return self.create_spin_matrices(N)
        elif k > 3:
            print('Only upto k=3 for now')
            return
        indices = self.generate_indices(N, k)

        P = np.size(indices, 0)

        S_minus_r = np.zeros([P, P], dtype=complex)
        S_minus_l = np.zeros([P, P], dtype=complex)
        for i in range(len(indices)):
            for j in range(len(indices)):
                m_r, m_l = indices[i]
                n_r, n_l = indices[j]
                if n_r == m_r - 1 and n_l == m_l:
                    S_minus_r[i, j] = (m_r * (N - m_r - m_l + 1)) ** 0.5
                elif n_r == m_r and n_l == m_l - 1:
                    S_minus_l[i, j] = (m_l * (N - m_r - m_l + 1)) ** 0.5
        S_plus_r = S_minus_r.T
        S_plus_l = S_minus_l.T
        return S_plus_r, S_minus_r, S_plus_l, S_minus_l

    def create_spin_matrices(self, N):
        '''

        :param N:
        :return:
        '''

        m = np.linspace(1, N, N, dtype=complex)
        S_plus = np.diag((m * (N - m + 1)) ** 0.5, k=-1)
        S_minus = np.diag((m * (N - m + 1)) ** 0.5, k=1)
        Sx = (S_plus + S_minus) / 2
        Sy = (S_plus - S_minus) / 2j
        Sz = np.array(np.diag(np.linspace(-N, N, N + 1)) / 2, dtype=object)
        # print(S_plus@S_minus-S_minus@S_plus-N*(N+2)*np.eye(N+1)+Sz**2-2*Sz)
        return Sx, Sy, Sz

    def create_a_and_a_dagger(self, N):
        '''

        :param N:
        :return:
        '''
        a_vec = np.linspace(1, N, N, dtype=complex) ** 0.5
        a = np.diag(a_vec, k=1)
        a_dagger = a.T
        return a, a_dagger

    def even_cat_state(self, alpha, N):
        '''

        :param alpha:
        :param N:
        :return:
        '''
        m_vec = np.linspace(0, N, N + 1)
        n_vec = np.linspace(0, N, N + 1)
        n, m = np.meshgrid(n_vec, m_vec)
        rho_cat = np.array(1 / (2 * (1 - np.exp(-2 * np.abs(alpha) ** 2))) * alpha ** m \
                           * np.conjugate(alpha) ** n / (factorial(n) * factorial(m)) ** 0.5, dtype=complex)
        for k in range(N):
            for l in range(N):
                if k % 2 != 0 or l % 2 != 0:
                    rho_cat[k, l] = 0
        rho_cat = rho_cat / np.trace(rho_cat)
        return rho_cat

    def coherent_light_state(self, alpha, N):
        '''

        :param alpha:
        :param N:
        :return:
        '''
        m_vec = np.linspace(0, N, N + 1)
        n_vec = np.linspace(0, N, N + 1)
        n, m = np.meshgrid(n_vec, m_vec)
        rho_coherent = np.array(np.exp(-np.abs(alpha) ** 2) * alpha ** m \
                                * np.conjugate(alpha) ** n / (factorial(n) * factorial(m)) ** 0.5, dtype=complex)
        rho_coherent = rho_coherent / (np.trace(rho_coherent))
        return rho_coherent

    def fock_light_state(self, m, N):
        rho_fock = np.zeros([N + 1, N + 1], dtype=complex)
        rho_fock[m, m] = 1
        return rho_fock

    def cat_state(self, ):
        pass

    def fourier_transform_matrix(self, time, Sx_t):
        Sx_w = np.zeros(np.shape(Sx_t), dtype=complex)
        for i in range(np.size(Sx_w, 0)):
            for j in range(np.size(Sx_w, 1)):
                Sx_w[i, j, :] = np.fft.fft(Sx_t[i, j, :])
        return Sx_w

    def dagger_3d(self, a_t):
        a_dagger_t = np.zeros(np.shape(a_t), dtype=complex)
        for i in range(np.size(a_t, 2)):
            a_dagger_t[:, :, i] = np.conj(a_t[:, :, i].T)
        return a_dagger_t
