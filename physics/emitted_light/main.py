# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import sys, pathlib
sys.path.append(
    pathlib.Path(__file__).parent.__str__()
)

from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import calculate_W_t_no_markov
from Operators import make_3d_exp, find_modes
from calculate_W_t_no_markov import W_t_no_markov
from Operators import Operators
from distribution_functions import Wigner
from pylab import *
from moviepy.editor import *
from distribution_functions import Atomic_state_on_bloch_sphere
import qutip

INFINITY = 1000
NUM_POINTS = 1000
NUM_TIMES = 10001


# Press the green button in the gutter to run the script.
def plot_wigner(x, p, wigner, f):
    ax2 = f.add_subplot()
    ax2.pcolormesh(x, p, wigner, cmap='bwr')
    m = ax2.collections[0]
    # ax2.set_xlabel('$q$', fontsize=16)
    # ax2.set_ylabel('$p$', fontsize=16)
    # ax2.set_aspect('equal')
    # g = plt.colorbar(m)
    # g.set_label('$W(q,p)$', fontsize=16)
    m.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()


def test_Wigner_from_moments():
    '''

    :return:
    '''
    N = 3

    NUM_MOMENTS = 12
    op = Operators(N)
    Sx, Sy, Sz = op.create_spin_matrices(N)
    w = Wigner(alpha_max=4, N_alpha=200)
    rho = op.fock_light_state(N, N)
    # rho = expm(1j * np.pi / 2 * Sx) @ rho @ expm(-1j * np.pi / 2 * np.conj(Sx.T))
    a, a_dagger = op.create_a_and_a_dagger(N)

    # m, n = np.meshgrid(np.linspace(0, NUM_MOMENTS, NUM_MOMENTS + 1), np.linspace(0, NUM_MOMENTS, NUM_MOMENTS + 1))
    moments = op.operator_moments(NUM_MOMENTS, rho, a, a_dagger)

    x, p, wigner = w.calc_Wigner_from_moments(moments)
    f = plt.figure()
    plot_wigner(x, p, wigner, f)


def make_matrix_movie(x, p, tensor, omega, xmax=2, ymax=2, text_label='$\\frac{\Omega - \Omega_0}{\Gamma}=$ ',
                      color_label='$W(q,p,\omega)$', x_label='$q$', y_label='$p$'):
    for index in range(len(omega)):
        print(str(index) + '/' + str(len(omega)))
        f = plt.figure()
        plt.pcolormesh(x, p, np.real(tensor[:, :, index]), cmap='bwr')
        ax = f.axes[0]
        m = ax.collections[0]
        plt.xlabel('$q$', fontsize=16)
        plt.ylabel('$p$', fontsize=16)
        # plt.text(0.1, 1.7, '$\\frac{\omega -\Omega_0}{\Gamma} \;= \;' + str(
        #     format((omega[index] - 1) / 0.03, ".2f")), fontsize=16)
        # g = plt.colorbar(m, shrink=0.5)
        # g.set_label('$W(q,p)$', fontsize=16)
        m.set_clim(-0.15, 0.15)  # np.max(np.abs(tensor[:, :, index])), np.max(np.abs(tensor[:, :, index])))
        plt.tight_layout()
        plt.savefig('Images/omega/' + str(index) + '.png')
        plt.close(f)
    make_video('omega/', length=len(omega))
    indices = [0, 10, 20, 30, 40, 50, 60, 70]


def calc_wigner(NUM_MOMENTS, Sminus_t, op, omega, time, t_0, g_w, atomic_rho, w, omega_0=0):
    '''

    :return:
    '''
    N = np.size(Sminus_t, 0) - 1
    dt = time[1] - time[0]
    matrix_exp_iwt = make_3d_exp(N, time, omega)
    Sminus_cumsum = np.cumsum(Sminus_t * dt * matrix_exp_iwt, axis=2)
    if np.size(g_w) > 1:
        moments = op.operator_moments(NUM_MOMENTS, atomic_rho,
                                      np.exp(-1j * omega * time[t_0]) * np.conj(g_w[omega_0]) * Sminus_cumsum[
                                                                                                :, :,
                                                                                                t_0],
                                      np.exp(1j * omega * time[t_0]) * g_w[omega_0] *
                                      np.conj(Sminus_cumsum[:, :, t_0].T))
    else:
        moments = op.operator_moments(NUM_MOMENTS, atomic_rho,
                                      np.exp(-1j * omega * time[t_0]) * np.conj(g_w) * Sminus_cumsum[:, :,
                                                                                       t_0],
                                      np.exp(1j * omega * time[t_0]) * g_w *
                                      np.conj(Sminus_cumsum[:, :, t_0].T))

    x, p, wigner = w.calc_Wigner_from_moments(moments)
    dx = x[1, 2] - x[1, 1]
    dp = p[2, 1] - p[1, 1]
    return x, p, wigner, Sminus_cumsum


def calc_wigner_for_omega(NUM_MOMENTS, time, op, omega_arr, Sminus_t, g_w, atomic_rho):
    N = np.size(Sminus_t, 0) - 1
    dt = time[1] - time[0]
    t_0 = 10000
    w = Wigner(N_alpha=100, alpha_max=4)

    tensor = np.zeros([w.N_alpha, w.N_alpha, len(omega_arr)])
    photon_number = np.zeros(np.shape(omega_arr))
    for index, omega in enumerate(omega_arr):
        x, p, wigner, Sminus_cumsum = calc_wigner(NUM_MOMENTS, Sminus_t, op, omega, time, t_0, g_w, atomic_rho, w,
                                                  index)
        photon_number[index] = np.abs(g_w[index]) ** 2 * np.trace(
            atomic_rho @ np.conj(Sminus_cumsum[:, :, t_0].T) @ Sminus_cumsum[:, :, t_0])
        tensor[:, :, index] = wigner
        print(str(index) + '/' + str(len(omega_arr)))
    domega = omega_arr[1] - omega_arr[0]
    print('sum of n=' + str(sum(photon_number) * domega))
    return x, p, tensor, photon_number


def make_video(label, length=500):
    img_clips = []
    path_list = []
    # accessing path of each image
    for i in range(length):
        path_list.append(os.path.join('Images/' + label + str(i) + '.png'))
    # creating slide for each image
    for img_path in path_list:
        slide = ImageClip(img_path, duration=1 / 10)
        img_clips.append(slide)
    # concatenating slides
    video_slides = concatenate_videoclips(img_clips, method='compose')
    # exporting final video
    video_slides.write_videofile(label[:-1] + ".mp4", fps=24)


def plot_wigner_videos(atomic_wigner, time_s, rho_atoms, rho_light, rho_eta):
    w = Wigner(N_alpha=400, alpha_max=5)
    rho_atoms = np.rot90(np.rot90(rho_atoms).T, k=3)
    N = np.size(rho_light, 1) - 1
    print(N)
    M = np.size(rho_atoms, 0) - 1
    op = Operators(N)
    DELTA = 100

    for index in range(0, len(time_s), DELTA):
        print(str(index) + '/' + str(len(time_s)))
        fig = atomic_wigner.Wigner_BlochSphere(500, np.size(rho_atoms, 1) - 1, [], rho_atoms[index], 'rho')

        ax2 = fig.add_subplot(312)
        x = np.linspace(-4.5, 4.5, 400)
        wigner = qutip.wigner(qutip.Qobj(rho_light[index]), x, x)
        # divnorm = matplotlib.colors.TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=1)

        ax2.pcolormesh(x, x, wigner / np.max(np.abs(wigner)), cmap='bwr')
        m = ax2.collections[0]
        ax2.set_aspect('equal')

        m.set_clim(-1, 1)
        ax2.set_aspect('equal')
        plt.xticks([], fontsize=30)
        plt.yticks([], fontsize=30)
        ax3 = fig.add_subplot(313)

        wigner = qutip.wigner(qutip.Qobj(rho_eta[index]), x, x)
        ax3.pcolormesh(x, x, wigner / np.max(np.abs(wigner)), cmap='bwr')
        plt.xticks([], fontsize=30)
        plt.yticks([], fontsize=30)
        m = ax3.collections[0]
        ax3.set_aspect('equal')
        plt.colorbar(m)
        m.set_clim(-1, 1)
        ax3.set_aspect('equal')

        plt.savefig('Images/time/' + str(index // DELTA) + '.png')
        plt.close(fig)
    make_video('time/', length=len(time_s) // DELTA)
    # indices = [0,10,20,30,40,50,60,70]
    # fig = plt.figure()
    # for i in range(len(indices)):
    #     atomic_wigner.Wigner_BlochSphere(100, np.size(rho_atoms, 1) - 1, [], rho_atoms[indices[i]], 'rho',ax[0,i])
    #     x = np.linspace(-w.alpha_max, w.alpha_max, w.N_alpha)
    #     wigner = qutip.wigner(qutip.Qobj(rho_light[indeces[i]]), x, x)
    #     ax1.pcolormesh(x, x, wigner, cmap='bwr')
    #     m = ax[1,i].collections[0]
    #     ax[1,i].set_aspect('equal')
    #
    #     m.set_clim(-0.15, 0.15)  # np.max(np.abs(wigner)), np.max(np.abs(wigner)))
    #     ax[1,i].set_aspect('equal')
    #     plt.xticks([], fontsize=30)
    #     plt.yticks([], fontsize=30)
    #
    #     wigner = qutip.wigner(qutip.Qobj(rho_eta[indeces[i]]), x, x)
    #     ax[2,i].pcolormesh(x, x, wigner, cmap='bwr')
    #     m = ax3.collections[0]
    #     ax[2,i].set_aspect('equal')
    #     m.set_clim(-0.4, 0.4)
    #     ax[2,i].set_aspect('equal')
    #     plt.xticks([], fontsize=30)
    #     plt.yticks([], fontsize=30)
    # plt.show()


def plot_Sz(rho_t, Sz_t, time, deriv=False):
    T = 1000
    expectation_h = [np.trace(rho_t[:, :, 0] @ Sz_t[:, :, i]) for i in range(len(time))]
    expectation_s = [np.trace(rho_t[:, :, i] @ Sz_t[:, :, 0]) for i in
                     range(len(time))]
    # moments_h = [[np.trace(rho_t[:, :, 0] @ np.linalg.matrix_power(np.conj(Splus_t[:, :, T]).T, n) @
    #                        np.linalg.matrix_power(Splus_t[:, :, T], m)) for m in range(5)] for n in range(5)]
    # moments_s = [[np.trace(rho_t[:, :, T] @ np.linalg.matrix_power(np.conj(Splus_t[:, :, 0]).T, n) @
    #                        np.linalg.matrix_power(Splus_t[:, :, 0], m)) for m in range(5)] for n in range(5)]
    # f, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.imshow(np.abs(moments_h))
    # ax1.set_title('Heisenberg')
    # ax2.imshow(np.abs(moments_s))
    # ax2.set_title('Schrodinger')
    # f3 = plt.figure()
    dt = (time[2] - time[1]) * W_t_no_markov.GAMMA
    if deriv:
        # line, = plt.plot(time[:-1] * superradiance_no_cavity.GAMMA, -np.diff(expectation_h) / dt, linewidth=3)
        # plt.ylabel('$I(t)\;(\hbar \Omega_0 \Gamma)$', fontsize=16)
        t = np.linspace(-1, 0, 1000)
        # plt.xlim([0, 3])
        return time[:-1] * W_t_no_markov.GAMMA, -np.diff(expectation_h) / dt

    else:
        line, = plt.plot(time[:-1] * W_t_no_markov.GAMMA, expectation_h, linewidth=3)

        line.set_label('Heisenberg')
        line2, = plt.plot(time * W_t_no_markov.GAMMA, expectation_s, linewidth=3, linestyle='--')
        line2.set_label('Schrodinger')
        plt.ylabel('$<S_z>(t)$', fontsize=16)
        line3, = plt.plot(time * W_t_no_markov.GAMMA, -0.5 + np.exp(-W_t_no_markov.GAMMA * time),
                          linewidth=1, linestyle='-', color='black')
        line3.set_label('Analytic')
    plt.xlabel('$time\; (\Gamma^{-1})$', fontsize=16)


def plot_intensity(Splus_cumsum, Sminus_t, k, g_w, c, atomic_rho, op, N):
    k = np.linspace(-INFINITY, INFINITY, NUM_POINTS, dtype=complex)
    I_x = np.zeros([len(k), 100])
    jump = 1000
    a_k = op.find_a_k(Sminus_t, k, time, 10000, g_w, c)
    mat = np.zeros([N + 1, N + 1])
    mat[1, 1] = 1
    print(a_k @ mat)
    for t_0 in range(0, 20000, jump):
        E_xplus, E_xminus, x = op.find_a_x(a_k, k)
        x, I_x[:, t_0 // jump] = op.find_intensity(E_xminus, atomic_rho, x)
        plt.plot(x, I_x[:, t_0 // jump], linewidth=2)
        plt.xlabel('x', fontsize=14)
        plt.ylabel('$<E^{\dagger}(x)E(x)>$', fontsize=14)
        plt.ylim([-0.02, 1.1])
        plt.savefig('Images/intensity/' + str(t_0 // jump) + '.png')
        plt.clf()
    plt.show()

    plt.legend(['Simulation', '$-1 + 2e^{-\\Gamma t}$ fit'], fontsize=16)
    plt.legend(fontsize=14, loc='right')
    plt.xlim([0, 2])
    plt.show()


def get_spectrum(a_f_t, rho_init_f, time, omega, N, M):
    '''

    :param a_f_t:
    :param rho_init_f:
    :return:
    '''
    op = Operators(M)
    spectrum = np.zeros(np.shape(omega))
    dt = time[2] - time[1]
    for i, omega_0 in enumerate(omega):
        print(omega_0)
        matrix_exp_iwt = make_3d_exp((N + 1) * (M + 1) - 1, time, omega_0)
        a_omega = np.sum(a_f_t * matrix_exp_iwt * dt, axis=2)
        a_dagger_omega = np.conj(a_omega.T)
        spectrum[i] = np.trace(rho_init_f @ a_dagger_omega @ a_omega)
        print(spectrum[i])
    return spectrum


def plot_spectrum(omega, N, photon_number, domega):
    f2 = plt.figure()
    plt.plot(omega, N * photon_number / (sum(photon_number * omega) * domega), linewidth=2)
    plt.xlabel('$\Omega$', fontsize=16)
    plt.ylabel('$\\frac{dn}{d\Omega}$', fontsize=16)
    GAMMA = 0.015
    y = omega / (np.pi * GAMMA) * (GAMMA ** 2 / ((omega - 1) ** 2 + GAMMA ** 2))
    plt.plot(omega, y / sum(y * domega), linewidth=2,
             linestyle='dashed')
    plt.legend(['Simulation', 'Analytic fit'], fontsize=16)
    plt.show()


def plot_Pn(rho):
    m = np.linspace(0, np.size(rho, 1) - 1, np.size(rho, 1))
    plt.plot(m, [rho[n, n] for n in range(np.size(rho, 0))], linewidth=2)
    plt.xlabel('$n$', fontsize=16)
    plt.ylabel('$p_n$', fontsize=16)


def single_cycle(W_t, kappa, kappa_s, atomic_rho, light_rho, M, N, state):
    W_t.kappa = kappa
    W_t.kappa_s = kappa_s

    rho_f_initial = qutip.tensor(atomic_rho, light_rho)
    times = np.linspace(0, 20, 200)
    dt = times[1] - times[0]
    colors = ['red', 'green', 'black', 'blue', 'orange']

    # W_t.kappa = kappa
    time_s, rho_f_t, negativity, mode, number = W_t.solve_cavity_sr(rho_f_initial, M, N, times)
    rho_f_initial = qutip.tensor(atomic_rho, light_rho, light_rho)
    time_s, rho_f_t = W_t.get_light_outside(rho_f_initial, mode, M, N, times)
    rho_eta = np.array([rho_f_t[i].ptrace(2) for i in range(len(times))])
    return qutip.fidelity(qutip.Qobj(rho_eta[-1]), state)


import multiprocessing as mp


def worker_function(kappa, q, W_t, kappa_s_vec, atomic_rho, light_rho, M, N, state):
    """
    do some work, put results in queue
    """
    arr = np.zeros(len(kappa_s_vec))
    for j, kappa_s in enumerate(kappa_s_vec):
        arr[j] = single_cycle(W_t, kappa, kappa_s, atomic_rho, light_rho, M, N, state)
        print('$\\kappa_cat$= ' + str(kappa))
        print('$\\gamma_cat$= ' + str(kappa_s))

    res = (kappa, arr)
    print(res)
    q.put(res)


def listener(q):
    """
    continue to listen for messages on the queue and writes to file when receive one
    if it receives a '#done#' message it will exit
    """
    with open('output.txt', 'a') as f:
        while True:
            m = q.get()
            if m == '#done#':
                break
            f.write(str(m) + '\n')
            f.flush()


def make_maps(kappa_s_vec, kappa_vec, atomic_rho, light_rho, N, M, W_t):
    '''

    :return:
    '''
    fidelity_state = np.zeros([len(kappa_vec), len(kappa_s_vec)])
    j = 0

    manager = mp.Manager()
    q = manager.Queue()
    file_pool = mp.Pool(1)
    file_pool.apply_async(listener, (q,))

    pool = mp.Pool(16)
    jobs = []
    state = light_cat(M, 1.3j)
    state = qutip.basis(M, M - 1)
    for i, kappa in enumerate(kappa_vec):
        job = pool.apply_async(worker_function, (kappa, q, W_t, kappa_s_vec, atomic_rho, light_rho, M, N, state))
        jobs.append(job)
    # for job in jobs:
    #     job.get()
    q.put('#done#')  # all workers are done, we close the output file
    pool.close()
    pool.join()


def atomic_cat(N, alpha):
    psi = qutip.basis(N, 0)
    Sx = qutip.jmat((N - 1) / 2, 'x')
    Sy = qutip.jmat((N - 1) / 2, 'y')
    # to_exp = 1j * Sx * alpha
    # to_exp2 = 1j * Sx * alpha * np.cos(2 * np.pi / 3) + 1j * Sy * alpha * np.sin(2 * np.pi / 3)
    # to_exp3 = 1j * Sx * alpha * np.cos(4 * np.pi / 3) + 1j * Sy * alpha * np.sin(4 * np.pi / 3)
    to_exp = 1j * Sx * alpha
    to_exp2 = -1j * Sx * alpha
    # psi = ((to_exp.expm() + to_exp2.expm() + to_exp3.expm()) * psi)
    psi = ((to_exp.expm() + to_exp2.expm()) * psi)

    return psi / psi.norm()


def light_cat(N, alpha):
    psi = qutip.basis(N, 0)
    Sx = qutip.jmat((N - 1) / 2, 'x')
    Sy = qutip.jmat((N - 1) / 2, 'y')
    to_exp = 1j * Sx * alpha
    to_exp2 = 1j * Sx * alpha * np.cos(2 * np.pi / 3) + 1j * Sy * alpha * np.sin(2 * np.pi / 3)
    to_exp3 = 1j * Sx * alpha * np.cos(4 * np.pi / 3) + 1j * Sy * alpha * np.sin(4 * np.pi / 3)

    psi = ((to_exp.expm() + to_exp2.expm() + to_exp3.expm()) * psi)
    return psi / psi.norm()


def calc_intensity_and_entanglement_entropy(atomic_rho, light_rho, kappas, M, N, times, W_t, colors, spec=False):
    '''

    :return:
    '''
    W_t.kappa_s = 0
    entanglement_entropy = np.zeros([len(kappas), len(times)])
    intensity = np.zeros([len(kappas), len(times)])
    S = np.zeros([len(kappas), len(times)])

    a = qutip.tensor(qutip.qeye(M), qutip.destroy(N))
    f1 = plt.figure()
    f2 = plt.figure()
    f3 = plt.figure()
    ax1 = f1.add_subplot()
    ax2 = f2.add_subplot()
    ax3 = f3.add_subplot()
    S = np.load('spectrum_cat.npy')
    w = np.load('omega_cat.npy')
    number_op = a.dag()*a
    Sz = qutip.jmat(5,'z') + 5
    print(Sz)
    n = 3.3
    for j, kappa in enumerate(kappas):
        print(j)
        W_t.kappa = kappa
        rho_f_initial = qutip.tensor(atomic_rho, light_rho)
        # time_s, rho_f_t, entropy, mode, number, w, S[j, :] = W_t.solve_cavity_sr(rho_f_initial, M, N, times, spec=True)
        time_s, rho_f_t, entropy, mode, number = W_t.solve_cavity_sr(rho_f_initial, M, N, times, spec=False)

        print(sum(S[j, :]) * (w[2] - w[1]))
        entanglement_entropy[j, :] = np.array([qutip.entropy_vn(rho_f_t[i], 2) for i in range(len(times))])
        intensity[j, :] = np.array([kappa * qutip.expect(number_op, rho_f_t[i]) for i in range(len(times))])
        ax1.plot(times, intensity[j, :], linewidth=2, color=colors[j])
        ax2.plot(times, entanglement_entropy[j, :], linewidth=2, color=colors[j])
        dw = w[2]-w[1]
        # ax3.plot(w, n*S[j, :]/(sum(S[j, :]*dw)), linewidth=2, color=colors[j])

    ax1.set_xlim([0, 10])
    ax2.set_xlim([0, 10])
    ax3.set_xlim([-10, 10])

    # plt.legend(['     ' for j in range(len(kappas))], fontsize=30)
    plt.show()
    np.save('intensity_vs_kappa', intensity)
    np.save('entanglement_entropy', entanglement_entropy)
    np.save('spectrum_cat', S)
    np.save('omega_cat', w)

    np.save('times', times)
    np.save('kappas', kappas)


def squeezed_state(N, alpha):
    psi = qutip.basis(N, N - 1)
    Sx = qutip.jmat((N - 1) / 2, 'x')
    Sy = qutip.jmat((N - 1) / 2, 'y')
    to_exp = 1j * Sx * Sx * alpha
    psi = to_exp.expm() * psi
    return psi / psi.norm()


def atom_to_outside_light(rho:np.ndarray)->np.ndarray:
    # Basic parameters:
    N = rho.shape[0]  # Size of atom basis (number of atoms + 1)
    M = N  # Size of light basis
    times = np.linspace(0, 10, 100)  # time vector - in units of g^-1
    
    # Mirrors params:
    xsi = 2         #reabsorbtion efficiency
    kappa = (2 * (M - 1) / xsi) ** 0.5
    kappa_s = 0
    W_t = W_t_no_markov(op, N, M, atomic_rho, light_rho, kappa, kappa_s)
    
    # Init matrices:
    atomic_rho = qutip.Qobj(rho)  # uncomment this line
    light_rho = qutip.ket2dm(qutip.basis(M, 0))  #initial light state
    
    # flip density:
    Sx = qutip.jmat((N - 1) / 2, 'x')
    to_exp = 1j*Sx*np.pi
    atomic_rho = to_exp.expm()*atomic_rho*(to_exp.dag()).expm()     #qutip uses density matrices that are flipped
    rho_f_initial = qutip.tensor(atomic_rho, light_rho)
    
    # Compute exact evolution:
    time_s, rho_f_t, negativity, mode, number = W_t.solve_cavity_sr(rho_f_initial, M, N, times)
    rho_f_initial = qutip.tensor(atomic_rho, light_rho, light_rho)
    time_s, rho_f_t = W_t.get_light_outside(rho_f_initial, mode, M, N, times)

    rho_atoms = np.array([rho_f_t[i].ptrace(0) for i in range(len(times))])
    rho_light = np.array([rho_f_t[i].ptrace(1) for i in range(len(times))])      #light inside the cavity as a function of time
    rho_eta   = np.array([rho_f_t[i].ptrace(2) for i in range(len(times))])      #light outside in a traveling pulse as a function of time

def main():
    '''
    '''
    N = 11    #Size of atom basis (number of atoms + 1)
    M = 11    #Size of light basis

    # atomic_rho = qutip.Qobj(nir_rho)  #uncomment this line


    atomic_rho = qutip.ket2dm(atomic_cat(N,1.3j))      #initial atomic state example cat state
    # atomic_rho = qutip.ket2dm(qutip.basis(N,N-1))    #initial atomic state example fock state
    light_rho = qutip.ket2dm(qutip.basis(M, 0))        #initial light state
    
    #     
    Sx = qutip.jmat((N - 1) / 2, 'x')
    to_exp = 1j*Sx*np.pi
    atomic_rho = to_exp.expm()*atomic_rho*(to_exp.dag()).expm()     #qutip uses density matrices that are flipped
    rho_f_initial = qutip.tensor(atomic_rho, light_rho)
    
    
    op = Operators(N)
    times = np.linspace(0, 10, 100)         #time vector - in units of g^-1
    dt = times[1] - times[0]

    xsi = 2         #reabsorbtion efficiency
    kappa = (2 * (M - 1) / xsi) ** 0.5
    kappa_s = 0
    W_t = W_t_no_markov(op, N, M, atomic_rho, light_rho, kappa, kappa_s)
    # W_t.kappa = (2 * (M - 1) / xsi) ** 0.5
    # W_t.kappa_s = 0


    # calc_intensity_and_entanglement_entropy(atomic_rho, light_rho, kappa_vec, M, N, times, W_t, colors)
    # fidelity = make_maps(kappa_s_vec, kappa_vec, atomic_rho, light_rho, N, M, W_t)

    time_s, rho_f_t, negativity, mode, number = W_t.solve_cavity_sr(rho_f_initial, M, N, times)

    rho_f_initial = qutip.tensor(atomic_rho, light_rho, light_rho)
    time_s, rho_f_t = W_t.get_light_outside(rho_f_initial, mode, M, N, times)

    rho_atoms = np.array([rho_f_t[i].ptrace(0) for i in range(len(times))])
    rho_light = np.array([rho_f_t[i].ptrace(1) for i in range(len(times))])      #light inside the cavity as a function of time
    rho_eta   = np.array([rho_f_t[i].ptrace(2) for i in range(len(times))])       #light outside in a traveling pulse as a function of time

    # Save data to file
    np.save('rho_atoms', rho_atoms)
    np.save('rho_light', rho_light)
    np.save('rho_eta', rho_eta)
    qutip.plot_wigner(qutip.Qobj(rho_eta[-1]),cmap='bwr',colorbar=True)
    plt.title('Quantum state of mode')

    plt.show()

    plt.plot(times,np.abs(mode)**2)
    plt.title('Shape of mode')
    plt.xlabel('time $(g^{-1})$',fontsize=14)
    plt.ylabel('Intensity $(g)$')
    plt.show()


def test_unitary():
    prob = np.zeros([198, 1])
    N = 200
    atomic_psi = np.zeros([N + 1, 1])
    atomic_psi[20] = 1
    for N in range(2, 200):
        atomic = Atomic_state_on_bloch_sphere(N)
        op = Operators(N)
        Sx, Sy, Sz = op.create_spin_matrices(N)
        atomic_rho = np.zeros([N + 1, N + 1])
        for n in range(N):
            atomic_rho[n, n] = math.comb(N, n) / 2 ** N
        atomic_rho = scipy.linalg.expm(-1j * pi / 2 * Sx) @ atomic_rho @ scipy.linalg.expm(1j * pi / 2 * np.conj(Sx.T))

        prob[N - 2] = atomic_rho[0, 0] * 2
    fig = plt.figure()
    t = np.linspace(2, 200, 198)
    plt.plot(t, prob, linewidth=2)
    plt.plot(t, t ** -0.5, linewidth=2, linestyle='dashed')
    # plt.legend(['           ', '           '], frameon=False, fontsize=30)
    plt.xlim([0, 200])
    plt.ylim([0, 1])
    plt.xticks([], fontsize=18)
    plt.yticks([], fontsize=18)

    plt.show()
    atomic_psi = np.zeros([N + 1])
    atomic_psi[1] = 1
    atomic_psi_new = scipy.linalg.expm(1j * pi / 2 * Sx) @ atomic_psi.T
    atomic_psi_new[np.abs(atomic_psi_new) < 10 ** -3] = 0
    # plt.show()


def solve_sr_equations(op, N, atomic_rho, omega, delta_t, gamma, num_times:int):
    sr_solver = Superradiance_no_cavity(op, N, atomic_rho, delta_t, gamma, num_times)

    time, Sx_t, Sy_t, Sz_t, rho_t = sr_solver.solve_ode()
    dt = time[1] - time[0]
    matrix_exp_iwt = make_3d_exp(N, time, omega)
    Splus_cumsum = np.cumsum((Sx_t + 1j * Sy_t) * dt * np.conj(matrix_exp_iwt), axis=2)
    return Splus_cumsum, Sx_t, Sy_t, Sz_t, rho_t, time

def decay(
    rho: np.matrix,
    delta_t : float,
    gamma : float = 1.0,
    num_time_steps : int = NUM_TIMES,
)->np.matrix:
    # Constants:
    omega1 = 1
    # Fill missing inputs:
    num_momments = rho.shape[0]
    num_atoms = num_momments-1
    
    # init helper objects:
    atomoc = Atomic_state_on_bloch_sphere(num_atoms)
    op = Operators(num_atoms)
    
    # Solve:
    _, _, _, _, rho_t, time = solve_sr_equations(
        op, 
        num_atoms, 
        rho, 
        omega1, 
        gamma=gamma, 
        delta_t=delta_t, 
        num_times=num_time_steps
    )

    return rho_t

def fid_under_v(fid, v, kappa, kappa_s):
    kappa_under = []
    print(v)
    for i in range(len(kappa)):
        j = where(fid[i, :] < v)[0]
        if not np.any(j):
            j = np.size(kappa_s) - 1
        else:
            j = j[0]
        print(j)
        kappa_under.append(kappa_s[j])
    return kappa_under


if __name__ == '__main__':
    import sys, pathlib
    sys.path.append(
        
    )
    atom_to_outside_light()


