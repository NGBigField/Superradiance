# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from operator import le
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import os

try:
    import light_wigner.superradiance_no_cavity
    from light_wigner.superradiance_no_cavity import Superradiance_no_cavity
    from light_wigner.Operators import make_3d_exp
    from light_wigner.Operators import Operators
    from light_wigner.distribution_functions import Wigner
    from light_wigner.distribution_functions import Atomic_state_on_bloch_sphere
except ImportError:
    import superradiance_no_cavity
    from   superradiance_no_cavity import Superradiance_no_cavity
    from   Operators import make_3d_exp
    from   Operators import Operators
    from   distribution_functions import Wigner
    from   distribution_functions import Atomic_state_on_bloch_sphere


import matplotlib.animation as animation
from pylab import *
from moviepy.editor import *
from matplotlib import cm, colors


# For type hinting:
from typing import (
    Optional,
)

INFINITY = 1000
NUM_POINTS = 1000
NUM_TIMES = 10001
NUM_VIDEO_FRAMES = 10

# Press the green button in the gutter to run the script.


def plot_wigner(x, p, wigner, f):
    ax2 = f.add_subplot()
    ax2.pcolormesh(x, p, wigner, cmap='bwr')
    m = ax2.collections[0]
    ax2.set_xlabel('$q$', fontsize=16)
    ax2.set_ylabel('$p$', fontsize=16)
    ax2.set_aspect('equal')
    g = plt.colorbar(m, shrink=0.5)
    g.set_label('$W(q,p)$', fontsize=16)
    m.set_clim(-np.max(np.abs(wigner)), np.max(np.abs(wigner)))
    plt.tight_layout()
    plt.show()


def test_Wigner_from_moments():
    '''

    :return:
    '''
    N = 1
    NUM_MOMENTS = 11
    op = Operators(N)
    w = Wigner(alpha_max=4)
    rho = np.zeros([11, 11])
    for i in range(10):
        rho[i, i] = 1 / 11
    a, a_dagger = op.create_a_and_a_dagger(N)

    # m, n = np.meshgrid(np.linspace(0, NUM_MOMENTS, NUM_MOMENTS + 1), np.linspace(0, NUM_MOMENTS, NUM_MOMENTS + 1))
    moments = op.operator_moments(NUM_MOMENTS, rho, a, a_dagger)

    x, p, wigner = w.calc_Wigner_from_moments(moments)
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
        plt.text(0.1, 1.7, '$\\frac{\omega -\Omega_0}{\Gamma} \;= \;' + str(
            format((omega[index] - 1) / 0.03, ".2f")) + '\;\Gamma$', fontsize=16)
        g = plt.colorbar(m, shrink=0.5)
        g.set_label('$W(q,p)$', fontsize=16)
        m.set_clim(-np.max(np.abs(tensor[:, :, index])), np.max(np.abs(tensor[:, :, index])))
        plt.tight_layout()
        plt.savefig('Images/omega/' + str(index) + '.png')
        plt.close(f)
    make_video('omega/', length=len(omega))


def solve_sr_equations(op, N, atomic_rho, omega, delta_t, gamma, num_times:int):
    sr_solver = Superradiance_no_cavity(op, N, atomic_rho, delta_t, gamma, num_times)

    time, Sx_t, Sy_t, Sz_t, rho_t = sr_solver.solve_ode()
    dt = time[1] - time[0]
    matrix_exp_iwt = make_3d_exp(N, time, omega)
    Splus_cumsum = np.cumsum((Sx_t + 1j * Sy_t) * dt * np.conj(matrix_exp_iwt), axis=2)
    return Splus_cumsum, Sx_t, Sy_t, Sz_t, rho_t, time


def calc_wigner(NUM_MOMENTS, Sminus_t, op, omega, time, t_0, g_w, atomic_rho, w, omega_0=0):
    '''

    :return:
    '''
    N = np.size(Sminus_t, 0) - 1
    dt = time[1] - time[0]
    matrix_exp_iwt = make_3d_exp(N, time, omega)
    Sminus_cumsum = np.cumsum(Sminus_t * dt * matrix_exp_iwt, axis=2)
    if np.size(g_w) > 1:
        moments, a_avg = op.operator_moments_b(NUM_MOMENTS, atomic_rho,
                                               np.exp(-1j * omega * time[t_0]) * np.conj(g_w[omega_0]) * Sminus_cumsum[
                                                                                                         :, :,
                                                                                                         t_0],
                                               np.exp(1j * omega * time[t_0]) * g_w[omega_0] *
                                               np.conj(Sminus_cumsum[:, :, t_0].T))
    else:
        moments, a_avg = op.operator_moments_b(NUM_MOMENTS, atomic_rho,
                                               np.exp(-1j * omega * time[t_0]) * np.conj(g_w) * Sminus_cumsum[:, :,
                                                                                                t_0],
                                               np.exp(1j * omega * time[t_0]) * g_w *
                                               np.conj(Sminus_cumsum[:, :, t_0].T))

    x, p, wigner = w.calc_Wigner_from_moments(moments, a_avg)
    print(a_avg)
    dx = x[1, 2] - x[1, 1]
    dp = p[2, 1] - p[1, 1]
    return x, p, wigner, Sminus_cumsum


def calc_wigner_for_omega(NUM_MOMENTS, time, op, omega_arr, Sminus_t, g_w, atomic_rho):
    N = np.size(Sminus_t, 0) - 1
    dt = time[1] - time[0]
    t_0 = 100000
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


def make_video(label, length=100):
    from pathlib import Path
    img_clips = []
    path_list = []
    # accessing path of each image
    for i in range(length+1):
        try:
            path_list.append(os.path.join('Images/' + label + str(i) + '.png'))
        except:
            continue
    # creating slide for each image
    for img_path in path_list:
        slide = ImageClip(img_path, duration=1 / 10)
        img_clips.append(slide)
    # concatenating slides
    video_slides = concatenate_videoclips(img_clips, method='compose')
    # exporting final video
    video_slides.write_videofile(label[:-1] + ".mp4", fps=24)


def plot_wigner_videos(NUM_MOMENTS, atomic_wigner, Sx_t, Sy_t, Sz_t, rho_t, time, omega, op, g_w):
    w = Wigner(N_alpha=100)
    Splus_t = (Sx_t + 1j * Sy_t)
    Sminus_t = np.zeros(np.shape(Splus_t))

    for i in range(len(time)):
        Sminus_t[:, :, i] = np.conj(Splus_t[:, :, i].T)

    frame_index : int = 0
    for index in range(0, NUM_TIMES, NUM_TIMES//NUM_VIDEO_FRAMES):
        print(str(index) + '/' + str(len(time)))
        fig = atomic_wigner.Wigner_BlochSphere(100, np.size(rho_t, 0) - 1, [], rho_t[:, :, index], 'rho')
        ax2 = fig.add_subplot(122)
        x, p, wigner, Sminus_cumsum = calc_wigner(NUM_MOMENTS, Sminus_t, op, omega, time, index, g_w, rho_t[:, :, 0], w)
        ax2.pcolormesh(x, p, wigner, cmap='bwr')
        m = ax2.collections[0]
        ax2.set_xlabel('$q$', fontsize=16)
        ax2.set_ylabel('$p$', fontsize=16)
        ax2.set_aspect('equal')
        ax2.text(0.1, 1.7, '$t \;= \;' + str(format(time[index] / 0.03 ** -1, ".2f")) + '\;\Gamma^{-1}$', fontsize=16)
        g = plt.colorbar(m, shrink=0.5)
        g.set_label('$W(q,p)$', fontsize=16)
        m.set_clim(-np.max(np.abs(wigner)), np.max(np.abs(wigner)))
        plt.tight_layout()
        plt.savefig('Images/time/' + str(frame_index) + '.png')
        plt.close(fig)
        frame_index += 1

    make_video('time/', length=NUM_VIDEO_FRAMES)


def plot_Sz(rho_t, Sz_t, time):
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
    line, = plt.plot(time * superradiance_no_cavity.GAMMA, expectation_h, linewidth=3)
    line.set_label('Heisenberg')
    line2, = plt.plot(time * superradiance_no_cavity.GAMMA, expectation_s, linewidth=3, linestyle='--')
    line2.set_label('Schrodinger')
    plt.xlabel('$time\; (\Gamma^{-1})$', fontsize=16)
    plt.ylabel('$<S_z>(t)$', fontsize=16)
    # plt.show()
    line3, = plt.plot(time * superradiance_no_cavity.GAMMA, -0.5 + np.exp(-superradiance_no_cavity.GAMMA * time),
                      linewidth=1, linestyle='-', color='black')
    line3.set_label('Analytic')
    plt.legend()
    plt.show()


def plot_intensity(Splus_cumsum, Sminus_t, k, g_w, c, atomic_rho):
    k = np.linspace(-INFINITY, INFINITY, NUM_POINTS, dtype=complex)
    I_x = np.zeros([len(k), 100])
    jump = 1000
    a_k = op.find_a_k(Sminus_t, k, time, 10000, g_w, c)
    mat = np.zeros([N + 1, N + 1])
    mat[1, 1] = 1
    print(a_k @ mat)
    for t_0 in range(0, 20000, jump):
        E_xplus, E_xminus, x = op.find_a_x(a_k, k)
        print(t_0)
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
    plt.xlim([0, 10])
    plt.show()
def plot_spectrum(omega, N, photon_number, domega):
    f2 = plt.figure()
    plt.plot(omega, N * photon_number / (sum(photon_number * omega) * domega), linewidth=2)
    plt.xlabel('$\Omega$', fontsize=16)
    plt.ylabel('$\\frac{dn}{d\Omega}$', fontsize=16)

    plt.show()


def visualize_light_from_atomic_density_matrix(
    rho : np.matrix, 
    num_atmos : int,
    gamma : float = 0.03,
    delta_t : Optional[float] = None,
) -> None :

    # Constants and params:
    num_moments = num_atmos + 1    
    OMEGA_0 = 1
    omega1 = 1

    # Complete missing values:
    if delta_t is None:
        delta_t = 2 / gamma

    # g_w = superradiance_no_cavity.GAMMA ** 0.5 / (np.pi ** 0.5)
    g_w = gamma ** 0.5 / (np.pi ** 0.5)
    c = 0.3

    atomic = Atomic_state_on_bloch_sphere(num_atmos)
    op = Operators(num_atmos)


    Splus_cumsum, Sx_t, Sy_t, Sz_t, rho_t, time = solve_sr_equations(op, num_atmos, rho, omega1, gamma=gamma, delta_t=delta_t, num_times=NUM_TIMES)
    Splus_t = (Sx_t + 1j * Sy_t)
    Sminus_t = op.dagger_3d(Splus_t)

    if not os.path.isdir('images'):
        os.mkdir('images')
    if not os.path.isdir('images/time'):
        os.mkdir('images/time')
    if not os.path.isdir('images/omega'):
        os.mkdir('images/omega')
    # plot_Sz(rho_t, Sz_t, time)

    plot_wigner_videos(num_moments, atomic, Sx_t, Sy_t, Sz_t, rho_t, time, omega1, op, g_w)

    w = Wigner()
    omega = np.linspace(0, 6, 200)
    g_omega = g_w * (omega / OMEGA_0) ** 0.5
    domega = omega[1] - omega[0]

    #
    x, p, wigner_w, photon_number = calc_wigner_for_omega(num_moments, time, op, omega, Sminus_t, g_omega, rho)
    make_matrix_movie(x, p, wigner_w, omega, x_label='$q$', y_label='$p$',
                      text_label='$\\frac{\Omega - \Omega_0}{\Gamma}\;=\;$', color_label='$W(q,p)$')
    # plot_spectrum(omega, N, photon_number, domega)

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

    rho_final = rho_t[:,:,-1]
    return rho_final
    

def main():
    '''
    ### Hi Nir :) to use this:
    ### (1) change N for the number of atoms,
    ### (2) change Num moments, should be greater than number of photons that are emmited. the code gets slow if this number is large (>40)
    ### (3) change atomic_rho as a numpy array.(N+1XN+1) in the symetric state basis. (rho_mn=<m|rho|n>)
    '''
    gamma = 0.3
    g_w = gamma ** 0.5 / (np.pi ** 0.5)
    delta_t = np.log(2)/gamma

    NUM_ATOMS = 1    ###change###
    NUM_MOMENTS = 2   ###change###  # usually N+1

    atomic = Atomic_state_on_bloch_sphere(NUM_ATOMS)
    op = Operators(NUM_ATOMS)

    atomic_rho = op.fock_light_state(NUM_ATOMS, NUM_ATOMS)    ###change###


    omega1 = 1
    Splus_cumsum, Sx_t, Sy_t, Sz_t, rho_t, time = solve_sr_equations(op, NUM_ATOMS, atomic_rho, omega1, gamma=gamma, delta_t=delta_t, num_times=NUM_TIMES)
    rho_final = rho_t[:,:,-1]
    Splus_t = (Sx_t + 1j * Sy_t)
    Sminus_t = op.dagger_3d(Splus_t)

    if not os.path.isdir('images'):
        os.mkdir('images')
    if not os.path.isdir('images/time'):
        os.mkdir('images/time')
    if not os.path.isdir('images/omega'):
        os.mkdir('images/omega')
    # plot_Sz(rho_t, Sz_t, time)

    plot_wigner_videos(NUM_MOMENTS, atomic, Sx_t, Sy_t, Sz_t, rho_t, time, omega1, op, g_w)

    w = Wigner()
    omega = np.linspace(0, 6, 200)
    g_omega = g_w * (omega / superradiance_no_cavity.OMEGA_0) ** 0.5
    domega = omega[1] - omega[0]




def test_unitary():
    N = 40
    NUM_MOMENTS = 41
    atomic = Atomic_state_on_bloch_sphere(5)
    op = Operators(N)
    Sx, Sy, Sz = op.create_spin_matrices(N)
    atomic_rho = op.fock_light_state(6, N)
    atomic_rho_new = scipy.linalg.expm(-1j * pi * Sx.T) @ atomic_rho @ scipy.linalg.expm(1j * pi * Sx)
    fig = atomic.Wigner_BlochSphere(100, N, [], atomic_rho, 'rho')
    fig2 = atomic.Wigner_BlochSphere(100, N, [], atomic_rho_new, 'rho')

    plt.show()


if __name__ == '__main__':
    main()
    # test_unitary()
    
    # test_Wigner_from_moments()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
