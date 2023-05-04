import numpy as np
import qutip
import matplotlib
import matplotlib.pyplot as plt
from qutip import *
from scipy.integrate import solve_ivp
import scipy
from scipy.fft import fft, fftfreq
import numpy as np

# Number of sample points
OMEGA_1 = 1.  # Laser Rabi Frequency (*2)
OMEGA_0 = 0  # atomic transition frequency
DEFAULT_TIME_RES = 200
GAMMA = 1
GAMMA_s = 0.
DEFAULT_T_FINAL = 0.6  # GAMMA ** -1 * 30
cmap = matplotlib.cm.bwr


def coupling1_coef(t, args):
    coupling_eta = args['coupling_eta']
    times = args['times']
    index = np.where(np.abs(times - t) == np.min(np.abs(t - times)))[0]
    return coupling_eta[index]


def coupling1_coef_conj(t, args):
    return np.conj(coupling1_coef(t, args))


def coupling2_coef(t, args):
    coupling_nu = args['coupling_nu']
    times = args['times']
    index = np.where(np.abs(times - t) == np.min(np.abs(t - times)))[0]
    return coupling_nu[index]


def coupling2_coef_conj(t, args):
    return np.conj(coupling2_coef(t, args))


def H2_coef(t, args):
    return np.conj(coupling1_coef(t, args)) * coupling2_coef(t, args)


def H2_coef_conj(t, args):
    return np.conj(H2_coef(t, args))


def laser_coef(t, args):
    laser = args['laser']
    times = args['times']
    index = np.where(np.abs(times - t) == np.min(np.abs(t - times)))[0]
    # plt.plot(mode)
    # plt.show()
    return laser[index]


def create_operators(atom_dim, light_dim, num_modes, three_level=False):
    sig_z, sig_p, sig_m = qutip.jmat((atom_dim - 1) / 2, 'z'), qutip.jmat((atom_dim - 1) / 2, '+'), qutip.jmat(
        (atom_dim - 1) / 2, '-')

    a_eta, sp, sm, sz = tensor(qeye(atom_dim), destroy(light_dim)), \
                        tensor(sig_p, qeye(light_dim)), \
                        tensor(sig_m, qeye(light_dim)), \
                        tensor(sig_z, qeye(light_dim))

    return a_eta, sp, sm, sz


def get_coupling_from_mode(mode, times, position=1, coupling_prev=None):
    '''

    :param mode:
    :param times:
    :param position:
    :param coupling_prev:
    :return:
    '''
    dt = times[2] - times[1]
    coupling = np.zeros_like(mode)
    if not np.any(mode):
        return coupling

    for i in range(len(mode)):
        S = np.sum(np.abs(mode[:i + 1]) ** 2 * dt) ** 0.5
        if S > 0:
            coupling[i] = -np.conj(mode[i]) / S
    return coupling


def get_light_outside(rho_init, times, GAMMA, light_dim, atom_dim, mode_eta, progress_bar:bool=True):
    '''

    :param rho_init:
    :param light_dim:
    :param atom_dim:
    :param times:
    :return:
    '''

    dt = times[2] - times[1]
    num_modes = 10
    a_eta, sp, sm, sz = create_operators(atom_dim, light_dim, num_modes=num_modes, three_level=False)

    H1 = 1j / 2 * sm.dag() * a_eta * GAMMA ** 0.5
    H1_dag = -1j / 2 * sm * a_eta.dag() * GAMMA ** 0.5

    OMEGA_0 = 0.
    H0 = OMEGA_0 * sz

    coupling_eta = get_coupling_from_mode(mode_eta, times)

    H_laser = (sm + sp)

    opts = Options()
    opts.store_states = True

    args = {'coupling_eta': coupling_eta, 'times': times}
    H_lst = [H0, [H1, coupling1_coef], [H1_dag, coupling1_coef_conj]]
    L_lst = [GAMMA ** 0.5 * sm, [a_eta, coupling1_coef_conj]]

    result = mesolve(H_lst, rho_init, times, [L_lst, [GAMMA_s ** 0.5 * sm]], args=args, options=opts, progress_bar=progress_bar)
    rho_f = result.states[-1].ptrace([1])
    return times, rho_f


def create_laser_signal(times, pulse_times, theta_pulses, OMEGA_1):
    mask = np.zeros_like(times)
    for i, t in enumerate(pulse_times):
        mask[(times > t) & (times <= GAMMA * (t + theta_pulses[i] / 2 / OMEGA_1))] = 1
    return mask * OMEGA_1


def plot_modes(times, modes, ax):
    for mode in modes:
        ax.plot(times, np.real(mode), linewidth=2)
    ax.set_xlabel('time $\\Gamma^{-1}$', fontsize=10)
    ax.set_ylabel('$Real(amplitude)$', fontsize=10)
    ax.legend(['mode 1', 'mode 2'], fontsize=10)
    ax.set_title('')


def plot_intensity(times, result, ax):
    ax.plot(times, np.array(result.expect).T)
    # ax.set_xticks(fontsize=10)
    # ax.set_yticks(fontsize=10)
    ax.set_xlabel('time $\\Gamma^{-1}$', fontsize=10)
    ax.set_ylabel('$<S_z>$', fontsize=10)


def plot_g2(times, g2):
    f, (ax1, ax2) = plt.subplots(1, 2)
    m = ax1.pcolormesh(times, times, np.abs(g2), cmap='bwr')
    m = ax1.collections[0]
    m.set_clim(-1, 1)

    ax2.plot(times - times[len(g2) // 2], [g2[len(g2) // 2, i] for i in range(len(g2))])
    ax2.set_xlim([-50, 50])
    plt.show()


def get_g1_modes(H, psi_init, times, sig_m, sig_p, GAMMA_s, num_modes=10, plot_mods:bool=False):
    dt = times[1] - times[0]
    corr = correlation_2op_2t(H, psi_init, times, times,
                              [[GAMMA ** 0.5 * sig_m, GAMMA_s ** 0.5 * sig_m]], sig_p, sig_m)
    corr_cor = np.zeros_like(corr)
    for i in range(np.size(corr, 0)):
        corr_cor[i, i + 1:] = corr[i, :-i - 1]
    corr_cor = corr_cor + np.conj(corr_cor.T)
    for i in range(np.size(corr, 0)):
        corr_cor[i, i] = corr[i, 0]

    g1 = np.zeros_like(corr_cor)
    for i in range(len(corr_cor)):
        if np.abs(corr_cor[i, i]) > 0.01:
            # g2[i, :] = corr_g2[i, :] / corr_cor[i, i] ** 2
            g1[i, :] = corr_cor[i, :] / corr_cor[i, i]

    # plot_g2(times, g2)
    # plot_g1
    if plot_mods:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)

        ax1.pcolormesh(times, times, np.real(g1), cmap='bwr')
        m = ax1.collections[0]
        m.set_clim(-1, 1)
        ax1.set_xlabel('t1', fontsize=12)
        ax1.set_ylabel('t2', fontsize=12)
        h = plt.colorbar(m)
        h.set_label('$g^{(1)}(t_1,t_2)$')
        
    w, v = np.linalg.eig(corr_cor * GAMMA * dt)

    modes = []
    occupations = []
    for i in range(3):
        modes.append(v[:, w == np.sort(w)[-i - 1]])
        occupations.append(np.sort(np.abs(w))[-i - 1])
        if plot_mods:
            ax2.plot(times, np.real(v[:, w == np.sort(w)[-i - 1]]))
            ax2.set_xlabel('times ($\\Gamma^{-1}$)', fontsize=12)
            
    if plot_mods:
        ax2.legend(['mode 1', 'mode 2', 'mode 3', 'mode 4'], fontsize=12)
        num_oc = 15
        ax3.bar(np.linspace(1, num_oc, num_oc), np.abs(np.sort(w)[:-num_oc - 1:-1]))
        ax3.set_xlabel('mode', fontsize=12)
        ax3.set_ylabel('occupation', fontsize=12)
        
    return np.real(modes), np.real(occupations)


def make_plots(times, modes, rho_eta_mode_a, rho_eta_mode_b, result, occupations):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    plot_intensity(times, result, ax1)
    plot_modes(times, modes, ax2)

    print(rho_eta_mode_a)
    qutip.plot_wigner(qutip.Qobj(rho_eta_mode_a), cmap=cmap, colorbar=True, fig=fig, ax=ax3)
    qutip.plot_wigner(qutip.Qobj(rho_eta_mode_b), cmap=cmap, colorbar=True, fig=fig, ax=ax4)
    ax3.set_title('')
    ax4.set_title('')

    # ax3.bar(np.linspace(1, len(occupations), len(occupations)), np.real(occupations))
    plt.show()


def simulate_emitter(times, atom_dim, light_dim, atomic_rho, light_rho,
                     plot_mods=False, state_outside=True, progress_bar=True):
    sig_z, sig_p, sig_m = qutip.jmat((atom_dim - 1) / 2, 'z'), qutip.jmat((atom_dim - 1) / 2, '+'), qutip.jmat(
        (atom_dim - 1) / 2, '-')

    H = [0 * sig_z]

    # opts = qutip.Options(nsteps=15000, atol=1e-13, rtol=1e-13)
    # opts.store_states = True
    # psi_init = qutip.basis(atom_dim, 1)
    dt = times[1] - times[0]

    modes, occupations = get_g1_modes(H, atomic_rho, times, sig_m, sig_p, GAMMA_s, plot_mods=plot_mods)
    mode_1 = modes[0]
    mode_1 = np.array(mode_1 / (np.sum(np.abs(mode_1) ** 2 * dt) ** 0.5))

    if not state_outside:
        return occupations

    rho_init = qutip.tensor(atomic_rho, ket2dm(qutip.basis(light_dim, 0)))

    times, rho_eta_mode_a = get_light_outside(rho_init, times, GAMMA, light_dim, atom_dim, mode_1, progress_bar=progress_bar)

    return rho_eta_mode_a


def example():
    times = np.linspace(0, t_final, DEFAULT_TIME_RES)

    atom_dim = 40
    light_dim = atom_dim + 1

    Sx = qutip.jmat((atom_dim - 1) / 2, 'x')
    atomic_rho = qutip.ket2dm(qutip.basis(atom_dim, atom_dim - 1))  # initial atomic state example all excited state
    atomic_rho = qutip.ket2dm(qutip.basis(atom_dim, 0))
    to_exp = 1j * Sx * np.pi / 2
    to_exp2 = -1j * Sx * np.pi / 2

    atomic_rho = (to_exp.expm() + to_exp2.expm()) * atomic_rho * (
            to_exp.expm() + to_exp2.expm()).dag()  # initial atomic state example cat state
    atomic_rho = atomic_rho/np.trace(atomic_rho)
    light_rho = qutip.ket2dm(qutip.basis(light_dim, 0))  # initial light state

    to_exp = 1j * Sx * np.pi
    atomic_rho = to_exp.expm() * atomic_rho * (to_exp.dag()).expm()  # qutip uses density matrices that are flipped

    # make_fidelity_map(atom_dim, light_dim, times, omega1s, gammass)

    rho_f = simulate_emitter(times, atom_dim, light_dim, atomic_rho, light_rho, plot=True,
                             state_outside=True)  # rho_f is what you need

    qutip.plot_wigner(rho_f, cmap='bwr', colorbar=True)
    plt.show()


def main(atomic_rho_in:np.matrix, time_resolution:int=DEFAULT_TIME_RES, t_final:float=DEFAULT_T_FINAL, progress_bar:bool=True, plot_mods:bool=False) -> np.matrix :
    
    ## Derive simple info from inputs and check:
    atom_dim = atomic_rho_in.shape[0]
    assert atomic_rho_in.shape[0]==atomic_rho_in.shape[1]
    light_dim = atom_dim 
    
    ## Define helpers
    Sx = qutip.jmat((atom_dim - 1) / 2, 'x')
    to_exp = 1j * Sx * np.pi 
    
    ## Prepare inputs:
    times = np.linspace(0, t_final, time_resolution)
    light_rho_initial = qutip.ket2dm(qutip.basis(light_dim, 0))  # initial light state
    atomic_rho = Qobj(atomic_rho_in)
    # plot_wigner(atomic_rho, cmap='bwr')    
    atomic_rho = to_exp.expm() * atomic_rho * (to_exp.dag()).expm()  # qutip uses density matrices that are flipped
    
    ## Calc
    light_rho_final = simulate_emitter(times, atom_dim, light_dim, atomic_rho, light_rho_initial, plot_mods=plot_mods, state_outside=True, progress_bar=progress_bar)
    
    
    ## Transform to numpy object:
    light_rho = light_rho_final.data
    if scipy.sparse.issparse(light_rho):
        light_rho = light_rho.todense()
    
    return light_rho
    

if __name__ == '__main__':
    example()
