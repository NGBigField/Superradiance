# ==================================================================================== #
#| Imports:                                         
# ==================================================================================== #

if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.parent.__str__()
    )
# For type annotations:
from typing import Callable

# Import useful types:
from algo.optimization import BaseParamType
from algo.coherentcontrol import Operation

# Basic algo mechanisms:
from algo.coherentcontrol import CoherentControl
from physics.famous_density_matrices import ground_state

# for numerics:
import numpy as np

# Our best optimized results:    
from scripts.optimize.cat4_i     import best_sequence_params as cat4_params
from scripts.optimize.cat2_i     import best_sequence_params as cat2_params
from scripts.optimize.gkp_hex    import best_sequence_params as gkp_hex_params
from scripts.optimize.gkp_square import best_sequence_params as gkp_square_params

# Cost function:
from algo.common_cost_functions import fidelity_to_cat, fidelity_to_gkp

# for plotting:
from utils.visuals import plot_matter_state, plot_wigner_bloch_sphere, plot_plain_wigner, ViewingAngles, BlochSphereConfig, save_figure, draw_now
from utils import assertions, saveload
import matplotlib.pyplot as plt

# for printing progress:
from utils import strings

# for enums:
from enum import Enum, auto

# for sleeping:
from time import sleep

# For emitted light calculation 
from physics.emitted_light_approx import main as calc_emitted_light

# For writing results to file:
from csv import DictWriter

# ==================================================================================== #
#| Constants:
# ==================================================================================== #

# DEFAULT_COLORLIM = (-0.1, 0.2)
DEFAULT_COLORLIM = None


# ==================================================================================== #
#| Helper types:
# ==================================================================================== #

class StateType(Enum):
    GKPHex = auto()
    GKPSquare = auto()
    Cat2 = auto()
    Cat4 = auto()


# ==================================================================================== #
#| Inner Functions:
# ==================================================================================== #


def print_params_canonical(operations:list[Operation], thetas:list[float], state_name:str="something")->None:

    ## Write result: =
    output_file = "Params "+state_name+".csv"

    def get_dict_write(f):
        return DictWriter(f, fieldnames=["nx", "ny", "nz", "theta", "a", "b"], lineterminator="\n")

    with open(output_file, 'w') as f:
        row = dict(
            nx="nx",
            ny="ny",
            nz="nz",
            theta="theta",
            a="a",
            b="b"
        )
        dict_writer = get_dict_write(f)
        dict_writer.writerow(row)

    def write_row(row:dict):
        with open( output_file ,'a') as f:
            dict_writer = get_dict_write(f)
            dict_writer.writerow(row)
            


    i_theta = 0
    i_step = -1
    row = dict()

    for operation in operations:
        n_theta = operation.num_params
        theta = [thetas[i] for i in range(i_theta, i_theta+n_theta)]
        i_theta += n_theta


        if operation.name == 'rotation':
            i_step += 1
            assert len(theta)==3
            row = dict()

            x, y, z = theta
            norm_ = np.sqrt(x**2 + y**2 + z**2)
            x, y, z = [val/norm_ for val in theta]
            theta = norm_
            n_str_ : str = ""
            for v in [x, y, z]:
                n_str_ += f"{v:.5}, "
            n_str_ = n_str_[:-2]
            str_ = f"n{i_step}="+n_str_
            str_ += f"\ntheta{i_step}={theta:.5}"
            print(str_)

            row["nx"] = x
            row["ny"] = y
            row["nz"] = z
            row["theta"] = theta
        
        elif operation.name == 'squeezing':
            assert len(theta)==2
            a, b = theta
            print(f"a{i_step}={a:.5}\nb{i_step}={b:.5}")

            row["a"] = a
            row["b"] = b

            write_row(row)

        else:
            raise ValueError("Not a supported print case")

    write_row(row)


def _get_emitted_light(state_type:StateType, final_state:np.matrix, fidelity:float) -> np.matrix:
    # Basic info:
    file_name = f"Emitted-Light {state_type.name} fidelity={fidelity}"
    sub_folder = "Emitted-Light"
    # Get or calc:
    if saveload.exist(file_name, sub_folder=sub_folder):
        emitted_light_state = saveload.load(file_name, sub_folder=sub_folder)        
    else:
        emitted_light_state = calc_emitted_light(final_state, t_final=0.1, time_resolution=1000)
        saveload.save(emitted_light_state, name=file_name, sub_folder=sub_folder)
    # Return:
    return emitted_light_state #type: ignore


def _get_best_params(
    type_:StateType, 
    num_atoms:int,
    num_intermediate_states:int
) -> tuple[
    list[BaseParamType],
    list[Operation]
]:
    if type_ is StateType.GKPHex:
        return gkp_hex_params(num_atoms, num_intermediate_states=num_intermediate_states)
    elif type_ is StateType.GKPSquare:
        return gkp_square_params(num_atoms, num_intermediate_states=num_intermediate_states)
    elif type_ is StateType.Cat4:
        return cat4_params(num_atoms, num_intermediate_states=num_intermediate_states)
    elif type_ is StateType.Cat2:
        return cat2_params(num_atoms, num_intermediate_states=num_intermediate_states)
    else:
        raise ValueError(f"Not an option '{type_}'")


def _get_type_inputs(
    state_type:StateType, num_atoms:int, num_intermediate_states:int
) -> tuple[
    CoherentControl,
    np.matrix,
    list[float],
    list[Operation],
    Callable[[np.matrix], float]
]:
    # Get all needed data:
    params, operations = _get_best_params(state_type, num_atoms, num_intermediate_states)
    initial_state = ground_state(num_atoms=num_atoms)
    coherent_control = CoherentControl(num_atoms=num_atoms)
    cost_function = _get_cost_function(type_=state_type, num_atoms=num_atoms)
    
    # derive theta:
    theta = [param.get_value() for param in params]
    
    return coherent_control, initial_state, theta, operations, cost_function
   
   
def _get_cost_function(type_:StateType, num_atoms:int) -> Callable[[np.matrix], float]:
    if type_ is StateType.GKPHex:
        return fidelity_to_gkp(num_atoms=num_atoms, gkp_form="hex")
    elif type_ is StateType.GKPSquare:
        return fidelity_to_gkp(num_atoms=num_atoms, gkp_form="square")        
    elif type_ is StateType.Cat4:
        return fidelity_to_cat(num_atoms=num_atoms, num_legs=4, phase=np.pi/4)
    elif type_ is StateType.Cat2:
        return fidelity_to_cat(num_atoms=num_atoms, num_legs=2, phase=np.pi/2)        
    else:
        raise ValueError(f"Not an option '{type_}'")

def _print_fidelity(final_state:np.matrix, cost_function:Callable[[np.matrix], float], prefix:str="") -> float:
    cost = cost_function(final_state)
    fidelity = -cost
    print(prefix+f"Fidelity = {fidelity}")
    return fidelity
    
        
def _get_movie_config(
    create_movie:bool, num_transition_frames:int, state_type:StateType
) -> CoherentControl.MovieConfig:
    # Basic data:
    fps=30
    
    bloch_sphere_config = BlochSphereConfig(
        alpha_min=0.2,
        resolution=250,
        viewing_angles=ViewingAngles(
            elev=-45
        )
    )
    
    # Movie config:
    movie_config=CoherentControl.MovieConfig(
        active=create_movie,
        show_now=False,
        num_freeze_frames=fps//2,
        fps=fps,
        bloch_sphere_config=bloch_sphere_config,
        num_transition_frames=num_transition_frames,
        temp_dir_name=state_type.name     
    )
    
    return movie_config

# ==================================================================================== #
#| Declared Functions:
# ==================================================================================== #



def plot_sequence(
    state_type:StateType = StateType.GKPSquare,
    num_atoms:int = 40,
    folder:str|None = None
):
    # constants:

    # get
    coherent_control, initial_state, theta, operations, cost_function = _get_type_inputs(state_type=state_type, num_atoms=num_atoms, num_intermediate_states=0)
    
    # derive:
    n = assertions.integer( (len(operations)-1)/2 )
    
    # iterate: 
    for i in range(n+1):
        print(strings.num_out_of_num(i, n))

        # derive params for this iteration:
        if i==0:
            theta_i = []
            operations_i = []
        else:
            theta_i = theta[:i*5+3]
            operations_i = operations[:i*2+1]
        name = state_type.name+f"-{i:02}"

        # Get state:
        state_i = coherent_control.custom_sequence(state=initial_state, theta=theta_i, operations=operations_i)
    
        # plot light:
        plot_plain_wigner(state_i)
        save_figure(folder=folder, file_name=name+" - Light")        

        # plot bloch:
        plot_wigner_bloch_sphere(state_i, view_elev=-90, alpha_min=1, title="", num_points=200)
        save_figure(folder=folder, file_name=name+" - Sphere")
        
        # Sleep and close open figures:
        sleep(1)
        plt.close("all")




## Main:
def print_all_fidelities(num_atoms=40):

    for state_type in StateType:

        # Basic info
        coherent_control, initial_state, theta, operations, cost_function = _get_type_inputs(state_type=state_type, num_atoms=num_atoms, num_intermediate_states=0)
        movie_config = _get_movie_config(create_movie=False, num_transition_frames=0, state_type=state_type)    
        state_name = state_type.name
        num_steps = sum([1 for op in operations if op.name=="squeezing"])
        
        # print stuff:
        print("")
        print(f"===========")
        print(f"State: {state_name!r}")
        print(f"num steps={num_steps}")

        # Create final matter state:
        matter_state = coherent_control.custom_sequence(state=initial_state, theta=theta, operations=operations, movie_config=movie_config)
        matter_fidelity = -cost_function(matter_state)

        # Get light state:
        emitted_light_state = _get_emitted_light(state_type, matter_state, matter_fidelity)
        emitted_light_fidelity = -cost_function(emitted_light_state)

        # print stuff:
        print(f"Matter Fidelity={matter_fidelity}")
        print(f"Light Fidelity={emitted_light_fidelity}")
        print("")
         
            


def plot_all_best_results(
    create_movie:bool = False,
    num_atoms:int = 40
):
    for state_type in StateType:
        print(" ")
        print(state_type.name)
        plot_result(state_type, create_movie, num_atoms)        
        print(" ")
        
    print("Done.")
    

def plot_result(
    state_type:StateType,
    create_movie:bool = False,
    num_atoms:int = 20,
    num_graphics_points:int = 200):
    
    # derive:
    num_transition_frames = 20 if create_movie else 0
    state_name = state_type.name
    if num_atoms != 40:
        state_name += f"{num_atoms}"
    

    # get
    coherent_control, initial_state, theta, operations, cost_function = _get_type_inputs(state_type=state_type, num_atoms=num_atoms, num_intermediate_states=num_transition_frames)
    movie_config = _get_movie_config(create_movie, num_transition_frames, state_type)    
    num_steps = sum([1 for op in operations if op.name=="squeezing"])

#N=10
#steps=3
#fidelity:0.9936255862486427
    #theta=[ 1.4470e-04,  4.6672e-01, -9.9268e-01,  2.9659e-02,  6.6818e-01, -3.1939e-01, -1.2439e+00, -1.1267e+00,  1.1434e-01,  6.9204e-01,  4.2567e-01,  2.4735e+00,  1.8104e-01, -1.4641e-01, -3.5958e-01, -2.5825e-01, -2.7518e+00, -2.6135e-01]

#N=20
#steps=6
#fidelity:0.9649173575745271
    #theta=[ 1.3254e-01,  3.5439e-01, -3.4821e-02,  2.0952e-02,  3.0676e-01, -1.6963e-03,  3.3742e-01, -1.3229e+00,  3.9762e-02,  2.0592e-01,  6.8072e-03, -9.8407e-04, -5.7724e-02,  4.6752e-02, -1.0915e-01,  2.1457e+00, -9.3248e-01,  1.7924e+00, -3.5026e-02, -2.7228e-01, -2.2552e+00,  2.2877e+00, -1.7418e+00, -1.1362e+00, -2.6776e-01,  5.5782e-03, -2.3941e+00, -1.0607e+00,  6.2408e-02, -1.6803e-01,  5.0719e-02, -1.1725e+00,  1.2399e+00]

    #N=30
    #fidelity:0.9827267642335243
    #theta=[+0.0371251371052456 , -0.0999429408543143 , -0.0736321340967989 , +0.0216382303669356 , +0.2404117744408015 , 
#-0.0397517799262118 , +0.3046649921165432 , -1.0883600177733002 , +0.0166430686142774 , +0.2238876791319606 , 
#-0.0164479098467608 , +0.1341565799988126 , +0.0055981211750407 , -0.0187763941534583 , -0.0785622420327235 , 
#+2.4276814658531580 , -0.9544107563076063 , +1.3948786061625289 , -0.0439007510416618 , -0.2877428154140896 , 
#-2.0000203133445229 , +2.2123772331384473 , -1.3990410176224630 , -1.1044655173778124 , -0.2419290803901949 , 
#+0.1052023685987893 , -2.2761946360305831 , -1.2244630323780572 , +0.0983869837147157 , -0.1402544298012540 , 
#-0.0893263224082521 , -0.2026731688384964 , +0.0390758055955956 , +0.4080504310829461 , +0.1660945401775654 , 
#+1.6612568203093048 , -0.1749686725995780 , -0.4155264392168456 , +0.0037125466583421 , +0.1485760395891290 , 
#+0.7528298083142935 , +0.3139771868447770 , +0.3920113423013637 , +0.0796802064106102 , -0.0704016259079576 , 
#+0.0239599785319919 , -0.2392206678332991 , -0.0303951114846882 , -0.0901115586079922 , +0.0559725249145085 , 
#+0.1437019825717479 , -0.3641075385649459 , -0.3878211085902062 , -0.3387319579763898 , +0.9937377422832840 , 
#-0.0161579487331315 , -1.2982581800870960 , +1.0732834575988681]
    #N=40
    # 98.38 fidelity
    #theta = [ -0.0014068851683347 , +0.1215726163875156 , -0.3673153331306065 , +0.0249399804630062 , +0.2269478535658498 , 
    # -0.0190652512267484 , +0.2510723996692263 , -1.0607112513459973 , +0.0024516613121189 , +0.2156517674728328 , 
    # +0.0319448795412145 , +0.1558487506505765 , -0.0062749313184094 , -0.0039112520369149 , -0.0663501151821770 , 
    # +2.4499465316584099 , -1.1174755201996209 , +1.4082519226188759 , -0.0483762205467460 , -0.2901126138956479 , 
    #-2.2320871070585953 , +2.2071586343941316 , -1.5501525031566996 , -1.1049653620985869 , -0.2459256908092352 , 
    #+0.0586744248216218 , -2.5988976065955574 , -1.0735986268925384 , +0.0287576930899591 , -0.1139451953576366 , 
    #-0.2675536546291929 , -0.2192988258684723 , +0.0562553082535889 , +0.4029275777194282 , +0.1678248016415093 , 
    #+1.6503807621359785 , -0.2424417180525995 , -0.4117323968553763 , -0.0077387709781708 , +0.1030606188833836 , 
    #+0.6584106259075400 , +0.1722046531631806 , +0.3612403708545883 , +0.0450059509382275 , -0.0653746424180170 , 
    #+0.0560022534150009 , -0.2381812144256178 , +0.0760193203431392 , -0.0819591524260033 , +0.0493671010508360 , 
    #+0.1628340991897231 , -0.3401516974436361 , -0.3380686055728756 , -0.3092228385753361 , +0.9926172832861413 , 
    #-0.0109342817022105 , -0.9717446733114196 , +1.0844122307811856] 



    # create matter state:
    matter_state = coherent_control.custom_sequence(state=initial_state, theta=theta, operations=operations, movie_config=movie_config)
    
 
    ## Naive projection onto plain:
    # plot_plain_wigner(matter_state, with_colorbar=True)
    # save_figure(file_name=state_name+" - Projection - colorbar")
    # plot_plain_wigner(matter_state, with_colorbar=False)
    # save_figure(file_name=state_name+" - Projection")

    ## plot bloch:
    plot_wigner_bloch_sphere(matter_state, alpha_min=1.0, title="", num_points=num_graphics_points, view_elev=-90)
    plot_plain_wigner(matter_state, with_colorbar=True, colorlims=DEFAULT_COLORLIM)
    plt.show()
    #save_figure(file_name=state_name+" - Sphere")
    
    #save_figure(file_name=state_name+" - Light")

    # print  Data:
    #print(f"State: {state_name!r}")
    #print(f"num steps={num_steps}")
    #fidelity = _print_fidelity(matter_state, cost_function, "Matter state ")

    # Print params:
    # print_params_canonical(operations, theta, state_name)   

    ## plot light:
    #emitted_light_state = _get_emitted_light(state_type, matter_state, fidelity)
    # plot_plain_wigner(emitted_light_state, with_colorbar=True, colorlims=DEFAULT_COLORLIM)
    # save_figure(file_name=state_name+" - Light - colorbar")
    #plot_plain_wigner(emitted_light_state, with_colorbar=False, colorlims=DEFAULT_COLORLIM, with_axes=False, num_points=num_graphics_points)
    #save_figure(file_name=state_name+" - Light", tight=True)
    
    #fidelity = _print_fidelity(emitted_light_state, cost_function, "Emitted light ")

    ## # plot complete matter picture:
    # bloch_config = BlochSphereConfig()
    # plot_matter_state(final_state, config=bloch_config)
    # save_figure(file_name=state_type.name+" - Matter")

    #return fidelity


def create_movie(
    state_type:StateType = StateType.GKPHex,
    num_atoms:int = 40,
    num_transition_frames = 40
):
    # derive:
    
    # Print:
    print(f"State: {state_type.name!r}")

    # get
    coherent_control, initial_state, theta, operations, cost_function = _get_type_inputs(state_type=state_type, num_atoms=num_atoms, num_intermediate_states=num_transition_frames)
    movie_config = _get_movie_config(True, num_transition_frames, state_type)
    
    # create state:
    final_state = coherent_control.custom_sequence(state=initial_state, theta=theta, operations=operations, movie_config=movie_config)
    
    # print  fidelity:
    _print_fidelity(final_state, cost_function)    
    return cost_function(final_state)



if __name__ == "__main__":
    # plot_sequence()
    # create_movie()
    plot_result(StateType.GKPSquare)
    # plot_all_best_results()
    # print_all_fidelities()
    
    print("Done.")