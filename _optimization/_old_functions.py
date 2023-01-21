def learn_optimized_metric(
    initial_state : _DensityMatrixType,
    metric : Metric = Metric.NEGATIVITY,
    max_iter : int=1000, 
    num_pulses : int=5, 
    initial_guess : Optional[np.array] = None,
    save_results : bool=True,
) -> LearnedResults:
    
    # Choose cost-function:
    if metric is Metric.PURITY:
        measure = lambda state: purity(state)
    elif metric is Metric.NEGATIVITY:
        measure = lambda state: (-1)*negativity(state)
    else:
        raise ValueError("Not a valid option")
    
    coherent_control = _coherent_control_from_mat(initial_state)
    def _cost_func(theta) -> float:
        final_state = coherent_control.coherent_sequence(state=initial_state, theta=theta)        
        return measure(final_state)        

    # Call base function:
    return _common_learn(
        initial_state=initial_state,
        max_iter=max_iter,
        num_pulses=num_pulses,
        cost_function=_cost_func,
        save_results=save_results,
        initial_guess=initial_guess,
    )
    
    
def learn_specific_state(
    initial_state:_DensityMatrixType, 
    target_state:_DensityMatrixType, 
    max_iter : int=100, 
    num_pulses : int=3, 
    initial_guess : Optional[np.array] = None,
    save_results : bool=True,
) -> LearnedResults:

    # Check inputs:
    assertions.density_matrix(initial_state)
    assertions.density_matrix(target_state)
    assert initial_state.shape == target_state.shape
 
    # cost function:
    coherent_control = _coherent_control_from_mat(initial_state)
    def _cost_func(theta:np.ndarray) -> float :  
        final_state = coherent_control.coherent_sequence(state=initial_state, theta=theta)
        cost = fidelity(initial_state, final_state) * -1
        return cost
    # Call base function:
    return _common_learn(
        initial_state=initial_state,
        max_iter=max_iter,
        num_pulses=num_pulses,
        cost_function=_cost_func,
        save_results=save_results,
        initial_guess=initial_guess,
    )




    
    
def creating_gkp_algo(
    num_moments:int=40
) -> LearnedResults:

    ## Check inputs:
    assertions.even(num_moments)
    
    ## Define operations:
    initial_state = Fock.excited_state_density_matrix(num_moments)
    coherent_control = CoherentControl(num_moments=num_moments)
    standard_operations : CoherentControl.StandardOperations = coherent_control.standard_operations(num_intermediate_states=0)
    Sp = coherent_control.s_pulses.Sp
    Sx = coherent_control.s_pulses.Sx
    Sy = coherent_control.s_pulses.Sy
    Sz = coherent_control.s_pulses.Sz


    noon_data = _load_or_find_noon(num_moments)


    cat_creation_half_params = noon_data.params
    cat_creation_half_params[T4_PARAM_INDEX] = cat_creation_half_params[T4_PARAM_INDEX] / 5


    # Define new initial state:
    cat_creation_operations = \
        noon_data.operation + \
        [standard_operations.power_pulse_on_specific_directions(power=1, indices=[0])] + \
        noon_data.operation

    cat_creation_params = []
    cat_creation_params.extend(cat_creation_half_params)
    cat_creation_params.append(pi)  # x pi pulse
    cat_creation_params.extend(cat_creation_half_params)

    # our almost gkp state:
    cat_state = coherent_control.custom_sequence(state=initial_state, theta=cat_creation_params, operations=cat_creation_operations)

    ## Center the cat-state:
    print("Center the cat-state")
    operations = [
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0,1,2]),
    ]
    def cost_function(final_state:_DensityMatrixType) -> float : 
        observation_mean = np.trace( final_state @ Sp )
        cost = abs(observation_mean)
        return cost
    results = learn_custom_operation(
        num_moments=num_moments, initial_state=cat_state, cost_function=cost_function, operations=operations, max_iter=max_iter, initial_guess=None
    )
    cat_state = results.final_state

    ## Force cat-state to be on the bottom:
    z_projection = np.real(np.trace( cat_state @ Sz ))
    if z_projection>0:
        cat_state = coherent_control.pulse_on_state(cat_state, x=pi)

    ## Aligning with the y axis:
    print("Aligning with the y axis")
    operations = [
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[2]),
    ]
    def cost_function(final_state:_DensityMatrixType) -> float : 
        observation_mean = np.trace( final_state @ Sx @ Sx )
        cost = observation_mean
        return cost
    results = learn_custom_operation(
        num_moments=num_moments, initial_state=cat_state, cost_function=cost_function, operations=operations, max_iter=max_iter, initial_guess=None
    )
    cat_state = results.final_state



    ## Learn GKP:
    # define operation:
    gkp_creation_operations = [
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0]),
        standard_operations.squeezing(),
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0,1,2]),
    ]
    # define cost function:
    target_state = gkp.goal_gkp_state(num_moments=num_moments)
    def cost_function(final_state):
        return (-1) * metrics.fidelity(final_state, target_state)
    gkp_results = learn_custom_operation(
        num_moments=num_moments, 
        initial_state=cat_state, 
        cost_function=cost_function, 
        operations=gkp_creation_operations, 
        max_iter=max_iter, 
        initial_guess=None
    )
    sounds.ascend()
    visuals.plot_city(gkp_results.final_state)
    visuals.draw_now()
    print(f"GKP fidelity is { -1 * gkp_results.score}")
    gkpn_creation_params = gkp_results.theta




    def _trial(x:float) -> _DensityMatrixType:
        trial_state = coherent_control.pulse_on_state(cat_state, x=x)
        _wigner(trial_state, title=f"x={x}")
        return trial_state
        
    trial_state = _trial(0.15)

    mid_placed_state = coherent_control.pulse_on_state(trial_state, x=-pi/2)




def _run_many_guesses(
    min_num_pulses:int=3,
    max_num_pulses:int=16, 
    num_tries:int=5,
    num_moments:int=8
) -> LearnedResults:

    # Track the best results:
    best_results : LearnedResults = LearnedResults(score=1e10) 

    # For movie:
    # target_state  = Fock(num_moments//2).to_density_matrix(num_moments=num_moments)
    coherent_control = CoherentControl(num_moments=num_moments)
    movie_config = CoherentControl.MovieConfig(
        active=True,
        show_now=False,
        num_transition_frames=10,
        num_freeze_frames=5,
        fps=3,
        bloch_sphere_resolution=25,
        score_str_func = lambda state: _score_str_func(state)
    )    
    
    
    for num_pulses in range(min_num_pulses, max_num_pulses+1):
        for _ in range(num_tries):
            # Run:
            try:
                results = creating_gkp_algo(num_pulses=num_pulses, num_moments=num_moments)
            except Exception as e:
                errors.print_traceback(e)
            # Check if better than best:
            if results.score < best_results.score:
                best_results = results
                # Print and record movie:
                print(results)
                coherent_control.coherent_sequence(results.initial_state, theta=results.theta, movie_config=movie_config)
                
    saveload.save(best_results, "best_results "+strings.time_stamp())
    print("\n")
    print("\n")
    print("best_results:")
    print(best_results)
    return best_results

