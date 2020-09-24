# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""

import igraph as ig
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from itertools import combinations, chain
import pandas as pd
from numba import jit,vectorize, float64




def init_health(cluster_sizes, cluster_init_sick, cluster_init_healed):
    '''function to initialize the "health" vector, which is an n-length np array, indicating the 
    health status for each agent at the beginning of the simulation. This  function is called 
    once to initialize the simuluation. Values are:
    0 = susceptible
    1 = sick
    2 = healed'''
    health_array = np.array([])
    
    for triplet in zip(cluster_sizes,cluster_init_sick, cluster_init_healed):
        size, n_sick, n_healed = triplet
        health_ct = np.random.permutation(np.concatenate((np.ones(n_sick),np.ones(n_healed)*2,np.zeros(size-n_sick-n_healed)))).astype(int)
        health_array = np.append(health_array, health_ct).astype(int)

    return health_array.astype(np.uint8)
    


    
def test_vector(tests_cts, cluster_sizes):
    ''' the input tests_cts indicates the absolute number of tests allocated to each county. 
    This function allocates the tests randomly within each county, and returns a n-length np 
    array with 1/0, with 1s indicating that a test was allocated to that agent.'''
    agents_total = np.sum(cluster_sizes)
    counter = 0
    test_array = np.empty(agents_total)
    for ct_test, c_size in zip(tests_cts, cluster_sizes):     #to do: change that people who have positive test result from previous periods and are in lockdown are not tested anymore
        assert c_size - ct_test >=0, "more tests were allocated to a county than there are people in it"
        county_tested = np.concatenate((np.ones(ct_test),np.zeros(c_size-ct_test)))
        np.random.shuffle(county_tested)
        test_array[counter:counter+c_size] = county_tested
        counter += c_size
    return test_array.astype(np.int8)   


def equal_allo_fct(cluster_sizes, n_total_tests):
    '''allocates tests evenly across counties (regardless of county size). This allocation fct 
    is mainly intended for testing and as a base-case comparison'''
    # print(cluster_sizes, n_total_tests)
    n_tests_ct = round(n_total_tests / cluster_sizes.shape[0])
    tests_cts = np.array(cluster_sizes.shape[0] *[n_tests_ct])

    tests_cts = rounding_fix_allo(tests_cts, n_total_tests)
    
    return tests_cts

def allo_estim_sick_even(cluster_sizes, n_total_tests, estim_sick, weight_estim_sick, weight_even):
    '''allocates tests to county, according to 1) the (absolute) number of sick people in a county and
    2) according to the counties' sizes. The weights determine the relative importance of these two
    factors'''
    if sum(estim_sick) == 0:
        tests_cts = equal_allo_fct(cluster_sizes, n_total_tests)
        return tests_cts
    else:
        weights_estim_sick_cts  = estim_sick*cluster_sizes / np.sum(estim_sick*cluster_sizes)
        weights_even_cts        = cluster_sizes / np.sum(cluster_sizes)
        alloc_weights = (weights_estim_sick_cts * weight_estim_sick + weights_even_cts *weight_even) / (weight_estim_sick + weight_even)
        tests_cts = np.round(alloc_weights * n_total_tests).astype(int)
        tests_cts = rounding_fix_allo(tests_cts, n_total_tests)
        return tests_cts.astype(int)
    


def allo_estim_sick_lockdownNew(cluster_sizes, remaining_cts, estim_sick, n_total_tests):
    '''returns 1) an array indicating how the tests are allocated across counties. Counties with an estimed infection rate
    above the threshold are allocated no tests. These counties are returned as a second vector, and
    should be put in lockdown'''
    
    
    if np.sum(estim_sick[remaining_cts]) != 0: #can't allocate according to %-sick in the remaining counties if they are all estimated to have 0% sick
        weights_estim_sick_cts  = estim_sick[remaining_cts]*cluster_sizes[remaining_cts] / np.sum(estim_sick[remaining_cts]*cluster_sizes[remaining_cts])
        tests_cts = np.round(weights_estim_sick_cts * n_total_tests).astype(int)
        tests_cts = rounding_fix_allo(tests_cts, n_total_tests)
        final_test_allocations=  np.zeros(cluster_sizes.shape[0])
        final_test_allocations[remaining_cts] = tests_cts
        
    elif not np.any(remaining_cts): ##what to do when ALL counties are above the threshold
        final_test_allocations = equal_allo_fct(cluster_sizes, n_total_tests)
    
    else:
        # print(cluster_sizes, remaining_cts)
        final_test_allocations=  np.zeros(cluster_sizes.shape[0])
        final_test_allocations[remaining_cts] = equal_allo_fct(cluster_sizes[remaining_cts], n_total_tests)

    return final_test_allocations.astype(int)
  
    



def rounding_fix_allo(tests_cts, n_total_tests):
    '''helper function to ensure that not too many or too little tests are allocated
    - if there are 100 tests evenly allocated to three counties, this would be 99 tests in total,
    and hence one last test needs to be allocated'''
    while np.sum(tests_cts) > n_total_tests:
        rand_ct = np.random.randint(tests_cts.shape[0])
        if tests_cts[rand_ct] >0: #to avoid picking a county with no tests
            tests_cts[rand_ct] -= 1
    while np.sum(tests_cts) < n_total_tests:
        rand_ct = np.random.randint(tests_cts.shape[0])
        if tests_cts[rand_ct] >0: #to avoid picking a county with no tests; to keep allos clean
            tests_cts[rand_ct] += 1
    return tests_cts

##Numba test
# def hypothetical_infections_slow(indiv_probs):
#     x = np.zeros(len(indiv_probs))
#     for i in range(len(indiv_probs)):
#         x[i] = np.random.random() < indiv_probs[i]
#     return x   
  
# @jit(nopython=True) #this is 100 times faster than the above
# def hypothetical_infections(indiv_probs):
#     x = np.zeros(len(indiv_probs))
#     for i in range(len(indiv_probs)):
#         x[i] = np.random.random() < indiv_probs[i]
#     return x   




def init_toi(sick_initial, healed_initial, sps): 
    ''' function to initialize the time of infection (toi) for those agents that are sick (or healed) 
    at the beginning of the simulation. Healed agents are assigned a toi of -infinity.
    The sick get a toi in line with cure_time. Important: it's assumed that the past infections
    occured evenly spread over time! So no clustering of infections in the time-dimension. There's a bug in there somewhere,
    which is fixed (I think) with the while loop, but not quite sure why this even occurs; to be fixed! '''
    
    toi = np.random.randint(((-sps.cure_time)+1),0, size=len(sick_initial)) #makes sense to use uniform here, but could also use other distribution to reflect infection clustered in the time-dimension.
    toi = np.where(sick_initial ==1, toi, np.inf)
    toi = np.where(healed_initial ==1, -np.inf, toi)
    
    cure_time_vector    = np.ceil(np.random.normal(sps.cure_time, sps.cure_time_sd, len(sick_initial))).astype(np.int8) 
    cure_time_vector    = np.where(cure_time_vector<1,1,cure_time_vector) #to avoid curetimes of smaller than 1
    cure_time_vector[sick_initial] = sps.cure_time #quick and dirty without sd
    
    problem_hits = (sick_initial * (cure_time_vector < (1 - toi)) ).astype(bool) #because then they would already be healed
    while np.any(problem_hits):
        # print(np.sum(problem_hits))
        toi[problem_hits] = toi[problem_hits]+1
        problem_hits = (sick_initial * (cure_time_vector < (1 - toi)) ).astype(bool)
   
    return toi, cure_time_vector


def get_county_sums_from_vector(vector, cluster_sizes):
    '''vector should be 0/1 numpy array of length n_agents. returns a numpy array 
    (length: number of counties) that contains the sum of the quantity in the vector'''
    counter = 0
    sums = np.zeros(len(cluster_sizes))
    
    for i in range(len(cluster_sizes)):
        csize = cluster_sizes[i]
        ct_sum = np.sum(vector[counter:counter+csize])
        sums[i] = ct_sum
        counter += csize
    return sums


def estimation_sick(hospitalized_sums, test_sums, pos_test_sums, cluster_sizes, sps): 
    '''the actual number of sick people is not know to policy makers, hence the number of
    sick people must be estimated ( for instance for lockdown rule).
    Returns the estimated percent of sick people in a county  (not the absolute numbers).
    '''
    #####check whether this makes sense for all of the different uses of the returned quantity!
    
    perc_admitted = sps.prob_hospitalized
    
    all_estimations =[]
    for tests_allocated, pos_tests, hospitalized, csize in zip(test_sums, pos_test_sums, hospitalized_sums, cluster_sizes): 
       
        perc_pop_hospitalized   = hospitalized / csize #the percentage of the county's population that is hospitalized
        perc_sick_estimated_from_hospital = perc_pop_hospitalized / perc_admitted 
        if tests_allocated != 0:
            perc_pos_tests          = pos_tests / tests_allocated
            estimation = (perc_pos_tests * tests_allocated + perc_sick_estimated_from_hospital * hospitalized) / (tests_allocated + hospitalized) 
            #statistically I don't think this is correct, strictly speaking.
            # It's not true that ceteris paribus we get more confident about the sick-percentage estimated from the percentage
            # of hospitalized people. We'd get more certain with the actual number of sick people increasing.. so this is kinda bootstrapped / 
            #inferred. not a big deal, but to be fixed.
        else:
            estimation = perc_sick_estimated_from_hospital
            
        all_estimations.append(estimation)
        
    assert np.all((np.array(all_estimations)>=0)), "one estim is wrong; can't be < 0"
    return np.array(all_estimations)



def lockdown_array(cluster_sizes, cluster_ld, sps):
    '''returns an n-length np-array consisting of 1/0, with 1s indicating that an 
    agent is in a county that is in lockdown '''
    
    ld_array = np.array([])
    
    for ld_bool, size in zip(cluster_ld, cluster_sizes):
        n_adherer = int(sps.perc_adherence_ld * size)
        ld_county = np.random.permutation(np.concatenate((np.ones(n_adherer)*ld_bool,np.zeros(size-n_adherer)*ld_bool)))
        
        ld_array = np.concatenate((ld_array,ld_county))
    return ld_array.astype(int)



def lockdown_counties(estim_sick, counties_in_lockdown, ld_release_threshold, ld_impose_threshold):
    ''' returns a list containing the ids of all counties that have an estimated percentage of 
    sick people above the thresholds (2 thresholds: a lower threshold applies 
    for counties already in lockdown!)'''
    
    lockdown_cts = []
    for ct_estim, ct_ld in zip(estim_sick, counties_in_lockdown):
        if ct_ld:
            if ct_estim  < ld_release_threshold:
                lockdown_cts.append(False)
            else:
                lockdown_cts.append(True)
        else:
            if ct_estim  > ld_impose_threshold:
                lockdown_cts.append(True)
            else:    
                lockdown_cts.append(False)
    return np.array(lockdown_cts)

def all_possible_allocations(n_c, n_tests, batch_size, output_type=None):
    ''' returns all possible allocations of tests, if tests are allocated in a given batch size.
    side-note: fun little combination problem to figure out! My solution is much faster than
    other Python implementions I've seen, and is based on this way of rephrasing the problem: 
    https://brilliant.org/wiki/identical-objects-into-distinct-bins/?fbclid=IwAR2HDScg15AsKlzpXTHhUH4-cERaEjKwXJeHOYj_CF9-4R2pTDBRdCnrsIY
    '''
    assert (n_tests % batch_size == 0), "n_tests needs to be divisible by bucket size"
    n_batches = int(n_tests / batch_size)
    n_bars = n_c - 1
    all_bar_combis = np.array(list(combinations(range(n_bars+n_batches),n_bars)))
    z = np.ones((len(all_bar_combis),1))*(n_bars+n_batches)
    x = np.concatenate((all_bar_combis,z), axis = 1)
    x[:,1:] -= (x[:,:-1]+1)
    x = (x* batch_size).astype(int)
    if output_type == "dic": 
        list_dics = [{key:value for key, value in zip(range(n_c),z)} for z in x]
        return list_dics
    return x

def raw_costs(df,sps,r):
    '''returns a lockdown and a health measure as a proxy for costs
    df as usual a pandas DF returned from the run() function; sps is an instance of the Parameter_spread class,
    r is an instance of the Region class'''
    df2 = pd.wide_to_long(df,stubnames =["health", "countywide_lockdown", "quarantine"], i ="id", j="period")
    df2.reset_index(inplace=True)
    
    df2["lockdown_or_quarantine"] = (df2.countywide_lockdown + df2.quarantine) > 0 
    lockdown_measure = df2.lockdown_or_quarantine.mean()  #includes those who are quarantined because they were sick 
    
    df2["healed"] = df2.health == 2
    df2["sick"] = df2.health == 1
    last_period = sps.periods-1 #-1 because endpoint is not included
    df_last_period = df2[df2.period == last_period]
    n_healed        = df_last_period.healed.sum()
    n_sick          = df_last_period.sick.sum()
    health_measure  = (n_healed + n_sick) / r.n # percentage of people who had or have the disease
    return lockdown_measure, health_measure

def tracing(pos_tests, graph, sps):
    ''' gets all the contacts in the LAST period ONLY, of those people with pos test. Returns a random 
    sps.tracing_success_rate percent of them. Returns an n-length 1/0 array indicating which persons could be traced to have had
    contact with (at least) 1 sick person'''
    nested_neighbors = [(graph.neighbors(vertex)) for vertex in graph.vs(np.nonzero(pos_tests)[0])]
    
    contacts = list(chain(*nested_neighbors))
    n_successes = int(sps.tracing_success_rate*len(contacts))
    
    identified_contacts = list(set(np.random.permutation(contacts)[:n_successes])) #use set to avoid duplicates (contact with multiple sick people))
    
    vector = np.zeros(pos_tests.shape[0])
    if len(identified_contacts)> 0:
        vector[np.array(identified_contacts)] = 1
    return vector
    
def hospital_new_admissions(sick, time_of_infection, t, sps):
    '''returns the new admissions to the hospital (as a np array). Probabilities are independent'''
    prob_hospitalized = sps.prob_hospitalized
    time_gap = sps.until_hospital_time
    n = len(sick)
    eligible_for_hospital = np.where((t-time_of_infection==time_gap), 1,0) *sick #* (1-hospitalized) #*(1-hospitalized) needed if the np.where condition was e.g. "between 3 and 5 days" 
    hypothetical_hospitalized = np.random.choice([1,0], size = n, p = [prob_hospitalized, 1-prob_hospitalized] )
    new_admissions = hypothetical_hospitalized * eligible_for_hospital
    return new_admissions.astype(int)



            
def run(r, sps, alloc_fct, alloc_weights=None, animation = False):
    '''main function to run the simulation'''
    
    graph = r.graph #just as a shorthand
        
    health      = init_health(r.cluster_sizes, r.cluster_init_sick, r.cluster_init_healed)
    healed      = (health == 2).astype(int)
    sick        = (health == 1).astype(int)
    susceptible = np.ones(len(sick))-sick-healed
    n = int(len(health)) #just as shorthand
    time_of_infection, cure_time_vector = init_toi(sick, healed, sps)

    ###get an estimation of sick people for the very beginning (no testing done yet), which relies on %-hospitalized
    hospitalized = (sick * (time_of_infection <= -sps.until_hospital_time) 
                            *np.random.choice([1,0], size = n, p=[sps.prob_hospitalized,1-sps.prob_hospitalized])) 
    
    hospitalized_sums = get_county_sums_from_vector(hospitalized, r.cluster_sizes)
   
    estim_sick =  estimation_sick(hospitalized_sums, 
                                  test_sums     = np.zeros(r.cluster_sizes.shape[0]), 
                                  pos_test_sums = np.zeros(r.cluster_sizes.shape[0]), 
                                  cluster_sizes = r.cluster_sizes, 
                                  sps           = sps)
    
    ### initialize lockdowns and quarantine
    counties_in_lockdown = r.cluster_init_ld.copy()
    countywide_ld_array = lockdown_array(r.cluster_sizes, counties_in_lockdown, sps) 
    quarantine = hospitalized  #so at the start of simulation there were no tests, and hence the only individual quarantined people are those in the hospital
    quarantined_time_except_hospital_quarantine = np.ones(n) *np.inf 
    # ^ this is conceptually confusing, fix this to make code more readable. "quarantine" 
    #... vector includes hospitalized, but vector that contains the time at which agents were put in 
    #... quarantine cannot include the time at which admitted, otherwise hospitalized people may get 
    #... released before they are healed. 
    
    graph.vs["health"]  = health
    graph.vs["toi"] = time_of_infection #note: technically a bit redundant to have both toi and health, but convenient
    
    mobile_array = countywide_ld_array #here, it's just the countywide_ld_array, but in loop it includes quarantine (at start of simulation nobody is quarantined)
   
    if animation:    
        col_dic_clusters    = {i:plt.cm.tab20(i) for i in range(r.n_clusters)} #colors to display the k clusters as distinct
        col_dic_health      = {0: "forestgreen", 1: "red", 2: "blue", 3:"orange"} #orange = newly_infected
        size_dic2           = {0:15,1:20} #agents who are allocated a test are highlighted with larger shape
        shape_dic           = {0:"circle", 1:"rectangle", 2: "triangle-up"} 
        layout              = graph.layout_lgl()
        
        ####### to clean code up, could a) put animation in functions to be called, and b) use dict for visuals
        # animation_specs = {}
        # animation_specs["vertex_color"] = [col_dic_health[cid] for cid in graph.vs["health"]] 
        # animation_specs["vertex_size"]  = [size_dic[v] for v in graph.vs["health"]]
        # animation_specs["edge_color"]   = [col_dic_temp[eid] for eid in graph.es["temporary_edge"]]
        # animation_specs["edge_width"]   = 0.3
        # animation_specs["layout"]       = layout
        
        ig.plot(graph, "1.png",
                vertex_color = [col_dic_clusters[cid] for cid in graph.vs["cluster_ids"]], 
                bbox=(0,0,1300,950), layout=layout )
        
        ig.plot(graph, "2.png",
                vertex_color = [col_dic_health[cid] for cid in graph.vs["health"]] , 
                vertex_size  = 15, #[size_dic[v] for v in graph.vs["health"]],
                vertex_shape = [shape_dic[ld] for ld in mobile_array],
                edge_width   =  0.3,
                bbox=(0,0,1300,950),layout = layout)
    
    df = pd.DataFrame()
    
    ##############  M A I N     L O O P   ################
    for t in range(sps.periods):

        r.change_temporary_contacts() #in first period, first set of temporary edges is created, in the following periods, new temporary edges are created, old ones deleted 
        
        # if animation: #this is just to highlight the temporary edges; usually not useful to include
        #     ig.plot(graph, str(((t*3)))+"temporary_edges.png",
        #     vertex_color = [col_dic_health[cid] for cid in graph.vs["health"]] ,
        #     vertex_size  = [size_dic[v] for v in graph.vs["health"]], 
        #     vertex_shape = [shape_dic[ld] for ld in mobile_array],
        #     edge_color   = [col_dic_temp[eid] for eid in graph.es["temporary_edge"]],
        #     edge_width   =  0.3,         
        #     layout       = layout )
        
        curr_t =str(t)
        df['health'  +curr_t] = health #needs to be updated
        df["countywide_lockdown"+curr_t] = countywide_ld_array
        df["quarantine"+curr_t] = quarantine
        
        
        ########        
        sick_mobile_agents = sick * (1-quarantine) * (1-countywide_ld_array) #this is fine in the current setup because tests are perfect; cannot happen that someone is in a county with ld, and gets free because of false-positive test
        graph.vs["sick+mobile"] = sick_mobile_agents
        
        all_sick_mobile_neigh = np.array([sum([health for health in graph.vs(graph.neighbors(vertex))["sick+mobile"]]) for vertex in graph.vs])  #way too slow. inconceivable that igraph has no quick calculation of sums of neighbors' attributea

        prob_infect_array = 1 - np.power((1-sps.prob_infection),all_sick_mobile_neigh) 
        # new_infection_vector = hypothetical_infections(prob_infection) * susceptible ###not needed; numba is not quicker
        infectious_contact = (prob_infect_array > np.random.random(len(prob_infect_array))).astype(int) 
        new_infection_vector = infectious_contact * susceptible * (1-quarantine) * (1-countywide_ld_array) #if  people who are in quarantine / lockdown can be infected, change this here
              
        new_infected_indices = np.nonzero(new_infection_vector)
        health[new_infected_indices] = 1
        healed_now = ((t - time_of_infection) == cure_time_vector).astype(int)
        healed_now_indices = np.nonzero(healed_now)
        health[healed_now_indices] = 2
        time_of_infection[new_infected_indices]= t
        
        healed      = (health == 2).astype(int) #just as shorthand / convenience, to get more readable code
        sick        = (health == 1).astype(int)
        susceptible = (health == 0).astype(int)
        
        
        #######  HOSPITAL  #############
        vector_admissions_hospital  = hospital_new_admissions(sick, time_of_infection, t, sps)
        vector_released_from_hospital_now = hospitalized * healed_now 
        hospitalized = hospitalized + vector_admissions_hospital - vector_released_from_hospital_now
        hospitalized_sums = get_county_sums_from_vector(hospitalized, r.cluster_sizes)
        #######################
        
        
    
        #####conduct TESTS and administer lockdown ########
        if alloc_fct == 1:    
            tests_cts = equal_allo_fct(r.cluster_sizes, sps.n_tests)
        elif  alloc_fct == 2:
            tests_cts = alloc_weights
        elif alloc_fct == 3:
            weight_estim_sick  = alloc_weights["weight_estim_sick"]
            weight_even        = alloc_weights["weight_even"]
            tests_cts          = allo_estim_sick_even(r.cluster_sizes, sps.n_tests, estim_sick, weight_estim_sick, weight_even)
        elif alloc_fct == 4:
            threshold_release_allofct = alloc_weights["release_threshold"]
            threshold_impose_allofct  = alloc_weights["impose_threshold"]
            counties_in_lockdown = lockdown_counties(estim_sick, counties_in_lockdown, threshold_release_allofct, threshold_impose_allofct)          
            remaining_counties = (1-counties_in_lockdown).astype(bool)
            tests_cts = allo_estim_sick_lockdownNew(r.cluster_sizes, remaining_counties, estim_sick, n_total_tests= sps.n_tests)
          
                
        test_array  = test_vector(tests_cts, r.cluster_sizes)
        pos_tests   = test_array * sick #assumes perfect tests
        neg_tests   = test_array * (1-sick)
        
        pos_test_sums = get_county_sums_from_vector(pos_tests, r.cluster_sizes)
        estim_sick = estimation_sick(hospitalized_sums, tests_cts, pos_test_sums, r.cluster_sizes, sps)

        if alloc_fct != 4: #because alloc_fct 4 determines the lockdowns
            counties_in_lockdown = lockdown_counties(estim_sick, counties_in_lockdown, sps.ld_release_threshold, sps.ld_threshold) #todo: check if location within the loop is fine
        ##################
        
        
        countywide_ld_array = lockdown_array(r.cluster_sizes, counties_in_lockdown, sps) - neg_tests
        countywide_ld_array = np.where(countywide_ld_array<0,0, countywide_ld_array)
        traced_contacts = tracing(pos_tests, graph, sps) 
        quarantined_time_except_hospital_quarantine[np.nonzero(traced_contacts)] = t 
        #note that here it is simply implemented that traced contacts are qurantined, without any tests, even though prob of infection for them is only sps.prob_infection (unless multiple sick-contacts)
        quarantined_time_except_hospital_quarantine[np.nonzero(pos_tests)] = t
        
        release_from_quarantine_after_duration = ((t-quarantined_time_except_hospital_quarantine)==sps.quarantine_duration).astype(int)
        quarantine = quarantine + pos_tests + traced_contacts + vector_admissions_hospital - vector_released_from_hospital_now - release_from_quarantine_after_duration #-neg_tests; do not include here but in countywide_ld
        quarantine = np.where(quarantine<0,0,quarantine)
        quarantine = np.where(quarantine>1,1,quarantine)
        
        
        health[new_infected_indices] = 3 # newly infected == 3, after visualization turned back to 1
        graph.vs["health"] = health
        graph.vs["test"] =test_array
        mobile_array = (countywide_ld_array ==1)
        mobile_array = np.where(quarantine==1,2,mobile_array)
        
        if animation: #to highlight in the animation the agents that were just infected, and those who received tests
            ig.plot(graph, str((t*2)+4)+".png",
                    vertex_color = [col_dic_health[cid] for cid in graph.vs["health"]] , 
                    # vertex_size  = [size_dic[v] for v in graph.vs["health"]], 
                    vertex_size  = [size_dic2[v] for v in graph.vs["test"]],
                    vertex_shape = [shape_dic[ld] for ld in mobile_array],
                    edge_width = 0.3, bbox=(0,0,1300,950), layout = layout )
            
        health[new_infected_indices] = 1
        graph.vs["health"] = health
        
        if animation:
            ig.plot(graph, str((t*2)+5)+".png",
                    vertex_color = [col_dic_health[cid] for cid in graph.vs["health"]] , 
                    vertex_size  = 15, 
                    vertex_shape = [shape_dic[ld] for ld in mobile_array],
                    edge_width   = 0.3, bbox=(0,0,1300,950),layout = layout )           

    df["id"] = list(range(n))
    
    return df

    

def compare_allos(r, sps, alloc_fcts_and_weights, repeats =1, animation = False, verbose=True):
    '''function to compare different allocation functions (and / or the same function with
    different weights). "repeats" indicates the number of repeated runs of 
    each specification, which are then averaged.'''
    
    allo_df = pd.DataFrame(columns=['alloc_tuple', 'mean_lockdown','mean_health'])
    for alloc_tuple in alloc_fcts_and_weights:

        sum_lockdown = 0
        sum_health = 0
        alloc_fct, alloc_weights = alloc_tuple
        
        for k in range(repeats):
            
            df = run(r, sps, alloc_fct, alloc_weights, animation = False)
            
            percent_lockdown, health_measure = raw_costs(df,sps, r)
            sum_lockdown    += percent_lockdown
            sum_health      += health_measure
            
        mean_lockdown = sum_lockdown / repeats
        mean_health = sum_health /  repeats
        new_row = pd.Series({"alloc_tuple": (alloc_fct, alloc_weights), "mean_health" : mean_health, "mean_lockdown": mean_lockdown})
        allo_df = allo_df.append(new_row, ignore_index = True)
        if verbose:
            print(alloc_fct, alloc_weights, round(mean_lockdown,3), round(mean_health,3))
    
    return allo_df

