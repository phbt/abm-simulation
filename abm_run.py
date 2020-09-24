# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:25:34 2020

@author: phili
"""
from abm_functions import * 
from abm_region_class import * 




n_clusters = 5
mean_cluster_size = 300
cluster_sd = 0
# inter_cluster_connections = 0.05 #percent of connections between each cluster_pair

# # cluster_names   = np.array(["Höfen", "Bad", "HH", "Calw"])

cluster_sizes           = np.ceil(np.random.normal(mean_cluster_size, cluster_sd, size = n_clusters)).astype(int) #ensure not smaller than 0
cluster_init_sick       = np.array([30,20,1,2,1])
# cluster_init_healed     = np.array([8,8,8,8,8])
# cluster_init_healed     = np.random.randint(2,10,   size = n_clusters) 

# cluster_init_lockdown   = np.zeros(n_clusters).astype(bool)



# reg = Region(cluster_sizes,cluster_init_sick)
# test_pm = Spread_parameters()

# test_pm_no_tests = Spread_parameters(n_tests = 1)
# df = run(reg, test_pm,alloc_fct=1, animation = True)



# allo_df = compare_allos(reg, test_pm, [(2,np.array([40,40,40,40,40])),(2,np.array([0,0,0,200,0]))], repeats = 4)




high =0.25
medium= 0.08
low = 0.05

connection_intensity_between =     {
          (0, 1) : high,
          (0, 2) : high,
          (0, 3) : high,
          (0, 4) : high,
          (1, 2) : medium,
          (1, 3) : medium,
          (1, 4) : medium,
          (2, 3) : medium,
          (2, 4) : medium,
          (3, 4) : medium     }
####annoying - take care of reverse order of the above dic
for pair in connection_intensity_between.copy():
    c1, c2 = pair
    reversed_key = (c2, c1)
    connection_intensity_between[reversed_key] = connection_intensity_between[pair]

# reg2 = Region(cluster_sizes, cluster_init_sick,inter_cluster_intensities=connection_intensity_between)




# all_poss_alloc = all_possible_allocations(n_clusters, test_pm.n_tests, 50)
# alloc_fct_weight = list(zip(np.ones(len(all_poss_alloc))*2,all_poss_alloc))
# selection= np.random.permutation(alloc_fct_weight)[:15]



# x1 = np.linspace(0,1,num=8)
# x2 = np.flip(x1)


# alloc_fcts_and_weights3 = [(3,{"weight_estim_sick": x1[k], "weight_even":x2[k]}) for k in range(len(x1))]




############test different thresholds with lockdowns with NO tests
if False:
    n_tests4 = 0
    test_pm4 = Spread_parameters(periods = 90, n_tests = n_tests4)
    reg4 = Region(cluster_sizes,cluster_init_sick)
    threshold = np.linspace(0.05, 0.8, num = 10)
    alloc_fcts_and_weights4 = [(4,{"threshold": threshold[k]}) for k in range(len(threshold))]
    allo_df4 = compare_allos(reg4, test_pm4, alloc_fcts_and_weights4, repeats = 4) 
# df4 = run(reg4, test_pm4,alloc_fct=4, alloc_weights = {"threshold" :0.1}, animation = True)
###result: low threshold of 0.5 is best for health, worst for lockdown (so there is a clear tradeoff here)


############
if False:
    n_tests5 = 400 #np.min(cluster_sizes)
    cluster_sizes       = np.array([700,700,700,700,700])
    cluster_init_sick   = np.array([10,10,5,5,5])
    # cluster_init_sick   = np.array([0,0,0,0,0])
    
    rld = 0.01
    
    ld = 0.03


    # n_tests5 = 100
    test_pm5 = Spread_parameters(periods = 60, until_hospital_time=0, prob_hospitalized = 0.2,
                                 ld_threshold =ld, ld_release_threshold =rld, perc_adherence_ld=0.7, 
                                 n_tests = n_tests5)
    
    reg5 = Region(cluster_sizes,cluster_init_sick, symm_inter_cluster_connections=0.3,ws_nei=3, n_temp_contacts=0.5)
    impose_threshold = np.linspace(0.03, 0.2, num = 10)
    release_threshold = np.linspace(0.01, 0.1, num = 10) #so the release threshold grows at the same rate as impose_threshold

    # alloc_fcts_and_weights5 = [(4,{"impose_threshold": impose_threshold[k], "release_threshold":release_threshold[k]}) for k in range(len(release_threshold))]
    alloc_fcts_and_weights5 = [(4,{"impose_threshold": ld, "release_threshold":rld})
                               ] +  [(3,{"weight_estim_sick": 1, "weight_even":0})]

    allo_df5 = compare_allos(reg5, test_pm5, alloc_fcts_and_weights5, repeats = 10) 
    # df5 = run(reg5, test_pm5,alloc_fct=4, alloc_weights = {"impose_threshold" :ld, "release_threshold": rld}, animation = True)
    ########Result: alloc4 yields better results, (symm_inter at 30%) with perfect and non-perfect perc_adherence_ld:
    #    4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.049 0.039
    #    3 {'weight_estim_sick': 1, 'weight_even': 0} 0.049 0.053
    ########### (unlcear ordering)
    #    4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.071 0.052
    #    3 {'weight_estim_sick': 1, 'weight_even': 0} 0.065 0.061

    


if True:
    n_tests5 = 400 #np.min(cluster_sizes)
    cluster_sizes       = np.array([700,700,700,700,700])
    # cluster_init_sick   = np.array([10,10,5,5,5])
    # cluster_init_sick   = np.array([20,5,5,5,3])
    # cluster_init_sick   = np.array([40,5,5,5,3])
    # cluster_init_sick   = np.array([20,0,0,0,0])
    cluster_init_sick   = np.array([10,10,10,10,10])
    
    
    
    # rld = 0.01
    # ld = 0.03
    
    #used for the last runs:
    rld = 0.98
    ld = 0.99


    # n_tests5 = 100
    test_pm5 = Spread_parameters(periods = 60, until_hospital_time=0, prob_hospitalized = 0.2,
                                 ld_threshold =ld, ld_release_threshold =rld, perc_adherence_ld=0.7, 
                                 n_tests = n_tests5)
    
    reg5 = Region(cluster_sizes,cluster_init_sick, symm_inter_cluster_connections=0.05,ws_nei=3, n_temp_contacts=0.5)
    impose_threshold = np.linspace(0.03, 0.2, num = 10)
    release_threshold = np.linspace(0.01, 0.1, num = 10) #so the release threshold grows at the same rate as impose_threshold

    # alloc_fcts_and_weights5 = [(4,{"impose_threshold": impose_threshold[k], "release_threshold":release_threshold[k]}) for k in range(len(release_threshold))]
    alloc_fcts_and_weights5 = [(4,{"impose_threshold": ld, "release_threshold":rld})
                               ] +  [(3,{"weight_estim_sick": 1, "weight_even":0})]

    allo_df5 = compare_allos(reg5, test_pm5, alloc_fcts_and_weights5, repeats = 30) 
    # df5_4 = run(reg5, test_pm5,alloc_fct=4, alloc_weights = {"impose_threshold" :ld, "release_threshold": rld}, animation = True)
    # df5_3 = run(reg5, test_pm5,alloc_fct=3, alloc_weights = {"weight_estim_sick": 1, "weight_even":0}, animation = True)
   
    
    ###########################   Results: 
    # note: until_hospital_time = 0
    ######### cluster_init_sick   = np.array([10,10,5,5,5])
    ### with symm_inter_cluster = 0.05 and perc_adherence_ld = 0.7, the results are very similar:
    #   4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.045 0.037
    #   3 {'weight_estim_sick': 1, 'weight_even': 0} 0.04 0.036
    ### with symm_inter_c = 0.15 and perc_adherence_ld = 0.7 are already favorable for fct4
    #   4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.046 0.039
    #   3 {'weight_estim_sick': 1, 'weight_even': 0} 0.048 0.046
    ### with symm_inter_c = 0.15 and perc_adgerence_ld = 1: also favorable for fct4
    #   4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.053 0.035
    #   3 {'weight_estim_sick': 1, 'weight_even': 0} 0.057 0.047
    ### with symm_inter_c = 0.05 and perc_adherence = 1, results are pretty similar (yet still favorable for allcfct4)
    #   4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.062 0.034
    #   4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.052 0.034 (2nd run)
    #   4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.051 0.032 (3rd run)
    #   3 {'weight_estim_sick': 1, 'weight_even': 0} 0.05 0.037
    #   3 {'weight_estim_sick': 1, 'weight_even': 0} 0.052 0.038 (2nd run)
    #   3 {'weight_estim_sick': 1, 'weight_even': 0} 0.048 0.038 (3rd run)
   
    ###### np.array([20,5,5,5,3]) --> one county markedly higher %sick. 
    ### symm = 0.05, perc_adherence = 0.7. Now, same results for health, fct3 better for ld
    #   4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.048 0.037
    #   4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.037 0.035 (2nd run)
    #   3 {'weight_estim_sick': 1, 'weight_even': 0} 0.038 0.037
    #   3 {'weight_estim_sick': 1, 'weight_even': 0} 0.033 0.036 (2nd run)
    ### symm = 0.15 , perc_adherence = 0.7
    #   4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.055 0.041
    #   3 {'weight_estim_sick': 1, 'weight_even': 0}            0.046 0.037
    ### symm = 0.25, perc_adherence = 0.7
    #   4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.054 0.048
    #   3 {'weight_estim_sick': 1, 'weight_even': 0}            0.046 0.043
    
    #######    cluster_init_sick   = np.array([40,5,5,5,3])
    #   4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.057 0.047
    #   3 {'weight_estim_sick': 1, 'weight_even': 0} 0.064 0.065
    #   4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.051 0.047 (2nd run)
    #   3 {'weight_estim_sick': 1, 'weight_even': 0} 0.059 0.054 (2nd run)
    
    ######     cluster_init_sick   = np.array([20,0,0,0,0])
    ### symm =0.25, perc_adh  = 0.7
    #   4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.018 0.011
    #   3 {'weight_estim_sick': 1, 'weight_even': 0} 0.011 0.009
    #   4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.021 0.012 (2nd run)
    #   3 {'weight_estim_sick': 1, 'weight_even': 0} 0.011 0.008 (2nd run)
    
    ######     cluster_init_sick   = np.array([10,10,10,10,10])
    ### symm =0.25, perc_adh  = 0.7
    #   4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.069 0.051
    #   3 {'weight_estim_sick': 1, 'weight_even': 0} 0.065 0.062
    #   4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.071 0.052 (2nd)
    #   3 {'weight_estim_sick': 1, 'weight_even': 0} 0.076 0.066 (2nd)
    ### symm =0.15, perc_adh  = 0.7
    #   4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.072 0.042
    #   3 {'weight_estim_sick': 1, 'weight_even': 0} 0.062 0.054
    ### symm = 0.05,  perc_adh  = 0.7
    #   4 {'impose_threshold': 0.03, 'release_threshold': 0.01} 0.056 0.039
    #   3 {'weight_estim_sick': 1, 'weight_even': 0} 0.051 0.047
    
    #### symm_inter_cluster_connections=0.05,  cluster_init_sick = np.array([10,10,10,10,10])
    ###   rld = 0.03, ld = 0.08: now 3 better
    #   4 {'impose_threshold': 0.08, 'release_threshold': 0.03} 0.028 0.074
    #   3 {'weight_estim_sick': 1, 'weight_even': 0}            0.022 0.06
    ###   rld = 0.03,  ld = 0.1 still 3 better
    #   4 {'impose_threshold': 0.1, 'release_threshold': 0.03} 0.028 0.074
    #   3 {'weight_estim_sick': 1, 'weight_even': 0} 0.026 0.072
    ###   rld = 0.05, ld = 0.15 now 4 better....
    #   4 {'impose_threshold': 0.15, 'release_threshold': 0.05} 0.023 0.068
    #   3 {'weight_estim_sick': 1, 'weight_even': 0} 0.027 0.081







## df2 = run(reg, test_pm,alloc_fct=1, animation = True)


# allo_df2 = compare_allos(reg, test_pm, alloc_fcts_and_weights3, repeats = 10) #reg and reg2





######urban vs rural
if False:
    reg7 = Region(cluster_sizes, cluster_init_sick,inter_cluster_intensities=connection_intensity_between)
        
    alloc_fcts_and_weights3 = [(3,{"weight_estim_sick": x1[k], "weight_even":x2[k]}) for k in range(len(x1))]

    
    # allo_df3 = compare_allos(reg3, test_pm, alloc_fcts_and_weights3, repeats = 10)
    allo_df7 = compare_allos(reg7, test_pm, alloc_fcts_and_weights4, repeats = 6) #reg and reg2
    # df5 = run(reg3, test_pm,alloc_fct=4, alloc_weights = {"threshold": 0.12}, animation = True)
















