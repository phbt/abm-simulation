# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 11:11:21 2020

@author: phili
"""
import igraph as ig
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from itertools import combinations
import pandas as pd
from numba import jit,vectorize, float64


class Spread_parameters:
    '''class that contains the parameters that determine the spread of the disease, including parameters 
    governing the different policy interventions. 
    An explanation of the various parameters can be found in the accompanying read-me'''
    def __init__(self, 
                 periods                = 40,  #can be thought of as days (if not, parameters in the Region class should likely be adjusted as well)
                 cure_time              = 6, 
                 cure_time_sd           = 2,  
                 prob_infection         = 0.05,  #prob of infection if a contact occurs between sick and susceptible agent. to save ecomputational power, keep perc_edges_realized at 1 and decrease prob_infection instead 
                 prob_hospitalized      = 0.2, 
                 until_hospital_time    = 2, #the time after infection when people are admitted to the hospital. change to be a distribution instead!
                 n_tests                = 300,
                 ld_threshold           = 0.15,
                 ld_release_threshold   = 0.7,
                 #ld_decision_period     = None, #currently not implemented. the duration in periods for which a lockdown decision for a county is valid (not for individuals))
                 quarantine_duration    = 5, #days in quarantine after pos tests - choose wisely; otherwise too many leave quarantine prematurely (because of cure_time + cure_time_sd)
                 tracing_success_rate   = 0.6, 
                 perc_adherence_ld      = 0.9): #percentage of people who adhere to county_wide lockdown. Note: everyone is assumed to adhere to individually ordered quarantine
        
        self.periods                = periods
        self.cure_time              = cure_time
        self.cure_time_sd           = cure_time_sd
        self.prob_infection         = prob_infection
        self.prob_hospitalized      = prob_hospitalized
        self.until_hospital_time    = until_hospital_time
        self.n_tests                = n_tests    
        self.ld_threshold           = ld_threshold
        self.ld_release_threshold   = ld_release_threshold
        #self.ld_decision_period     = ld_decision_period
        self.quarantine_duration    = quarantine_duration
        self.tracing_success_rate   = tracing_success_rate
        self.perc_adherence_ld      = perc_adherence_ld

class Region:
    '''class to initialize the region, by creating its constituting counties. The method "change_temporary_contacts"
    can be called to add within-county-contacts to the graph (n_temp_contacts per agent, in expectation), and previously
    created temporary contacts are removed.
    Note that if inter_cluster_intensities is provided as an input, symm_inter_cluster_connections will
    be ignored.
    All inputs starting with "cluster" have to be numpy arrays of length k, i.e. number of counties.
    Inter-cluster-intensities is a dictionary containing the desired intensities of contacts for each
    county-pair (it suffices to provide a triangular matrix of pairs (i.e. (3,4): 0.15 is enough, no need
    to provide also (4,3): 0.15).
    An explanation of the various parameters can be found in the accompanying read-me.'''
    
    def __init__(self, cluster_sizes, cluster_init_sick, cluster_init_healed=None,
                 cluster_init_ld=None, cluster_names = None, 
                 inter_cluster_intensities=None, symm_inter_cluster_connections=0.1, ws_nei=3, 
                 ws_p = 0.3, n_temp_contacts = 0.5):
        
        self.n_clusters         = len(cluster_sizes)
        self.n                  = np.sum(cluster_sizes)
        self.cluster_sizes      = cluster_sizes.astype(int)
        self.cluster_init_sick  = cluster_init_sick.astype(int)
        self.n_temp_contacts    = n_temp_contacts
        self.cluster_init_healed    = cluster_init_healed.astype(int) if cluster_init_healed is not None else np.zeros(self.n_clusters).astype(int)
        self.cluster_init_ld        = cluster_init_ld if cluster_init_ld is not None else np.zeros(self.n_clusters).astype(bool)
        self.cluster_names          = cluster_names if cluster_names is not None else np.array(["name" + str(i) for i in range(self.n_clusters)])

        if inter_cluster_intensities is not None:
            for pair in inter_cluster_intensities.copy(): #to create reverse order of keys
                c1, c2 = pair
                reversed_key = (c2, c1)
                inter_cluster_intensities[reversed_key] = inter_cluster_intensities[pair]
            
            disjoint_graph = self.create_disjoint_graph(self.cluster_sizes, ws_nei, ws_p)
            self.graph = self.hetero_inter_cluster_connections(disjoint_graph, self.cluster_sizes, inter_cluster_intensities)
        else:
            disjoint_graph = self.create_disjoint_graph(self.cluster_sizes, ws_nei, ws_p)
            self.graph = self.create_inter_cluster_connections(self.cluster_sizes, disjoint_graph, symm_inter_cluster_connections)
            
    def create_disjoint_graph(self, cluster_sizes, ws_nei, ws_p):
        joint = ig.Graph()
        joint_ids = np.array([])
        id_counter = 0
        for size in cluster_sizes:
            h = ig.Graph.Watts_Strogatz(1,size, ws_nei, ws_p)
            joint = ig.Graph.disjoint_union(joint,h)
            joint_ids = np.append(joint_ids,[size * [id_counter]] )
            id_counter += 1
        joint.vs["cluster_ids"]=joint_ids    
        return joint
    
    def create_inter_cluster_connections(self, cluster_sizes, disjoint_graph, symm_inter_cluster_connections):
        n_clusters = len(cluster_sizes)
        pairs = combinations(range(n_clusters),2)
        for pair in pairs:
            s, f = pair
            n_connections = int(((cluster_sizes[s]+cluster_sizes[f])/2)*symm_inter_cluster_connections)
            seq_s = disjoint_graph.vs.select(cluster_ids_eq = s)
            seq_s = np.random.choice(seq_s, size = n_connections, replace = True)
            seq_f = disjoint_graph.vs.select(cluster_ids_eq = f)
            seq_f = np.random.choice(seq_f, size = n_connections, replace = True)
            new_edges = zip(seq_s, seq_f)
            disjoint_graph.add_edges(new_edges)
        disjoint_graph.es["temporary_edge"] = 0
        return disjoint_graph
    
    def hetero_inter_cluster_connections(self, disjoint_graph, cluster_sizes, connection_intensity_between):
        pairs = combinations(range(self.n_clusters),2)
        for pair in pairs:
            s, f = pair
            intensity_pair = connection_intensity_between[pair]
            n_connections  = int(((cluster_sizes[s]+cluster_sizes[f])/2)*intensity_pair )
            seq_s = disjoint_graph.vs.select(cluster_ids_eq = s)
            seq_s = np.random.choice(seq_s, size = n_connections, replace = True)
            seq_f = disjoint_graph.vs.select(cluster_ids_eq = f)
            seq_f = np.random.choice(seq_f, size = n_connections, replace = True)
            new_edges = zip(seq_s, seq_f)
            disjoint_graph.add_edges(new_edges)
        disjoint_graph.es["temporary_edge"] = 0
        return disjoint_graph
    
    def change_temporary_contacts(self):
        n_clusters = len(self.cluster_sizes)
        self.graph.delete_edges(self.graph.es.select(temporary_edge_eq=1))
        for i in range(n_clusters):

            cl_i = np.array([vertex.index for vertex in self.graph.vs.select(cluster_ids_eq =i)])
            k    =np.random.choice(cl_i,size = (int(cl_i.shape[0]*self.n_temp_contacts),2)) 
            
            existing_n_edges = len(self.graph.es)
            self.graph.add_edges(k)
            new_n_edges = len(self.graph.es)
            self.graph.es(range(existing_n_edges,new_n_edges))["temporary_edge"] = 1
            
            self.graph.delete_edges(np.nonzero(self.graph.is_loop())[0]) #deletes loops

    
    