(NOTE: an explanation for each function and class is contained in their respective docstrings) 

# An Agent-Based-Model for infectious diseases.

The model aims to illuminate the spread of infectious diseases in a structured network of agents, and to highlihht the importance of allocating tests in an optimal way. 
The main specification models a region that consists of k connected, heterogenous sub-regions. 
Policy tools that can be activated (and adjusted based on various parameter settings) include testing, test-and-trace, hospitalization and various lockdown schemes. The model investigates the interplay of the k-subregions, with their heterogeneous policy interventions, and their effect on overall disease-spread. Particular focus lies on
the allocation of tests (coupled with lockdown regimes), and their effect on a) time spent in lockdown and b) health-outcomes (i.e. the number of agents who become infected over the course of a run). The simulation can be visualized.

# Model architecture.
There are two main classes, both are in "abm_region_class". The class *Region* is used to construct the contact-structures within a region, which is constituted of 
multiple sub-regions (which are called counties). The individual counties are small-world networks (Watts-Strogatz type), and there are different ways to create contacts between the counties. The *Spread_parameter* class is used a) to specify the characteristics of the disease to be modeled (cure-time, infectiousness etc), and b) to determine the parameters for the various policy-interventions, like test-and-trace, hospitalization and allocating tests. 

# Parameters.
the parameters for the *Spread_parameter* class:
![Spread-parameters](https://github.com/phbt/abm-simulation/blob/master/parameter_description/parameters1.PNG?raw=true)

the parameters for the *Region* class:
![Region-parameters](https://github.com/phbt/abm-simulation/blob/master/parameter_description/parameters2.PNG?raw=true)
