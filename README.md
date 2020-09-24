# An Agent-Based-Model for infectious diseases.

It aims to illuminate the spread of infectious diseases in a structured network of agents and the important role of allocating tests in an optimal way. 
The main specification models a region that consists of k connected, heterogenous sub-regions. 
Policy tools that can be activated (and adjusted based on various parameter settings) include testing, test-and-trace, hospitalization and various lockdown schemes. The model illuminates the interplay of the k-subregions, with their heterogeneous policy interventions, and their effect on overall disease-spread. Particular focus lies on
the allocation of tests (coupled with lockdown regimes), and their effect on a) time spent in lockdown and b) health-outcomes (i.e. the number of agents who become infected over the course of a run).

# model architecture
There are two main classes: *Region* is used to construct the contact-structures within a region, which is constituted of 
multiple sub-regions (called counties). The *Spread_parameter* class is used a) specify the characteristics of the disease to be modeled (cure-time, infectiousness etc), 
and b) to determine the parameters for the policy-interventions. 

# Parameters
the parameters for the *Spread_parameter* class:
![Spread-parameters](https://github.com/phbt/abm-simulation/blob/master/parameter_description/parameters1.PNG?raw=true)

the parameters for the *Region* class:
![Region-parameters](https://github.com/phbt/abm-simulation/blob/master/parameter_description/parameters2.PNG?raw=true)
