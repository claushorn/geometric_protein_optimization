# Geometric Protein Optimization (GPO)

Preprint:

# Overview

GPO uses AI scoring functions to combines a large number of residue substitutions from diverse locations and fine-tune the global geometry of the protein.

The GPO algorithm BuildUp enables discrete gradient ascent. 


# Try it out live:
<stay tuned>

# Choice of reward function and model
The current implementation uses the sequence-based BIND model for protein-ligand binding affinity prediction. 
We are working on extending this to more accurate structure-based scoring functions like [GEMS](https://github.com/camlab-ethz/GEMS). 

# Software setup for local instalation: 


# Hardware requirements

BuildUp with the BIND model as a reward function runs on a LS40 GPU, but it even works on a M3 Max MacBook Pro with 36GB of memory. 

# License

This project is licensed under the MIT License. It is freely available for academic and commercial use.

# Citation

If you find this resource helpful, please cite the following publication:
