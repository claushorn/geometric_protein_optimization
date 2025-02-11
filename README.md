# Geometric Protein Optimization (GPO)

Preprint:

# Overview

GPO uses AI scoring functions to fine-tune the global geometry of the protein by combining a large number of residue substitutions from all over the protein.

The GPO algorithm BuildUp achieves unprecidented efficiency sinc it enables a form of *discrete gradient ascent* by leveraging the stylized facts of protein optimization we oberve (see paper). 

# Try it out live
<comming soon, stay tuned>

# Choice of scoring function
The current implementation uses the sequence-based BIND model for protein-ligand binding affinity prediction. 
We are working on extending this to more accurate structure-based scoring functions like [GEMS](https://github.com/camlab-ethz/GEMS). 

Do you need to optimize other types of protein function? Let us know! 

# Software setup for local instalation


# Hardware requirements

BuildUp with the BIND model as a reward function runs on a LS40 GPU, but it even works on a M3 Max MacBook Pro with 36GB of memory. 

# License

This project is licensed under the MIT License. It is freely available for academic and commercial use.

# Citation

If you find this resource helpful, please cite the following publication:
