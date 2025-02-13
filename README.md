# Geometric Protein Optimization (GPO)

Paper: [https://www.biorxiv.org/content/10.1101/2025.02.10.637504v1](https://www.biorxiv.org/content/10.1101/2025.02.10.637504v1)

# Overview

GPO uses AI scoring functions to fine-tune the global geometry of the protein by combining a large number of residue substitutions from all over the protein.

The GPO algorithm BuildUp achieves unprecidented efficiency by enabling *discrete gradient ascent*, leveraging three stylized facts of protein optimization that we identify (see paper). 

# Try it out live
<comming soon, stay tuned>

# Choice of scoring function
The current implementation uses the sequence-based BIND model for protein-ligand binding affinity prediction. 
We are working on extending this to more accurate structure-based scoring functions like [GEMS](https://github.com/camlab-ethz/GEMS). 

Do you need to optimize other types of protein function? Let us know! 

# Software setup for local instalation
* Installation
  ```
  git clone https://github.com/claushorn/geometric_protein_optimization.git
  ```
* Setup  
  Edit the complex name in config.yaml:
  ```
  protein:
    complex: 3PRS
  ```
  And create a corresponding file for the complex, e.g. '3PRS.yaml' with protein sequence and ligand SMILES:
  ```
  protein: STGSATTTPIDSLDDAYITP...
  ligand: CC(C)[CH](NC(=O)N(C)...
  ```  
* Run protein optimization
  ```
  python buildup.py
  ```

# Hardware requirements

BuildUp with the BIND model as a reward function runs on a LS40 GPU, but it even works on a M3 Max MacBook Pro with 36GB of memory. 

# License

This project is licensed under the MIT License. It is freely available for academic and commercial use.

# Citation

If you find this resource helpful, please cite the following publication:
```
@article {Horn_2025.02.10,
	author = {Wirtz, Dario and Horn, Claus},
	title = {Geometric Protein Optimization},
	year = {2025},
	doi = {10.1101/2025.02.10.637504},
	url = {http://biorxiv.org/lookup/doi/10.1101/2025.02.10.637504},
}
```

