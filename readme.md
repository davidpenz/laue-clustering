# Laue Clustering 
 
This repository contains the implementation for the paper "Iterative Indexing of Polycrystalline Laue Diffraction Patterns via Cluster-Driven Inference" accepted at ML4EMS@ECML26.

## How to use

First create a conda environment using `conda env create -f environment.yml`.

The experiments can be configured in the `configurations` folder where different materials and number of grains can be selected.

To reproduce the experiments, run `reproduce_experiments/run_all.py`.
