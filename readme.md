# Laue Clustering 
 
This repository contains the impolementation for the ECML submission "Iterative Indexing of Polycrystalline Laue Diffraction Patterns via Cluster-Driven Inference".

## How to use

First create a conda environment using `conda env create -f environment.yml`.

The experiments can be confidgured in the `reproduce_experiments/configurations` folder where different materials and number of grains can be selected.

To reproduce the experiments, run `reproduce_experiments/run_baseline.sh` for the baseline and `reproduce_experiments/run_evaluation.sh` for our approach.
