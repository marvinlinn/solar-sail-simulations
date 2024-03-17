# Solar Sail Simulation

Simulation of solar sail trajectories under Prof. Pister at the University of California: Berkeley for the Berkeley Low-cost Interplanetary Solar Sail (BLISS) project.

## Getting Started

Before running this notebook, please install `conda` as a python environment manager. If you're at this point, I'll assume you also have `jupyter lab` and/or  `jupyter notebook` installed.

To get the appropriate packages to run this notebook started, please first install a new conda environment through the configuration YAML in the repository. Use the following command while in the repository directory:

```
conda env create -f environment.yml
```

This will install the following dependancies into your a new conda environment called `solar-sail-sim`:

- numpy
- ipykernel
- matplotlib
- scipy
- spiceypy

To create a Python kernel for Jupyter to get access to, run the following command:

```
ipython kernel install --user --name=solar-sail-sim
```

Now you're all set!

## Contribution Guidelines

Make sure to commit regularly, and please make sure to **export a new YAML configuration file if you install new packages** to the python environment.

To install new python packages to the environment, please use use the following:

```
conda install -n solar-sail-sim <PACKAGE NAME>
```

To update the YAML configuration file, use the following while in the repository directory AND in the correct conda environment:

```
conda env export > solar-sail-sim.yml
```
## Student Contributers
**Spring 2024 (v0.1)** Marvin Lin, Andrew Ji, Shazaib Lalani, Matthew Cranny, Luke Harris