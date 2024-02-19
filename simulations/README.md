# Simulations

This folder contains all the necessary code to reproduce the simulation results in 
Section 6 and Appendix F.

The code is structured such that you only need to modify the values in `sim_vars.env` to run
the full suite of simulations (or any desired subset) on your machine. See the comments in that file for details
on what variables need to be specified.

We provide implementation details for computing setups both with a slurm machine and without a slurm machine. Note
that without a slurm machine we do not run deep RL methods due to their resource constraints.

### Running on Slurm Machine
1. Copy `simulations/` folder to your machine.
2. Create anaconda environment using the `environment.yml` file.
3. Create `d3rlpy.sif` file using the `d3rlpy` Docker image `docker://takuseno/d3rlpy:latest`. This can be done by running
`singularity pull d3rlpy.sif docker://takuseno/d3rlpy:latest` from your machine. Note that you can replace singularity
with Docker, although this will require modifications to the `deep_rl/slurm_dee_rl.q` and `deep_rl/slurm_dee_rl_run.q`
files.
4. Set the appropriate environment variables in `sim_vars.env`. 
5. Run `sbatch slurm_sim.q` from your machine.

### Running on non-Slurm Machine
1. Copy `simulations/` folder to your machine.
2. Create anaconda environment using the `environment.yml` file.
3. Set the appropriate environment variables in `sim_vars.env`. 
4. Run `bash seq_slurm.sh` from your machine.

*You can choose to run the deep RL methods using the `d3rlpy` Docker image 
`docker://takuseno/d3rlpy:latest`. However, we do not provide implementation details for this.*

Once the results are created and saved, you can use the files in `cleaning_and_plotting/` to summarize and 
analyze the data. `clean_all_results.py` will clean the results. Set the values at the top of this script to match
your system and run with `python clean_all_results.py`. You can then reproduce the plots and metrics reported in the 
paper using `plotting.ipynb` and `metrics.ipynb`.