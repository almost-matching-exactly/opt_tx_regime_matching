# Supplementary Material for Safe and Interpretable Estimation of Optimal Treatment Regimes

We include the files containing results from all synthetic simulations in the `simulations_data/` folder.
The contents of each file is outlined in Appendix F.3.

We also include all the code used to create the results in our AISTATS submission titled 
"Safe and Interpretable Estimation of Optimal Treatment Regimes." The code is broken up into two folders: 
`simulations/` and `real_world/`.

`simulations/` contains the code necessary to recreate the simulation results found in Section 6 and
Appendix F. See details in `simulations/README.md`.

`real_world/` contains the code used to produce the results in Section 7. Note that the data used in this section
can not be shared and thus we do not include it.

You can find an implementation of our method in the Python class
`OptMaTx` inside the `simulation/methods/opt_matching.py` file .