# FindingCRDs
Counter-Rotating Disk galaxy finder using MaNGA MAPS generated from the MaNGA DAP. 

# Instructions for running the FindingCRDs pipeline.

First, cd into where /FindingCRDs_Pipeline/ is located. Create a conda environment to keep everything happy and activate it. Then, run the pip install command on requirements.txt. This will install all needed python packages and the mangadap. Example code for this, in a terminal, is:
--
cd filepath/to/FindingCRDs_Pipeline/
conda create -n manga python=3.9 -y
conda activate manga

conda install -c conda-forge cvxopt -y

pip install sdss-mangadap==4.1.1

pip install -r requirements.txt


After everything is installed, we are ready to start. First we must set each process to only use a single thread. Then we can run the master script. Example code for this, in the same terminal window, is:
--
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_pipeline.py


This will run through all 3 steps: download CUBE files, run the MaNGA DAP with moments=4, then apply FindingCRDs.py. The final result will be a folder, FindingCRDs_Results. Its content is:
--
- /maybe_CRD/ will have "[plate]-[ifu].txt" with the boolean condition of the CRD checks (3V, 4V, 2sigma), as well as an overarching "maybe_CRD.txt" with all the [plate]-[ifu]s that have signs of counter-rotation.

- /no_CRD/ will have a "no_CRD.txt" with all the [plate]-[ifu]s that did not have signs of counter-rotation.

- "analyzed_galaxies.txt" is a text file of all galaxies that have been analyzed.

- "problems.txt" is a text file of all galaxies that FindingCRDs.py could not analyze.
--
