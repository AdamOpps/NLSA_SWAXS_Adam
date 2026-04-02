# NLSA SWAXS <!-- omit from toc -->

Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Python Environment Setup](#python-environment-setup)
- [Running NLSA from GUI](#running-nlsa-from-gui)
- [More Details](#more-details)

-=- -=- -=- -=- -=- -=- -=- -=- 

# Overview
NLSA_SWAXS is a collection of scripts and associated data files related to Nonlinear Laplacian Spectral Analysis (NLSA) and Small/Wide-Angle X-ray Scattering (SWAXS). This collection provides a GUI-driven tool that:
- Loads a dataset
- Prepares it for processing
- Executes an MPI-based multithreaded computation
- Performs sigma optimization
- Applies diffusion map (DM) dimensionality reduction
- Conducts a parameter search using DM eigenfunctions to refine the NLSA parameters
- Allows viewing of DM eigenfunctions, NLSA chronos, and singular values
- Reconstructs data from selected NLSA modes and saves it to disk.

# Requirements

- Python (recommended: latest version)
- Jupyter Notebook
- Additional dependencies may be required based on the scripts (see notebook headers for details)
Usage

# Python Environment Setup
<!-- ## Linux Setup
1.	Create a folder to download Conda by using the command:
```bash
mkdir -p ~/miniconda3
```
2.	Go into the created folder and download conda into your mortimer usinng below command.
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
```
3.	Run the below commands to setup for the global usage
```bash 
$ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
$ rm -rf ~/miniconda3/miniconda.sh
$ cd ~
$ source .bashrc
```
4. Create conda environment using the below commands
```bash
$ conda create --name <my-env>

Optional: Creating with specific python version
$ conda create --name <my-env> --python=<version>
```
5. Activate conda environment via the command ```conda activate <my-env>``` (to deactivate an active environment, use ```conda deactivate```).

2. Install the NLSA GUI dependencies
   1. numpy, scipy, h5py, and matplotlib:
   ```bash
   $ conda install numpy scipy h5py matplotlib
   ```
   2. MPI:
   ```bash
   $ conda install -c conda-forge mpi4py mpich
   ```
3. Clone the repository:
    ```bash
    git clone https://github.com/UWM-CXFEL/NLSA_SAXS.git
    cd NLSA_SAXS
    ```

## Windows Setup
1. Download Miniconda from https://www.anaconda.com/download.

2. Install the following dependencies from either Anaconda Navigator or Anaconda Terminal.

Using Navigator:
   - Go to Environments.
   - Create a new envitonment for NLSA dependencies. Select your new environment.
   - Click "Channels". Click "Add..." and enter ```conda-forge``` in the text box. Click "Update Channels".
   - Search all channels for the following and install the latest versions: numpy, scipy, h5py, matplotlib, mpi4py.

Using Anaconda Terminal:
1. create conda environment using the below commands
```bash
conda create --name <my-env>

Optional: Creating with specific python version
conda create --name <my-env> --python=<version>
```
-  Activating conda Environment. To activate this environment, use ```conda activate <my-env>``` (to deactivate an active environment, use ```conda deactivate```).

- Install the NLSA GUI dependencies
  -  numpy, scipy, h5py, and matplotlib:
   ```bash
   $ conda install numpy scipy h5py matplotlib
   ```
   - MPI:
   ```bash
   $ conda install -c conda-forge mpi4py mpich
   ```
3. Download the Github repository.

4.  -->
1. Setup a new conda environment.
2. In this new environment, install the following dependencies:
   - numpy
   - scipy
   - h5py
   - matplotlib
   - MPI: mpi4py, mpich (from conda-forge)
3. Clone this Github repository.
4. Create a new environment path variable called ```CXFEL_ROOT``` directing to the root of the cloned repo. Ex. ```~/.../NLSA_GUI```
   - This is required for our implementation of MPI multithreading to have a known, central script directory. Future updates may remove this requirement.
5. **Optional:** If performing *remote computing*, to a cluster or otherwise, Set up X11 forwarding.


# Running NLSA from GUI
To run the NLSA code with SWAXS data using the graphical user interface (described in Sec. 0.3 of the main article), the following steps are required:

1. Navigate to the ```./gui``` directory and run the Python code ```gui_nlsa.py```. This will launch the graphical interface (shown in Fig.7 from the article) where the main window includes functions for parameter search, diffusion map embedding, and NLSA.
   
2. In the “Settings” box, click the “Select Data File” button to browse and load the dataset for the analysis. The dataset must be in ```*.h5``` format.

3. From the data file, specify the “Data Matrix” for the analysis. This matrix should have dimensions $N×D$, where $N$ generally is the number of independent snapshots in a *defined order* (e.g., time delay for time-resolved data), and $D$ represents the number of pixels per snapshot.
   - For TR-SWAXS data in this work, $N$ is the number of time points in SWAXS measurements, and $D$ is the number of q-bins per profile.

4. Enter the desired values for the NLSA parameters in the corresponding fields. These include concatenation ($C$), number of nearest neighbors ($nN$), 
sigma factors ($σ_f$), and the number of diffusion map eigenvalues and eigenfunctions ($nEigs$). The default values are provided as a guideline. The other two parameters are “Data Chunk Size” and “Number of MPI Workers”, which handle multi-threaded computations in the code.

5. Click the “Run Parameter Search” button to start the search. This will run the diffusion map embedding for each specified parameter set, and the results are saved for generating the heatmap. A progress bar will appear to provide a real-time estimate of the computation time for this part of the program. Once the search is complete, an interactive heatmap will appear in the right panel (as displayed in Fig.7 of the article).
      - To generate the heatmap, the Pearson correlation is calculated between the eigenfunctions obtained from each pair of $nN$ and σf specified on the horizontal and vertical axes. A larger number of aligned eigenfunctions indicates a stronger correlation. The colormap reflects the correlation coefficients $\geq 0.90$.

6. Clicking any cell in the interactive heatmap opens a popup window (also in Fig.7 of the article), the parameters for that cell as well as the number of matching eigenfunctions. This window also includes two buttons, each corresponding to a specific set of parameters. Clicking either button will populate the “Selected nN” and “Selected Sigma Factor” fields for further analysis.

7. Press the “Plot DM Eigenfunctions” button to open a new tab visualizing the specified number of eigenfunctions along with their corresponding eigenvalues.

8. Return to the main tab and enter the desired "Number of Copies" required for the NLSA reconstruction. The value of $nCopy$ should be **less than** $C-1$.

9. Click the “Run NLSA” button to execute the remaining steps of the NLSA procedure. Upon completion, a new tab will appear showing singular values and chronos generated by NLSA (see Fig.8 in the article). The results, including U, S, V, and the reconstructed data X_recon, will be stored in a file with the naming format ```usv_nlsa_N{N}_nN{nN}_c{C}_sigma{sigma}_nCopy{nCopy}.h5```, where the brackets are filled with corresponding selected parameters.

10. In this new tab, the user can choose the desired chronos (modes) for the NLSA reconstruction. Selected chronos are outlined in red. Further, the indices of selected chronos are listed in the main tab.
       - The leading modes corresponding to the largest singular values are  usually selected, or if they align with prior knowledge of the dynamical system being studied.

11. Return to the main tab. Press the “Run NLSA Reconstruction” button to perform the final NLSA reconstruction using the selected modes. The code will extract the reconstruction by the specified mode from X_recon (produced in step 1) and write the unwrapped results into a new file with the name format ```data_reconst_N{N}_nN{nN}_c{C}_sigma{sigma}_modes{m1, m2,...}.h5``` in the ```./gui``` directory. Here, m1, m2, ... stand for the selected modes for the NLSA reconstruction. 
    - Note that, except for the final reconstructed data, all files and figures created by this GUI are moved to a folder generated at the start of the runtime within the GUI’s working directory. The naming convention of this folder is ```/temp_data_{YYYYMMDD_HHmmSS}```, where the bracketed string is a timestamp.

# More Details
See the documentation file in the ./doc folder for more implementation details.

Demo data included is simulated Calmodulin (1CFD to 2BBM) time series data, including clean and noisy versions of both the "measured" intensity, and the difference profiles. 
<!-- CaM: 1E5 photons per pulse, 1kHz rep rate, 10 sec exposure per frame, NO jitter or pulse broadening, replicate 3 (of 5) -->
