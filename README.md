# TwoBrainSystemsGeometricShapes

This is the code repository for the article "Two brain systems for the
perception of geometric shapes". The repository contains the scripts and
functions needed to reproduce all of the analyses and figures presented in the
article.

If you have any question, please send an email to
*mathias.sable-meyer@ucl.ac.uk*.


## Organization of the repository

The repository is organized in four subfolders:

* `./derive_theoretical_RDMs/` contains the code to derive the Representational
  Dissimilarity Matrices for our models, with respective subfolders for
  different models.
* `./online_intruder/` contains the data and analyses script to reproduce our
  analyses of the online intruder task.
* `./fMRI/` contains subfolders for the different analysis steps. In
  particular:
    * `00_behavior` contains all the data and scripts required to analyse the
      behavior data of participants inside the scanner
    * The other folders contain the analysis scripts to reproduce the analysis,
      numbered as follows:
        * `01_` Fitting the GLM to the data
        * `02_` Generating subject-level and group-level reports
        * `03_` Computing cluster-level pvalue with bootstrap, and associated plots
        * `04_` Plots specific to the ventral pathways
        * `05_` Plots of the betas associated to significant clusters
        * `06_` RSA analysis with searchlight
        * `07_` Additional analyses related to the RSA coefficient in significant clusters
* `./MEG/` contains subfolders for the different analysis steps. In particular:
    * `00_behavior` contains all the data and scripts required to analyse the
      behavior data of participants
    * The other folders contain the analysis scripts to reproduce:
        * `01_` The decoding analysis, and its correlation with behavior
        * `02_` The RSA analysis, in `sensor/` and `source/` spaces, and
        * `03_` The ERP analyses

Note that all the behavior data is provided in this repository, but
neuroimaging data has not been shared publicly yet for anonymisation reasons.
When it will be shared, the link to the associated open repositories will be
added to this repository. `fMRI` data provided will be preprocessed, while for
`MEG` data the raw data will be provided for the MEG data itself, but the
preprocessed data will be provided for the anatomical data associated. All of
the data will follow `BIDS` recommendations.

## Related toolboxes

The script assumes you have the following python packages installed, beside the
usual `numpy, matplotlib, etc.`:

* `mne` for the MEG analysis
* `mne-bids-pipeline` for the MEG preprocessing
    * Some decoding analysis were implemented as late steps in the authors'
      working copy of `mne-bids-pipeline`, therefore our version of that
      package is provided as an archive in the `MEG` directory.
* `nilearn` for the fMRI analyses
* `rsatoolbox` for RSA analyses
