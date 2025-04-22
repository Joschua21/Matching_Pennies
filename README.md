This repo contains code for analyzing photometry and behavior data. 

The code is split into two parts, analyzing photometry and behavior data with or without including GLM-HMM states. 
Generally, the .py files defines functions used in the jupyter notebooks. 
The photometry_analysis.py and dopamine_analysis_pipeline.ipynb files are for analysis without states, glm_hmm_analysis.py/ipynb for analysis including states. 
General parameters like input/output directory or sample rate are defined in the python files. 

## Code Version 
Both python files use pre-computed values for calculations, if available, to increase speed. If you change the way data is handled and thus cannot rely on pre-computed data, increment the CODE_VERSION parameter at the top of the code. This will force re-computation of data. 

## Behavior Dataframe
Option to load a filterd version of the parquet dataframe at the beginning of each notebook, which can be used be downstream function (behavior_df=behvavior_df). Increases speed of computations. 

## Z-scoring: 
Process_session function can either produce z-scored (z_score=True, default) or non-z-scored (z_score=False) data. Both are saved for the respective session in the output directory; non-z-scored is saved with suffix _raw. In any function that calls process_session, define what version (raw or z-scored) to be used. 

## Analysing "All" vs single subjid
Most functions have option to analyze all subjids grouped. Current implementation uses 2 functions, one for analkysing subjid "All", and a _single version of the function if specific subjid is defined. Analysis of "All" uses mean across all subjids.
Input either uses specific_subjects (list of subjids), or uses default list of subjids (JOA-M-0022 to 26). 

For state analysis, functions use a 0.8 threshold of being in each state when sorting trials by state. 

For questions/feedback/comments, please write joschua.geuter.22@ucl.ac.uk
