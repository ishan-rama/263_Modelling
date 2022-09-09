# 263_Modelling
ENGSCI 263 - Modelling of Wairakei geothermal subsidence.

This project undertakes a computer modelling study of the subsidence of surface features due to the Wairakei Geothermal System. It will be used
to assist with decision-making during the resource consent hearing, where the applicant: Contact Energy Ltd. is proposing to increase their mass
take from the reservoir.


Main files:
process_data.py - Function library loading and interpolating data.
model_solver.py - Function library for models, ode solvers.
benchmark.py - Function that produces benchmark plots for Model Verification.
callibration.py - Function library that callibrates parameters for pressure and subsidence models
forward_prediction.py - Function library that performs forward predictions on pressure and subsidence models callibrated in calibration.py

Test file:
test_model_solver.py - Unit tests for model_solver.py library


Setup
Text data files are already supplied and processed by process_data.py
Data Files:
sb_disp.txt - Subsidence Data
sb_mass.txt - Mass Extraction Rate data
sb_pres.txt - Pressure Data

Required Packages and Libraries
numpy - General array manipulations
scipy - For curve_fit function
matplotlib - For plotting
decimal - For Exception cases
warnings - Ingore runtime warnings

Usage

To produce all plots across all files, run the main.py file.
There is a display boolean variable that decides whether to save the plots
or display them on your screen. By default display will be set to True, and so 
all plots will be displayed one by one as you close them.

Project Status
Project is: complete / no longer being worked on.



