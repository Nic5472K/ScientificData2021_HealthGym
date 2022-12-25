<pre>
###===### 
# The Health Gym Project (HealthGym)
###===###

###===>>>++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Copyright (c) 2022. by Nicholas Kuo & Sebastiano Babieri, UNSW.                     +
# All rights reserved. This file is part of the Health Gym, and is released under the +
# "MIT Lisence Agreement". Please see the LICENSE file that should have been included +
# as part of this package.                                                            +
###===###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

###===###
Purpose of this Folder:
To demonstrate the necessary inputs to setup the WGAN-GP experiment.

Machine learning experiments can be made complicated on:
 	1) the data pre-processing front and
 	2) the model building front.
Complication 1) is made further complex by the fact that
 	i) you need to enquire access of the data from the data custodian, and
 	ii) that you need to apply inclusion/exclusion criteria on the raw data.
Thus we assembled some codes in this folder to enable you to get a quick feeling of how our repository works.

###===###
Dependencies:
 	numpy, 	        panda, 	        sklearn
 	matplotlib, 	itertools, 	random
 	yaml, 		seaborn, 	os

 	torch

#---
1) We will treat our synthetic sepsis dataset from the Scientific Data paper 
    as the ground truth dataset.
 	- Yes, we will be building another synthetic data on top of an originally synthesised 
 	  dataset to bypass the tedious inclusion/exclusion criteria.
 	- Refer to https://physionet.org/content/synthetic-mimic-iii-health-gym/1.0.0/ ,
 	  download C001_FakeSepsis.csv and place it in A000_Inputs.

2) Preparing the ground truth dataset from human-readable to machine-readable
 	- You will find that C001_FakeSepsis.csv is human readable.
 	- However, the numeric variables require various transformations (e.g., rescaling and  
 	  power transformation);
 	  and the levels in the non-numeric variables cannot be easily processed by the 
 	  WGAN-GP.
 	- Hence we will need to execute Z001_DataPreprocessing.py .
  	- This will generate a data descriptor file A001_DataTypes.csv in folder A000_Inputs;
 	- a transformed ground truth dataset A002_MyData.csv in folder A000_Inputs;
 	- and document back-transformation statistics (BTS) in Z001_Data/BTS/ .
 	- You will find that all binary and categorical variables in A002_MyData.csv now have
 	  one column per level;
 	- most numeric variables are now rescaled in the range of [0, 1]; and
 	- that some numeric variables are now treated as categorical variables.
 	- See more details in the fully commented Z001_DataPreprocesing.py script. 
        
3) Configuring experimental setup
 	- We now configure the various hyperparameters to train the WGAN-GP model.
 	- The configuration file is A003_Configurations.yaml in the A000_inputs folder.
 	- You can find more details on configuring the network in the Academic version of 
 	  our codes.
 	- If you are running the code for the first time, set
 		Epochs: 	100
 		Continue_YN: 	False
 		G_SD: 	B002_G_StateDict_Epoch000
 		D_SD: 		B002_D_StateDict_Epoch000
 		PreEpoch: 	0

4) Running the main file
 	- Now execute the code by running A003_Main.py .
 	- This will take some time and we presume you are using CUDA.

5) Checking the results
 	- Post training, you will find your synthetic dataset stored in
 	  /Z001_Data/Epoch_100/ ;
 	- the parameters of the trained WGAN-GP model will be stored in
 	  /Z002_Parameters/Epoch_100/ ; and
 	- some visual validations for the goodness will be stored in
 	  /Z000_Images/Epoch_100/ .
 
#---
Note:
For sepsis, you should be able to get some fair results from running the model for 100 epochs; 
you can get even better results after 200 epochs.

6) Further train your pre-trained network
 	- To fine-tune your network, head back to A003_Configurtions.yaml and tweak
 		Epochs: 	200
 		Continue_YN: 	True
 		G_SD: 	B002_G_StateDict_Epoch100
 		D_SD: 		B002_D_StateDict_Epoch100
 		PreEpoch: 	100
 	- Caution: do not change the other hyper-parameter settings.
 	  : )
 	- Re-execute the main script A003_Main.py ;
 	- and re-inspect your results in Z000_Image, Z001_Data, and Z002_Parameters.

#---
Further note:
A) This is not a standalone software
 	- If you wish to use your own dataset, you will need to manually modify 
 	  Z001_DataPreprocessing.py .
B) We assume you are generating time series data (e.g., blood glucose level over time)
 	- If you wish to generate static meta-data (e.g., patient height, age, gender),
 	  our code might not be the best for you.
 	- But feel free to add your own post-processing codes, such as to average over all
 	  generated data.
C) All patients in C001_FakeSepsis.csv have 20 units of time series data 
 	- No curriculum learning is at play here.
 	- Refer to the sepsis folder in the Academic version of our codes to learn more 
 	  on CL.

For further enquiries, please contact:
n.kuo@unsw.edu.au
        
###===###
Date updated: 25th of December, 2022 "eji u.4tp6"


