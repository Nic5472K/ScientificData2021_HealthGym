###===>>>++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Copyright (c) 2021. by Nicholas Kuo & Sebastiano Babieri, UNSW.                     +
# All rights reserved. This file is part of the Health Gym, and is released under the +
# "MIT Lisence Agreement". Please see the LICENSE file that should have been included +
# as part of this package.                                                            +
###===###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

###===>>>
# This is the 2nd of all files for WGAN on Sepsis

###===>>>
import  numpy                   as      np
import  pandas                  as      pd
import  itertools
import  random

import  torch
import  torch.utils.data        as      utils

#---
from    C003_Utils_Loders       import  *
from    D004_WGAN_GP            import  *
from    C006_Utils_SaveResults  import  *

###===>>>
# (Part 0)      Seed for reproducibility
seed = 0
random.seed(                seed)
np.random.seed(             seed)
torch.manual_seed(          seed)
torch.cuda.manual_seed(     seed)

###===>>>
# (Part 1)      Set up the hyperparameters of this script
#---
# Experimental setup
Hyper001_BatchSize  = 32
Hyper002_Epochs     = 400

#---
# Training setup
Hyper003_G_iter     = 5             # the ratio of critic-generator update
Hyper004_GP_Lambda  = 10            # the regularisation weight of gradient penalty
Hyper005_C_Lambda   = 10            # the regularisation weight of correlation alignment

#---
# Network setup
Hyper006_ID         = 128           # the sampling input dimension (ID) and the generator
Hyper007_HD         = 128           # the hidden dimension (HD) used across the generator & discriminator

#---
# Optimisation setup
Hyper008_LR         = 1e-3          # the learning rate used for both the generator & discriminator
Hyper009_Betas      = (0.9, 0.99)   # the moment coefficients used in the Adam optimiser

#---
# Continue to train a pre-trained variant
Hyper0010_Continue_YN = True
Hyper0011_G_SD        = 'B002_G_StateDict_Epoch300'
Hyper0012_D_SD        = 'B002_D_StateDict_Epoch300'
Hyper0013_PreEpoch    = 300

###===>>>
# (Part 2)      Load the transformed data, and the reference sheet
df = pd.read_csv("./SepsisData/A001_data_real_transformed.csv")

dtype = {
    "index": "int32",
    "name": "str",
    "type": "str",
    "num_classes": "int32",
    "embedding_size": "int32",
    "index_start": "int32",
    "index_end": "int32",
    "include": "bool",
}
data_types = pd.read_csv(
    "./SepsisData/A001_data_types.csv",
    usecols=dtype.keys(),
    dtype=dtype,
    index_col="index",
)
data_types = data_types[data_types["include"]]

###===>>>
# (Part 3)      Prepare the dataloaders for training WGAN
Dict_Len2IDs, Dict_ID2Rows, All_Loader, All_Trainable_Data = \
              Execute_C003(df, data_types, Hyper001_BatchSize)

###===>>>
# (Part 4)      Train the WGAN model
# find the correlation target from the training data
correlation_real = correlation(All_Trainable_Data)

# setup the WGAN GP
wgan_gp = Execute_D004(
            Hyper001_BatchSize, Hyper002_Epochs,
            Hyper003_G_iter, Hyper004_GP_Lambda, Hyper005_C_Lambda,
            Hyper006_ID, Hyper007_HD,
            Hyper008_LR, Hyper009_Betas,
            data_types,
            correlation_real,
            [Hyper0010_Continue_YN,
             Hyper0011_G_SD, Hyper0012_D_SD,
             Hyper0013_PreEpoch])

# and train it
wgan_gp.train(All_Loader)

###===>>>
# (Part 5)      Plot and save the results
# the results will appear in the SepsisImage folder
Execute_C006(wgan_gp, data_types, All_Trainable_Data)




            

