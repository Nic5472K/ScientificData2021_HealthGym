###===>>>++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Copyright (c) 2021. by Nicholas Kuo & Sebastiano Babieri, UNSW.                     +
# All rights reserved. This file is part of the Health Gym, and is released under the +
# "MIT Lisence Agreement". Please see the LICENSE file that should have been included +
# as part of this package.                                                            +
###===###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

###===>>>
# This is the 3rd of all files for WGAN on Sepsis

###===>>>
import  numpy               as  np
import  pandas              as  pd

import  itertools
import  random

import  torch
import  torch.utils.data    as  utils

###===>>>
def Execute_C003(
        df, data_types, Hyper001_BatchSize):

    ###===###
    # There's a couple of few differences between
    # the Hypotension dataset and the Sepsis dataset
    # the most important feature that influences training
    # is that not all patients have an equal amount of data points
    # this ranges from 1 all the way up to 20
    My_Patients = df['Admn001_ID']
    Unique_ID   = np.unique(np.array(list(My_Patients)))
    df = df.drop(['Admn001_ID'], axis = 1)

    ###===###
    # So it becomes important for us to grab those data
    # of a specific length and furthermore to be able to
    # find patients of a particular length
    Dict_Len2IDs = {}
    Dict_ID2Rows = {}

    ###===>>>
    # (Part 1)      Find the rows of data of each patient
    for itr in range(My_Patients.shape[0]):

        #---
        Cur_Patient = My_Patients[itr]

        #---
        if Cur_Patient not in Dict_ID2Rows.keys():
            Dict_ID2Rows[Cur_Patient] = [itr]

        else:
            Dict_ID2Rows[Cur_Patient].append(itr)

    ###===>>>
    # (Part 2)      Find those patients of the same length
    for itr in range(len(Dict_ID2Rows)):

        #---
        Cur_Patient = list(Dict_ID2Rows.keys())[itr]
        Len_Patient = len(Dict_ID2Rows[Cur_Patient])

        #---
        if Len_Patient not in Dict_Len2IDs.keys():
            Dict_Len2IDs[Len_Patient] = [Cur_Patient]

        else:
            Dict_Len2IDs[Len_Patient].append(Cur_Patient)

    ###===>>>
    # (Part 3)      Preparing the loaders which iterates data of
    #               [BatchSize, Length, Features]
    All_Loader = {}

    #---
    for itr in range(len(Dict_Len2IDs)):

        #---
        Cur_Len = list(Dict_Len2IDs.keys())[itr]

        Temp_DF = pd.DataFrame()
        Cur_IDs = Dict_Len2IDs[Cur_Len]

        #---
        for IndID in Cur_IDs:
            IdRows = Dict_ID2Rows[IndID]

            Temp_DF = Temp_DF.append(df.iloc[IdRows])

        #---
        data = Temp_DF.values

        # This is the part which we reshape the data into the desired shape
        data = data.reshape(
            (-1, Cur_Len, max(data_types["index_end"]))
        )
        
        num_patients = data.shape[0]
        data = utils.TensorDataset(
                torch.from_numpy(data).float(),
                torch.full((num_patients, 1, 1), Cur_Len),
                )
        trn_loader = utils.DataLoader(
            data, batch_size=Hyper001_BatchSize, shuffle=True, drop_last=True
        )

        All_Loader[Cur_Len] = trn_loader

    ###===>>>
    # (Part 4)      All trainable data
    # Below, we will also need to return a complete list of trainable data
    # This is done in order to calculate the correlations among variables
    All_Trainable_Data = []

    #---
    for Cur_Key in All_Loader.keys():
        Cur_Loader = All_Loader[Cur_Key]

        #---
        for batch_idx, (x, _) in enumerate(Cur_Loader):
            All_Trainable_Data.append(x)

    #---
    All_Trainable_Data = torch.cat(All_Trainable_Data, dim = 1)

    ###===###
    return Dict_Len2IDs, Dict_ID2Rows, All_Loader, All_Trainable_Data












    

