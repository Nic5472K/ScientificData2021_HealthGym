import  numpy                   as      np
import  pandas                  as      pd
import  itertools
import  random
import  yaml

import  torch
import  torch.utils.data        as      utils

###===>>>
def ExecuteB002(
        df, data_types, Hyper001_BatchSize):

    My_Patients = df['Admn001_ID']
    Unique_ID   = np.unique(np.array(list(My_Patients)))
    df = df.drop(['Admn001_ID'], axis = 1)

    Dict_Len2IDs = {}
    Dict_ID2Rows = {}

    ###===###
    for itr in range(My_Patients.shape[0]):

        #---
        Cur_Patient = My_Patients[itr]

        #---
        if Cur_Patient not in Dict_ID2Rows.keys():
            Dict_ID2Rows[Cur_Patient] = [itr]

        else:
            Dict_ID2Rows[Cur_Patient].append(itr)

    ###===###
    for itr in range(len(Dict_ID2Rows)):

        #---
        Cur_Patient = list(Dict_ID2Rows.keys())[itr]
        Len_Patient = len(Dict_ID2Rows[Cur_Patient])

        #---
        if Len_Patient not in Dict_Len2IDs.keys():
            Dict_Len2IDs[Len_Patient] = [Cur_Patient]

        else:
            Dict_Len2IDs[Len_Patient].append(Cur_Patient)

    ###===###
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

    ###===###
    All_Trainable_Data = []

    #---
    for Cur_Key in All_Loader.keys():
        Cur_Loader = All_Loader[Cur_Key]

        #---
        for batch_idx, (x, _) in enumerate(Cur_Loader):
            All_Trainable_Data.append(x)

    #---
    All_Trainable_Data = torch.cat(All_Trainable_Data, dim = 1)

    ###===>>>
    return Dict_Len2IDs, Dict_ID2Rows, All_Loader, All_Trainable_Data    

