###===>>>++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Copyright (c) 2021. by Nicholas Kuo & Sebastiano Babieri, UNSW.                     +
# All rights reserved. This file is part of the Health Gym, and is released under the +
# "MIT Lisence Agreement". Please see the LICENSE file that should have been included +
# as part of this package.                                                            +
###===###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

###===>>>
# This is the 1st of all files for WGAN on Sepsis

###===>>>
import  pandas                  as      pd
import  numpy                   as      np
from    sklearn.preprocessing   import  MinMaxScaler
import  matplotlib.pyplot       as      plt

import  torch

###===>>>
# (Part 1)      Load the data
data_dir  = "./SepsisData/"
file_name = "ZZZ_Sepsis_Data_From_R.csv"

MyData = pd.read_csv(data_dir + file_name)

#---
# (Part 1.1)    Remove weirdly behaving data points
#               with negative fluid volumes
MyData = MyData.drop([11164, 19127], axis = 0)

###===>>>
# (Part 2)      Creating empty data frames for
#               finalising the data

#---
# (Part 2.1)    The following is for documenting
#               features of the data
MyData_Types = pd.DataFrame()

MyData_Types["index"]           = []
MyData_Types["name"]            = []
MyData_Types["type"]            = []
MyData_Types["num_classes"]     = []
MyData_Types["embedding_size"]  = []
MyData_Types["include"]         = []
MyData_Types["index_start"]     = []
MyData_Types["index_end"]       = []

#---
# (Part 2.2)    The following is for storing
#               the transformed dataset
MyData_Transformed = pd.DataFrame()

MyData_Transformed['Admn001_ID'] = MyData['Admn001_ID']

###===>>>
# (Part 3)      Below, we sort out different types
#               of variables

#---
# (Part 3.1)    Float variables (FLT),
#               those that are "normal" normal (N2) &
#               those that are log normal (LN)
Flt_Variable_N2 = [ "Demo002_Age",
                    "Vitl002_HR", "Vitl003_SysBP", "Vitl004_MeanBP",
                    "Vitl005_DiaBP", "Vitl006_RR", "Vitl007_SpO2",
                    "Vitl008_Temp",
                    "Labs001_K", "Labs002_Na", "Labs003_Cl",
                    "Labs008_Ca", "Labs009_IonisedCa", "Labs010_CO2",
                    "Labs014_Albumin", "Labs015_Hb", "Labs021_pH",
                    "Labs024_BE", "Labs025_HCO3",
                    "Vent002_FiO2"
                    ]

Flt_Variable_LN = [ "Labs004_Glucose", "Labs005_BUN", "Labs006_Creatinine",
                    "Labs007_Mg", "Labs011_SGOT", "Labs012_SGPT",
                    "Labs013_TotalBili", "Labs016_WbcCount",
                    "Labs017_PlateletsCount", "Labs022_PaO2", "Labs023_PaCO2",
                    "Labs026_Lactate",
                    "Flud001_InputTotal", "Flud002_Input4H", "Flud003_MaxVaso",
                    "Flud004_OutputTotal", "Flud005_Output4H"
                    ]

#---
# (Part 3.2)    Binary variables (Bin)
Bin_Variable    = [ "Demo001_Gender", "Demo003_ReAd",
                    "Vent001_Mech"
                    ]

#---
# (Part 3.3)    Categorical variables (Cat),
#               those that are multi-classes (MTC) &
#               those that are not log normal (NLN)
Cat_Variable_MTC = ["Vitl001_GCS"
                    ]

# Note, the following variables are actually floats in nature
# however, they have long tails and do not follow log normalisation nicely
# To address this problem, we will simply treat them as quasi-categorical variables
Cat_Variable_NLN = ["Vitl007_SpO2", "Labs018_PTT", "Labs019_PT",
                    "Labs020_INR"
                    ]

###===>>>
# (Part 4)      Storing some important
#               back-transformation statistics (BTS) down the road
A001_BTS_Float = {}
A001_BTS_Float["Name"]      = []
A001_BTS_Float["min_X0"]    = []
A001_BTS_Float["max_X1"]    = []
A001_BTS_Float["LogNormal"] = [] # <- checks if the variable was log transformed

A001_BTS_nonFloat = {}
A001_BTS_nonFloat["Name"]      = []
A001_BTS_nonFloat["Type"]      = []
A001_BTS_nonFloat["Quantiles"] = [] # <- CAT_NLN will be made into classes by quantiles 


###===>>>
# (Part 5)      Sorting the data

# Minmax will be used for our data transformation
minmax_scaler = MinMaxScaler()

#---
# (Part 5.1)    Sorting out floats with normal distribution
for itr in range(len(Flt_Variable_N2)):
    #---
    if itr == 0:
        Cur_Types_Row = 0
        Cur_Index_Row = 0
    else:
        Cur_Types_Row = list(MyData_Types['index_end'])[-1]

    #---
    Cur_Name = Flt_Variable_N2[itr]
    Cur_Val  = np.array(list(MyData[Cur_Name]))

    #---
    # Appending the data type data frame
    MyData_Types = MyData_Types.\
                   append({'index'          : Cur_Index_Row,
                           'name'           : Cur_Name,
                           'type'           : 'real',
                           'num_classes'    : 1,
                           'embedding_size' : 1,
                           'include'        : True,
                           'index_start'    : Cur_Types_Row,
                           'index_end'      : Cur_Types_Row + 1
                           },
                          ignore_index = True
                          )
    #---
    # Documenting the BTS
    A001_BTS_Float["Name"].append(Cur_Name)
    
    Temp_Val = torch.tensor(Cur_Val).view(-1)
    A001_BTS_Float["min_X0"].append(Temp_Val.min().item())

    Temp_Val = Temp_Val - Temp_Val.min()
    A001_BTS_Float["max_X1"].append(Temp_Val.max().item())

    A001_BTS_Float["LogNormal"].append(False)

    #---
    # Appending the transformed data frame
    Cur_Val = minmax_scaler.fit_transform(Cur_Val.reshape(-1, 1))
    MyData_Transformed[Cur_Name] = Cur_Val

    #---
    Cur_Index_Row += 1
        
#---
# (Part 5.2)    Sorting out floats with log distribution
for itr in range(len(Flt_Variable_LN)):
    #---
    Cur_Types_Row = list(MyData_Types['index_end'])[-1]

    #---
    Cur_Name = Flt_Variable_LN[itr]
    Cur_Val  = np.array(list(MyData[Cur_Name]))

    Cur_Val  = np.log(Cur_Val + 1)

    #---
    MyData_Types = MyData_Types.\
                   append({'index'          : Cur_Index_Row,
                           'name'           : Flt_Variable_LN[itr],
                           'type'           : 'real',
                           'num_classes'    : 1,
                           'embedding_size' : 1,
                           'include'        : True,
                           'index_start'    : Cur_Types_Row,
                           'index_end'      : Cur_Types_Row + 1
                           },
                          ignore_index = True
                          )
    #---
    A001_BTS_Float["Name"].append(Cur_Name)
        
    Temp_Val = torch.tensor(Cur_Val).view(-1)
    A001_BTS_Float["min_X0"].append(Temp_Val.min().item())

    Temp_Val = Temp_Val - Temp_Val.min()
    A001_BTS_Float["max_X1"].append(Temp_Val.max().item())

    A001_BTS_Float["LogNormal"].append(True)

    #---
    Cur_Val = minmax_scaler.fit_transform(Cur_Val.reshape(-1, 1))
    MyData_Transformed[Cur_Name] = Cur_Val
        
    #---
    Cur_Index_Row += 1

#---
# (Part 5.3)    Sorting out the binary variables
for itr in range(len(Bin_Variable)):
    #---        
    Cur_Types_Row = list(MyData_Types['index_end'])[-1]

    #--
    Cur_Name  = Bin_Variable[itr]
    Cur_Val  = np.array(list(MyData[Cur_Name]))

    #---
    MyData_Types = MyData_Types.\
                   append({'index'          : Cur_Index_Row,
                           'name'           : Bin_Variable[itr],
                           'type'           : 'bin',
                           'num_classes'    : 2,
                           'embedding_size' : 2,
                           'include'        : True,
                           'index_start'    : Cur_Types_Row,
                           'index_end'      : Cur_Types_Row + 2
                           },
                          ignore_index = True
                          )
    #---
    A001_BTS_nonFloat["Name"].append(Cur_Name)
    A001_BTS_nonFloat["Type"].append("bin")

    # only CAT_NLNs use quantiles
    A001_BTS_nonFloat["Quantiles"].append({})

    #---
    # store the classes explicitly in separate columns
    # because we will be using "soft embedding" later in our WGAN
    for itr2 in range(2):
        Temp_Name = Cur_Name + '_' + str(itr2) 
        Temp_Val  = np.zeros_like(Cur_Val)

        Loc_Ele = np.where(Cur_Val == itr2)[0]
        Temp_Val[Loc_Ele] = 1

        MyData_Transformed[Temp_Name] = Temp_Val
        MyData_Transformed[Temp_Name] = (MyData_Transformed[Temp_Name]).\
                                        astype(int)
        
    #---
    Cur_Index_Row += 1

#---
# (Part 5.4)    Sorting out the MTC (GCS) variables
for itr in range(len(Cat_Variable_MTC)):
    #---
    Cur_Types_Row = list(MyData_Types['index_end'])[-1]

    #---
    Cur_Name  = Cat_Variable_MTC[itr]
    Cur_Val  = np.floor(np.array(list(MyData[Cur_Name])))

    #---
    MyData_Types = MyData_Types.\
                   append({'index'          : Cur_Index_Row,
                           'name'           : Cat_Variable_MTC[itr],
                           'type'           : 'cat',
                           'num_classes'    : 13,
                           'embedding_size' : 4,
                           'include'        : True,
                           'index_start'    : Cur_Types_Row,
                           'index_end'      : Cur_Types_Row + 13
                           },
                          ignore_index = True
                          )
    #---
    A001_BTS_nonFloat["Name"].append(Cur_Name)
    A001_BTS_nonFloat["Type"].append("GCS")
    A001_BTS_nonFloat["Quantiles"].append({})

    #---
    for itr2 in range(3, 16):
        Temp_Name = Cur_Name + '_' + str(itr2) 
        Temp_Val  = np.zeros_like(Cur_Val)

        Loc_Ele = np.where(Cur_Val == itr2)[0]
        Temp_Val[Loc_Ele] = 1

        MyData_Transformed[Temp_Name] = Temp_Val
        MyData_Transformed[Temp_Name] = (MyData_Transformed[Temp_Name]).\
                                        astype(int)
        
    #---
    Cur_Index_Row += 1

#---
# (Part 5.5)    Sorting out the NLN variable

# we will first define how many classes do we want for the NLNs
NLN_classes = 10

#---
for itr in range(len(Cat_Variable_NLN)):
    #---
    Cur_Types_Row = list(MyData_Types['index_end'])[-1]

    #---
    Cur_Name = Cat_Variable_NLN[itr]
    Cur_Val  = np.array(list(MyData[Cur_Name]))

    #---
    MyData_Types = MyData_Types.\
                   append({'index'          : Cur_Index_Row,
                           'name'           : Cat_Variable_NLN[itr],
                           'type'           : 'cat',
                           'num_classes'    : NLN_classes,
                           'embedding_size' : 4,
                           'include'        : True,
                           'index_start'    : Cur_Types_Row,
                           'index_end'      : Cur_Types_Row + NLN_classes
                           },
                          ignore_index = True
                          )
    #---
    A001_BTS_nonFloat["Name"].append(Cur_Name)
    A001_BTS_nonFloat["Type"].append("cat")
    A001_BTS_nonFloat["Quantiles"].append([np.quantile(Cur_Val, i/NLN_classes) for i in range(NLN_classes + 1)])

    #---
    for itr2 in range(NLN_classes):
        Temp_Name = Cur_Name + '_C' + str(itr2) 
        Temp_Val  = np.zeros_like(Cur_Val)

        # let's find the lower and the upper bounds of the 2 quantiles of interest
        Lower_bar = np.quantile(Cur_Val, itr2/NLN_classes)
        Upper_bar = np.quantile(Cur_Val, (itr2+1)/NLN_classes)

        # technical little difficulty
        if itr2 == (NLN_classes - 1):
            Upper_bar = Upper_bar * 1.05

        # find those desired variables by index
        Loc_Ele = np.all( [ [Cur_Val >= Lower_bar],
                            [Cur_Val <  Upper_bar] ],
                          axis = 0)[0]
        Temp_Val[Loc_Ele] = 1

        MyData_Transformed[Temp_Name] = Temp_Val
        MyData_Transformed[Temp_Name] = (MyData_Transformed[Temp_Name]).\
                                        astype(int)
    #---
    Cur_Index_Row += 1

###===>>>
# (Part 6)      Final type check and save

#---    
torch.save(A001_BTS_Float,    data_dir + 'A001_BTS_Float')
torch.save(A001_BTS_nonFloat, data_dir + 'A001_BTS_nonFloat')

#---
MyData_Types['index']           = (MyData_Types['index']).astype(         int)
MyData_Types['name']            = (MyData_Types['name']).astype(          str)
MyData_Types['type']            = (MyData_Types['type']).astype(          str)
MyData_Types['num_classes']     = (MyData_Types['num_classes']).astype(   int)
MyData_Types['embedding_size']  = (MyData_Types['embedding_size']).astype(int)
MyData_Types['include']         = (MyData_Types['include']).astype(       bool)
MyData_Types['index_start']     = (MyData_Types['index_start']).astype(   int)
MyData_Types['index_end']       = (MyData_Types['index_end']).astype(     int)

MyData_Types.to_csv(data_dir + 'A001_data_types.csv', index = False)

#---
MyData_Transformed.to_csv(data_dir + 'A001_data_real_transformed.csv', index = False)







