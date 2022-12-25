###===###
# Import some dependencies
import  numpy                   as      np
import  pandas                  as      pd
from    sklearn.preprocessing   import  MinMaxScaler
import  matplotlib.pyplot       as      plt

import 	torch

###===###
# Download our synthetic sepsis dataset from PhysioNet
# and place it in the A000_Inputs folder
Folder  = "/home/nic/Z2023/HealthGymV1.0_Software/A000_Inputs/"
File    = "C001_FakeSepsis.csv"

###===###
# Read in the synthetic sepsis dataset and treat it as the ground truth
MyData = pd.read_csv(Folder+File)

# Drop the unneeded Unnamed: 0 and Timepoints columns
MyData = MyData.drop(["Unnamed: 0", "Timepoints"], axis = 1)

# Rename the columns for our sanity
# 	Admn: 	Administrative purposes
# 	Demo: 	Demographics
# 	Vitl: 	Vital signs/variables
# 	Labs: 	Lab results
# 	Flud: 	Fluid measurements
# 	Vent: 	Ventilation
ColNameSwap = {"PatientID":         "Admn001_ID",
               "Age":               "Demo002_Age",
               "HR":                "Vitl002_HR",
               "SysBP":             "Vitl003_SysBP",
               "MeanBP":            "Vitl004_MeanBP",

               "DiaBP":             "Vitl005_DiaBP",
               "RR":                "Vitl006_RR",
               "Temp":              "Vitl008_Temp",
               "K":                 "Labs001_K",
               "Na":                "Labs002_Na",

               "Cl":                "Labs003_Cl",
               "Ca":                "Labs008_Ca",
               "IonisedCa":         "Labs009_IonisedCa",
               "CO2":               "Labs010_CO2",
               "Albumin":           "Labs014_Albumin",

               "Hb":                "Labs015_Hb",
               "pH":                "Labs021_pH",
               "BE":                "Labs024_BE",
               "HCO3":              "Labs025_HCO3",
               "FiO2":              "Vent002_FiO2",

               "Glucose":           "Labs004_Glucose",
               "BUN":               "Labs005_BUN",
               "Creatinine":        "Labs006_Creatinine",
               "Mg":                "Labs007_Mg",
               "SGOT":              "Labs011_SGOT",

               "SGPT":              "Labs012_SGPT",
               "TotalBili":         "Labs013_TotalBili",
               "WbcCount":          "Labs016_WbcCount",
               "PlateletsCount":    "Labs017_PlateletsCount",
               "PaO2":              "Labs022_PaO2",

               "PaCO2":             "Labs023_PaCO2",
               "Lactate":           "Labs026_Lactate",
               "InputTotal":        "Flud001_InputTotal",
               "Input4H":           "Flud002_Input4H",
               "MaxVaso":           "Flud003_MaxVaso",

               "OutputTotal":       "Flud004_OutputTotal",
               "Output4H":          "Flud005_Output4H",
               "Gender":            "Demo001_Gender",
               "ReAd":              "Demo003_ReAd",
               "Mech":              "Vent001_Mech",

               "GCS":               "Vitl001_GCS",
               "SpO2":              "Vitl007_SpO2",
               "PTT":               "Labs018_PTT",
               "PT":                "Labs019_PT",
               "INR":               "Labs020_INR"
               }

# Perform name swapping
MyData.rename(
    columns = {**ColNameSwap, **{v:k for k,v in ColNameSwap.items()}},
    inplace=True)

###===###
# Create A001_DataTypes.csv to document data property
MyData_Types = pd.DataFrame()

# Including
# 	index: 		--
# 	name:  		--
# 	type:  		Real/binary/categorical
# 	num_classes:	The amount of levels for each variable; fixed 1 for real
# 	embedding_size:	Projection dimension using soft-embeddings
# 	index_start: 	The first variable location in the concatenated features
# 	index_end: 	The pairing last location
MyData_Types["index"]           = []	
MyData_Types["name"]            = []
MyData_Types["type"]            = []
MyData_Types["num_classes"]     = [] 	
MyData_Types["embedding_size"]  = []
MyData_Types["include"]         = []
MyData_Types["index_start"]     = []
MyData_Types["index_end"]       = []

###===###
# Create called A002_MyData.csv to store a machine-readable ground-truth dataset
MyData_Transformed = pd.DataFrame()

# No transformation required for patient ID
MyData_Transformed["Admn001_ID"] = MyData["Admn001_ID"]

# Transformation procedure varies for 
# 	Flt: float
# 	Bin: binary
# 	Cat: categorical

#---
# There are 2 different types of flt variables
# 	N2: Those with Naturally Normal (N2) distributions
# 	LN: Those that can be Logged to become Normal (LN) 
Flt_Variable_N2 = \
[   "Demo002_Age",
    "Vitl002_HR",           "Vitl003_SysBP",        "Vitl004_MeanBP",
    "Vitl005_DiaBP",        "Vitl006_RR",
    "Labs001_K",            "Labs002_Na",           "Labs003_Cl",
    "Labs008_Ca",           "Labs009_IonisedCa",    "Labs010_CO2",
    "Labs014_Albumin",      "Labs015_Hb",           "Labs021_pH",
    "Labs024_BE",           "Labs025_HCO3",
    "Vent002_FiO2"]

Flt_Variable_LN = \
[   "Labs004_Glucose",      "Labs005_BUN",          "Labs006_Creatinine",
    "Labs007_Mg",           "Labs011_SGOT",         "Labs012_SGPT",
    "Labs013_TotalBili",    "Labs016_WbcCount",     "Labs017_PlateletsCount",
    "Labs022_PaO2",         "Labs023_PaCO2",        "Labs026_Lactate",
    "Flud001_InputTotal",   "Flud002_Input4H",      "Flud003_MaxVaso",
    "Flud004_OutputTotal",  "Flud005_Output4H"
    ]

#---
# Bin variables
Bin_Variable = \
[   "Demo001_Gender",      "Demo003_ReAd",
    "Vent001_Mech"
    ]

#---
# There are 2 different types of cat variables
# 	MTC: Those naturally with MulTi-Classes (MTC)
# 	NLN: Those flt-s that canNot be Logged to get Normal distribution (NLN)
Cat_Variable_MTC = \
[   "Vitl001_GCS"
    ]

Cat_Variable_NLN = \
[   "Vitl007_SpO2",         "Vitl008_Temp",
    "Labs018_PTT",          "Labs019_PT",           "Labs020_INR"
    ]

#---
# We need to separately store some back-transform statistics for later use
A001_BTS_Float                  = {}
A001_BTS_Float["Name"]          = []
A001_BTS_Float["min_X0"]        = []
A001_BTS_Float["max_X1"]        = []
A001_BTS_Float["LogNormal"]     = []

A001_BTS_nonFloat               = {}
A001_BTS_nonFloat["Name"]       = []
A001_BTS_nonFloat["Type"]       = []
A001_BTS_nonFloat["Quantiles"]  = []

###===###
# Call the helper function
minmax_scaler = MinMaxScaler()

#---
# For every Flt-N2
for itr in range(len(Flt_Variable_N2)):
    
    #---
    # if this is the first variable
    if itr == 0:
 	# initialise row number and index number in the DataTypes csv
        Cur_Types_Row = 0
        Cur_Index_Row = 0

    # otherwise
    else:
 	# update the row counts
        Cur_Types_Row = list(MyData_Types["index_end"])[-1]

    #---
    # Grab the corresponding variable and numpify it
    Cur_Name = Flt_Variable_N2[itr]
    Cur_Val  = np.array(list(MyData[Cur_Name]))

    #---
    # then document its properties
    # note, 
    # 	num_classes: 	1
    #   embedding_size: 1
    MyData_Types = MyData_Types.\
                   append({"index":             Cur_Index_Row,
                           "name":              Cur_Name,
                           "type":              "real",
                           "num_classes":       1,
                           "embedding_size":    1,
                           "include":           True,
                           "index_start":       Cur_Types_Row,
                           "index_end":         Cur_Types_Row + 1
                           },
                          ignore_index = True
                          )

    #---
    # Document the back-transformation statistics
    # to be transformed into the range of [0, 1]
    A001_BTS_Float["Name"].append(Cur_Name)

    # re-focus the min value to 0
    Temp_Val = torch.tensor(Cur_Val).view(-1)
    A001_BTS_Float["min_X0"].append(Temp_Val.min().item())

    # re-scale the max value to 1
    Temp_Val = Temp_Val - Temp_Val.min()
    A001_BTS_Float["max_X1"].append(Temp_Val.max().item())

    # Flt-N2 do not need to be logged
    A001_BTS_Float["LogNormal"].append(False)

    #---
    # Save the transformed data in the MyData csv
    Cur_Val = minmax_scaler.fit_transform(Cur_Val.reshape(-1, 1))
    MyData_Transformed[Cur_Name] = Cur_Val

    # tic....tick!
    Cur_Index_Row += 1

#---
# Now iterate through every Flt-LN variables
for itr in range(len(Flt_Variable_LN)):

    #---
    Cur_Types_Row = list(MyData_Types['index_end'])[-1]

    #---
    Cur_Name = Flt_Variable_LN[itr]
    Cur_Val  = np.array(list(MyData[Cur_Name]))

    # Logify the variable
    Cur_Val  = np.log(Cur_Val + 1)

    #---
    # Note, 
    # 	num_classes: 	1
    # 	embedding_size: 1
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
    A001_BTS_Float["Name"].append(Cur_Name)
        
    Temp_Val = torch.tensor(Cur_Val).view(-1)
    A001_BTS_Float["min_X0"].append(Temp_Val.min().item())

    Temp_Val = Temp_Val - Temp_Val.min()
    A001_BTS_Float["max_X1"].append(Temp_Val.max().item())

    # Flag the variable as logged
    A001_BTS_Float["LogNormal"].append(True)

    #---
    Cur_Val = minmax_scaler.fit_transform(Cur_Val.reshape(-1, 1))
    MyData_Transformed[Cur_Name] = Cur_Val
        
    # tic....tic....tick!!
    Cur_Index_Row += 1

#---
# Now iterate through all the Bin variables
for itr in range(len(Bin_Variable)):

    #---
    Cur_Types_Row = list(MyData_Types["index_end"])[-1]

    #---
    Cur_Name = Bin_Variable[itr]
    Cur_Val  = np.array(list(MyData[Cur_Name]))

    #---
    # Note,
    # 	num_classes: 	2
    # 	embedding_size: 2
    # 	index_end: 	Cur_Types_Row + 2
    MyData_Types = MyData_Types.\
                   append({"index":             Cur_Index_Row,
                           "name":              Cur_Name,
                           "type":              "bin",
                           "num_classes":       2,
                           "embedding_size":    2,
                           "include":           True,
                           "index_start":       Cur_Types_Row,
                           "index_end":         Cur_Types_Row + 2
                           },
                          ignore_index = True
                          )

    #---
    A001_BTS_nonFloat["Name"].append(Cur_Name)
    A001_BTS_nonFloat["Type"].append("bin")

    # Although Bin are non-numeric,
    # no qunatiles needed here
    A001_BTS_nonFloat["Quantiles"].append({})

    #---
    # Transform the non-numeric variables into a machine-readable version
    # For each availabel level (2 in the case for Bin)
    for itr2 in range(2):
        # Creates a column per level, and
        # suffixify the name with _1 or with _2
        Temp_Name = Cur_Name + '_' + str(itr2)
        
        # If originally of class 1, label 1 in _1, 0 otherwise
        # if originally of class 2, label 1 in _2, 0 otherwise
        Temp_Val  = np.zeros_like(Cur_Val)

	# Find the location of each levels
        Loc_Ele = np.where(Cur_Val == itr2)[0]
 	# Oneify the correct locations
        Temp_Val[Loc_Ele] = 1

	# Save the flagged locations of each level in the machine-readable dataset
        MyData_Transformed[Temp_Name] = Temp_Val
        MyData_Transformed[Temp_Name] = \
            (MyData_Transformed[Temp_Name]).astype(int)

    # tic....tic....tick**2!**3
    Cur_Index_Row += 1

#---
# Now iterate through all Cat_MTCs
for itr in range(len(Cat_Variable_MTC)):

    #---
    Cur_Types_Row = list(MyData_Types['index_end'])[-1]

    #---
    Cur_Name  = Cat_Variable_MTC[itr]
    Cur_Val  = np.floor(np.array(list(MyData[Cur_Name])))

    #---
    # The only MTC is GCS
    # a clinical system designed with minimum 3 and maximum 15 points
    # hence note,
    # 	num_classes: 	13
    # 	embedding_size:  4
    #	index_end: 	Cur_Types_Row + 13
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

    #---
    # No quantiles needed here
    A001_BTS_nonFloat["Quantiles"].append({})

    #---
    # You should know what the following loop is for by now...
    for itr2 in range(3, 16):
        Temp_Name = Cur_Name + '_' + str(itr2) 
        Temp_Val  = np.zeros_like(Cur_Val)

        Loc_Ele = np.where(Cur_Val == itr2)[0]
        Temp_Val[Loc_Ele] = 1

        MyData_Transformed[Temp_Name] = Temp_Val
        MyData_Transformed[Temp_Name] = (MyData_Transformed[Temp_Name]).\
                                        astype(int)
        
    # ( 0_0)/ toc
    Cur_Index_Row += 1

#---
# Iterate through the Cat_NLN variables
# We will bin them according to their 10%, 20%, 30% ... etc values
# hence 10 classes
NLN_classes = 10

for itr in range(len(Cat_Variable_NLN)):

    #---
    Cur_Types_Row = list(MyData_Types['index_end'])[-1]

    #---
    Cur_Name = Cat_Variable_NLN[itr]
    Cur_Val  = np.array(list(MyData[Cur_Name]))

    #---
    # Note,
    # 	num_classes: 	NLN_classes
    # 	embedding_size: 4
    # 	index_end: 	Cur_Types_Row + NLN_classes
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
    # the aforementioned 10%, 20%, 30% ... values
    A001_BTS_nonFloat["Quantiles"].append(
        [np.quantile(Cur_Val, i/NLN_classes) for i in range(NLN_classes + 1)])

    #---
    for itr2 in range(NLN_classes):
        Temp_Name = Cur_Name + '_C' + str(itr2) 
        Temp_Val  = np.zeros_like(Cur_Val)

        Lower_bar = np.quantile(Cur_Val, itr2/NLN_classes)
        Upper_bar = np.quantile(Cur_Val, (itr2+1)/NLN_classes)

        if itr2 == (NLN_classes - 1):
            Upper_bar = Upper_bar * 1.05

        Loc_Ele = np.all( [ [Cur_Val >= Lower_bar],
                            [Cur_Val <  Upper_bar] ],
                          axis = 0)[0]
        Temp_Val[Loc_Ele] = 1

        MyData_Transformed[Temp_Name] = Temp_Val
        MyData_Transformed[Temp_Name] = (MyData_Transformed[Temp_Name]).\
                                        astype(int)
    # ( 0_0 )
    Cur_Index_Row += 1

###===###
# Recalibrate everything one last time for sanity checking
MyData_Types['index']           = (MyData_Types['index']).astype(         int)
MyData_Types['name']            = (MyData_Types['name']).astype(          str)
MyData_Types['type']            = (MyData_Types['type']).astype(          str)
MyData_Types['num_classes']     = (MyData_Types['num_classes']).astype(   int)
MyData_Types['embedding_size']  = (MyData_Types['embedding_size']).astype(int)
MyData_Types['include']         = (MyData_Types['include']).astype(       bool)
MyData_Types['index_start']     = (MyData_Types['index_start']).astype(   int)
MyData_Types['index_end']       = (MyData_Types['index_end']).astype(     int)

# Store the back-transformation statistics
BTS_Folder = "/home/nic/Z2023/HealthGymV1.0_Software/Z001_Data/BTS/"
torch.save(A001_BTS_Float,      BTS_Folder + 'A001_BTS_Float')
torch.save(A001_BTS_nonFloat,   BTS_Folder + 'A001_BTS_nonFloat')

# Store the variable description file
# and the machine-readable transformed ground-truth
Input_Folder = "/home/nic/Z2023/HealthGymV1.0_Software/A000_Inputs/"
MyData_Types.to_csv(        Input_Folder + 'A001_DataTypes.csv', index = False)
MyData_Transformed.to_csv(  Input_Folder + 'A002_MyData.csv',    index = False)



















