###===>>>++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Copyright (c) 2022. by Nicholas Kuo & Sebastiano Babieri, UNSW.                     +
# All rights reserved. This file is part of the Health Gym, and is released under the +
# "MIT Lisence Agreement". Please see the LICENSE file that should have been included +
# as part of this package.                                                            +
###===###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

###===>>>
import  numpy                   as      np
import  pandas                  as      pd
import  itertools
import  random
import  yaml

import  torch
import  torch.utils.data        as      utils

###===>>>
MyConfig_Stream = open('./A000_Inputs/A003_Configurations.yaml', 'r')
MyConfig = yaml.load(MyConfig_Stream, Loader = yaml.FullLoader)

###===>>>
seed = MyConfig["seed"]
random.seed(                seed)
np.random.seed(             seed)
torch.manual_seed(          seed)
torch.cuda.manual_seed(     seed)

###===>>>
Hyper001_BatchSize      =        MyConfig["BatchSize"]
Hyper002_Epochs         =        MyConfig["Epochs"]
Hyper003_G_iter         =        MyConfig["G_iter"]
Hyper004_GP_Lambda      =        MyConfig["GP_Lambda"]
Hyper005_C_Lambda       =        MyConfig["C_Lambda"]
Hyper006_ID             =        MyConfig["ID"]
Hyper007_HD             =        MyConfig["HD"]
Hyper008_LR             = float( MyConfig["LR"]             )
Hyper009_Betas          = tuple( MyConfig["Betas"]          )
Hyper0010_Continue_YN   =        MyConfig["Continue_YN"]
Hyper0011_G_SD          =        MyConfig["G_SD"]
Hyper0012_D_SD          =        MyConfig["D_SD"]
Hyper0013_PreEpoch      =        MyConfig["PreEpoch"]

###===>>>
from A001_Others.B001_ReadCsvs import  *
df, data_types = \
    ExecuteB001()

###===>>>
from A001_Others.B002_UtilsLoaders import  *
Dict_Len2IDs, Dict_ID2Rows, All_Loader, All_Trainable_Data = \
    ExecuteB002(df, data_types, Hyper001_BatchSize)

###===>>>
from A001_Others.B003_WganGp import  *
wgan_gp = \
    ExecuteB003(
        All_Trainable_Data,
        Hyper001_BatchSize, Hyper002_Epochs,
        Hyper003_G_iter, Hyper004_GP_Lambda, Hyper005_C_Lambda,
        Hyper006_ID, Hyper007_HD,
        Hyper008_LR, Hyper009_Betas,
        data_types,
        [Hyper0010_Continue_YN,
         Hyper0011_G_SD, Hyper0012_D_SD,
         Hyper0013_PreEpoch]
        )

#---
wgan_gp.train(All_Loader)

###===>>>
from A001_Others.B004_SaveData   import  *
from A001_Others.B005_SaveImages import  *
from A001_Others.B006_SaveParams import  *
df_fake = ExecuteB004(data_types, wgan_gp, Hyper002_Epochs)
ExecuteB005(wgan_gp, data_types, All_Trainable_Data, df_fake, Hyper002_Epochs)
ExecuteB006(wgan_gp, Hyper002_Epochs)




























