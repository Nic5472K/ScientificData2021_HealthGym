import  numpy                   as      np
import  pandas                  as      pd
import  itertools
import  random
import  matplotlib.pyplot       as      plt

import  torch

ReadFrom        = "./Z001_Data/BTS/"
BST_Float       = torch.load(ReadFrom + "A001_BTS_Float")
BST_nonFloat    = torch.load(ReadFrom + "A001_BTS_nonFloat")

###===>>>
def Execute_C002(Data):

    #---
    Reverted_Data = Data[:, :35]

    for itr in range(35):
        CurVal   = Reverted_Data[:, itr]

        CurMinX0 = BST_Float[   "min_X0"][itr]
        CurMaxX1 = BST_Float[   "max_X1"][itr]
        CurLN    = BST_Float["LogNormal"][itr]

        CurVal = CurVal * CurMaxX1 + CurMinX0

        if CurLN:
            CurVal = torch.exp(CurVal) - 1
            
        if itr == 0:
            CurVal = CurVal / 365.25
            
        Reverted_Data[:, itr] = CurVal
                          
    #---
    Demo001 = Data[:, 35:37]
    Demo003 = Data[:, 37:39]
    Vent001 = Data[:, 39:41]

    _, Demo001 = Demo001.topk(1, dim = 1)
    _, Demo003 = Demo003.topk(1, dim = 1)
    _, Vent001 = Vent001.topk(1, dim = 1)

    Demo001 = Demo001.float()
    Demo003 = Demo003.float()
    Vent001 = Vent001.float()

    #---
    Vitl001 = Data[:, 41: 54]    
    Vitl007 = Data[:, 54: 64]
    Vitl008 = Data[:, 64: 74]
    Labs018 = Data[:, 74: 84]
    Labs019 = Data[:, 84: 94]
    Labs020 = Data[:, 94:104]

    _, Vitl001 = Vitl001.topk(1, dim = 1)
    Vitl001 = Vitl001 + 3
    Vitl001 = Vitl001.float()

    _, Vitl007 = Vitl007.topk(1, dim = 1)
    Vitl007 = Vitl007.float()

    _, Vitl008 = Vitl008.topk(1, dim = 1)
    Vitl008 = Vitl008.float()

    _, Labs018 = Labs018.topk(1, dim = 1)
    Labs018 = Labs018.float()

    _, Labs019 = Labs019.topk(1, dim = 1)
    Labs019 = Labs019.float()

    _, Labs020 = Labs020.topk(1, dim = 1)
    Labs020 = Labs020.float()

    #---
    Reverted_Data = torch.cat(
                        (Reverted_Data,
                         Demo001,
                         Demo003,
                         Vent001,
                         Vitl001,
                         Vitl007,
                         Vitl008,
                         Labs018,
                         Labs019,
                         Labs020
                         ), dim = 1)

    ###===>>>
    return Reverted_Data



