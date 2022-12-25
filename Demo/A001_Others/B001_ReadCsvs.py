import  numpy                   as      np
import  pandas                  as      pd
import  itertools
import  random
import  yaml

import  torch
import  torch.utils.data        as      utils

###===>>>
def ExecuteB001():
    df = pd.read_csv("./A000_Inputs/A002_MyData.csv")
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
        "./A000_Inputs/A001_DataTypes.csv",
        usecols=dtype.keys(),
        dtype=dtype,
        index_col="index",
    )

    ###===>>>
    return df, data_types
