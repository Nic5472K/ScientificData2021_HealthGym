# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Sebastiano Barbieri, UNSW.                                     +
#  All rights reserved. This file is part of the Health Gym, and is released under the   +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#  as part of this package.                                                              +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pickle
from pdb import set_trace as bp

import matplotlib.pyplot as plt
import pandas as pd
from config.hyperparameters import Hyperparameters as hp

if __name__ == "__main__":

    # load outputs
    print("Load...")
    dtype = {
        "icustay_id": "int32",
        "itemid": "int32",
        "charttime": "str",
        "value": "float32",
        "valueuom": "str",
    }
    parse_dates = ["charttime"]
    outputs = pd.read_csv(
        hp.mimic_dir + "outputs.csv",
        usecols=dtype.keys(),
        dtype=dtype,
        parse_dates=parse_dates,
    )

    # save
    print("Save...")
    outputs = outputs.drop("valueuom", axis=1).drop_duplicates()
    outputs.sort_values(
        by=["icustay_id", "charttime"], ascending=[True, True], inplace=True
    )
    outputs.to_pickle(hp.data_dir + "outputs.pkl")
    outputs.to_csv(hp.data_dir + "outputs.csv", index=False)

    # outputs.sort_values(by=['value'], ascending=False, inplace=True)
    # print('------------------')
    # print(outputs.head(30))
    # print('------------------')
    # print(outputs.describe(include='all'))
    # print('------------------')
    # print(outputs['valueuom'].value_counts(dropna=False))
    # print('------------------')
    # fig, ax = plt.subplots()
    # outputs['value'].hist(bins=100, ax=ax)
    # plt.show()
