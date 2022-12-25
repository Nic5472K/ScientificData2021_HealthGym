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

    # load fluid_boluses
    print("Load...")
    dtype = {
        "icustay_id": "int32",
        "itemid": "int32",
        "starttime": "str",
        "amount": "float32",
        "amountuom": "str",
    }
    parse_dates = ["starttime"]
    fluid_boluses = pd.read_csv(
        hp.mimic_dir + "fluid_boluses.csv",
        usecols=dtype.keys(),
        dtype=dtype,
        parse_dates=parse_dates,
    )

    print("Clean up...")
    # Fresh frozen plasma
    fluid_boluses = fluid_boluses[
        ~(
            fluid_boluses["itemid"].isin(hp.fresh_frozen_plasma_ids)
            & (fluid_boluses["amount"] > 3000)
        )
    ]

    # save
    print("Save...")
    fluid_boluses = fluid_boluses.drop("amountuom", axis=1).drop_duplicates()
    fluid_boluses.sort_values(
        by=["icustay_id", "starttime"], ascending=[True, True], inplace=True
    )
    fluid_boluses.to_pickle(hp.data_dir + "fluid_boluses.pkl")
    fluid_boluses.to_csv(hp.data_dir + "fluid_boluses.csv", index=False)

    # fluid_boluses = fluid_boluses[fluid_boluses['itemid'].isin(hp.platelets_ids)]
    # fluid_boluses.sort_values(by=['amount'], ascending=False, inplace=True)
    # print('------------------')
    # print(fluid_boluses.head(30))
    # print('------------------')
    # print(fluid_boluses.describe(include='all'))
    # print('------------------')
    # print(fluid_boluses['amountuom'].value_counts(dropna=False))
    # print('------------------')
    # fig, ax = plt.subplots()
    # fluid_boluses['amount'].hist(bins=100, ax=ax)
    # plt.show()
