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

    # load vasopressors
    print("Load...")
    dtype = {
        "icustay_id": "int32",
        "itemid": "int32",
        "starttime": "str",
        "endtime": "str",
        "rate": "float32",
        "rateuom": "str",
    }
    parse_dates = ["starttime", "endtime"]
    vasopressors = pd.read_csv(
        hp.mimic_dir + "vasopressors.csv",
        usecols=dtype.keys(),
        dtype=dtype,
        parse_dates=parse_dates,
    )

    print("Clean up...")
    # Norepinephrine
    vasopressors.loc[
        (vasopressors["itemid"].isin(hp.norepinephrine_ids))
        & (vasopressors["rate"] > 20),
        "rate",
    ] = (
        vasopressors["rate"] / 80
    )

    # Vasopressin
    vasopressors.loc[
        (vasopressors["itemid"].isin(hp.vasopressin_ids)) & (vasopressors["rate"] > 20),
        "rate",
    ] = (
        vasopressors["rate"] / 60
    )

    # save
    print("Save...")
    vasopressors = vasopressors.drop("rateuom", axis=1).drop_duplicates()
    vasopressors.sort_values(
        by=["icustay_id", "starttime"], ascending=[True, True], inplace=True
    )
    vasopressors.to_pickle(hp.data_dir + "vasopressors.pkl")
    vasopressors.to_csv(hp.data_dir + "vasopressors.csv", index=False)

    # vasopressors = vasopressors[vasopressors['itemid'].isin(hp.epinephrine_ids)]
    # vasopressors.sort_values(by=['rate'], ascending=False, inplace=True)
    # print('------------------')
    # print(vasopressors.head(30))
    # print('------------------')
    # print(vasopressors.describe(include='all'))
    # print('------------------')
    # print(vasopressors['rateuom'].value_counts(dropna=False))
    # print('------------------')
    # fig, ax = plt.subplots()
    # vasopressors['rate'].hist(bins=100, ax=ax)
    # plt.show()
