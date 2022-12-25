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

    # load charts
    print("Load...")
    dtype = {
        "icustay_id": "int32",
        "itemid": "int32",
        "charttime": "str",
        "value": "float32",
        "valueuom": "str",
    }
    parse_dates = ["charttime"]
    charts = pd.read_csv(
        hp.mimic_dir + "charts.csv",
        usecols=dtype.keys(),
        dtype=dtype,
        parse_dates=parse_dates,
    )

    print("Clean up...")
    # FiO2
    # replace percentages with fractions
    charts.loc[
        (charts["itemid"].isin(hp.FiO2_ids)) & (charts["value"] > 1), "value"
    ] = (charts["value"] / 100)
    # remove implausible values
    charts = charts[~((charts["itemid"].isin(hp.FiO2_ids)) & (charts["value"] > 1))]

    # blood pressure (systolic + diastolic + MAP)
    charts = charts[~(charts["itemid"].isin(hp.bp_ids) & (charts["value"] > 250))]

    # GCS (eye, verbal, motor)
    charts_gcs = charts[charts["itemid"].isin(hp.GCS_ids)]
    charts_others = charts[~charts["itemid"].isin(hp.GCS_ids)]
    # get complete scores and sum up eye/verbal/motor components
    complete_gcs_ids = (
        charts_gcs.groupby(["icustay_id", "charttime"])["itemid"].transform("nunique")
        == 3
    )
    charts_gcs = (
        charts_gcs.loc[complete_gcs_ids]
        .groupby(["icustay_id", "charttime"])["value"]
        .sum()
        .reset_index()
    )
    charts_gcs["itemid"] = 990000
    # concatenate back
    charts = pd.concat([charts_others, charts_gcs], sort=False)

    # save
    print("Save...")
    charts = charts.drop("valueuom", axis=1).drop_duplicates()
    charts.sort_values(
        by=["icustay_id", "charttime"], ascending=[True, True], inplace=True
    )
    charts.to_pickle(hp.data_dir + "charts.pkl")
    charts.to_csv(hp.data_dir + "charts.csv", index=False)

    # charts = charts[charts['itemid'].isin(hp.GCS_total_ids)]
    # charts.sort_values(by=['value'], ascending=False, inplace=True)
    # print('------------------')
    # print(charts.head(30))
    # print('------------------')
    # print(charts.describe(include='all'))
    # print('------------------')
    # print(charts['valueuom'].value_counts(dropna=False))
    # print('------------------')
    # fig, ax = plt.subplots()
    # charts['value'].hist(bins=100, ax=ax)
    # plt.show()
