# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Sebastiano Barbieri, UNSW.                                     +
#  All rights reserved. This file is part of the Health Gym, and is released under the   +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#  as part of this package.                                                              +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pickle
from pdb import set_trace as bp

import numpy as np
import pandas as pd
from config.hyperparameters import Hyperparameters as hp
from tqdm import tqdm

if __name__ == "__main__":
    print("Load...")
    dtype = {"icustay_id": "int32", "intime": "str"}
    parse_dates = ["intime"]
    ht_stays = pd.read_csv(
        hp.mimic_dir + "ht_stays.csv",
        usecols=dtype.keys(),
        dtype=dtype,
        parse_dates=parse_dates,
    )
    charts = pd.read_pickle(hp.data_dir + "charts.pkl")
    outputs = pd.read_pickle(hp.data_dir + "outputs.pkl")
    fluid_boluses = pd.read_pickle(hp.data_dir + "fluid_boluses.pkl")
    vasopressors = pd.read_pickle(hp.data_dir + "vasopressors.pkl")

    print("Compute hour of events from ICU admission...")
    charts = pd.merge(charts, ht_stays, on="icustay_id")
    outputs = pd.merge(outputs, ht_stays, on="icustay_id")
    fluid_boluses = pd.merge(fluid_boluses, ht_stays, on="icustay_id")
    vasopressors = pd.merge(vasopressors, ht_stays, on="icustay_id")

    df = pd.concat(
        [
            charts,
            outputs,
            fluid_boluses.rename(columns={"starttime": "charttime", "amount": "value"}),
        ]
    )
    df["hour"] = np.floor(
        (df["charttime"] - df["intime"]) / pd.Timedelta(hours=1)
    ).astype("int")
    df.drop(columns=["charttime", "intime"], inplace=True)

    print("Split vasopressors by hour...")
    df_list = []
    for index, row in tqdm(vasopressors.iterrows(), total=vasopressors.shape[0]):
        starthour = np.floor(
            (row["starttime"] - row["intime"]) / pd.Timedelta(hours=1)
        ).astype("int")
        endhour = np.floor(
            (row["endtime"] - row["intime"]) / pd.Timedelta(hours=1)
        ).astype("int")
        for hour in range(starthour, min(endhour + 1, 48)):
            starthour_time = row["intime"] + pd.DateOffset(hours=hour)
            endhour_time = row["intime"] + pd.DateOffset(hours=hour + 1)
            duration = min(row["endtime"], endhour_time) - max(
                row["starttime"], starthour_time
            )
            value = row["rate"] * (duration / pd.Timedelta(minutes=1))
            df_list.append([row["icustay_id"], row["itemid"], hour, value])
    vasopressors = pd.DataFrame(
        df_list, columns=["icustay_id", "itemid", "hour", "value"]
    )
    df = pd.concat([df, vasopressors], sort=False)

    # remove negative hours (allow -1, clipped to 0)
    df = df[df["hour"] >= -1]
    df["hour"] = df["hour"].clip(lower=0)

    print("Label variables...")
    df.loc[df["itemid"].isin(hp.serum_creatinine_ids), "name"] = "serum_creatinine"
    df.loc[df["itemid"].isin(hp.FiO2_ids), "name"] = "FiO2"
    df.loc[df["itemid"].isin(hp.lactic_acid_ids), "name"] = "lactic_acid"
    df.loc[df["itemid"].isin(hp.ALT_ids), "name"] = "ALT"
    df.loc[df["itemid"].isin(hp.AST_ids), "name"] = "AST"
    df.loc[df["itemid"].isin(hp.systolic_bp_ids), "name"] = "systolic_bp"
    df.loc[df["itemid"].isin(hp.diastolic_bp_ids), "name"] = "diastolic_bp"
    df.loc[df["itemid"].isin(hp.MAP_ids), "name"] = "MAP"
    df.loc[df["itemid"].isin(hp.PO2_ids), "name"] = "PO2"
    df.loc[df["itemid"].isin(hp.GCS_total_ids), "name"] = "GCS_total"
    df.loc[df["itemid"].isin(hp.urine_ids), "name"] = "urine"
    df.loc[df["itemid"].isin(hp.fluid_boluses_ids), "name"] = "fluid_boluses"
    df.loc[df["itemid"].isin(hp.vasopressors_ids), "name"] = "vasopressors"
    df.drop(columns=["itemid"], inplace=True)

    print("Pivot...")
    idx_min = df["name"].isin(["MAP", "systolic_bp", "diastolic_bp"])
    idx_sum = df["name"].isin(["urine", "fluid_boluses", "vasopressors"])
    df_min = df[idx_min].pivot_table(
        index=["icustay_id", "hour"], columns="name", values="value", aggfunc="min"
    )
    df_sum = df[idx_sum].pivot_table(
        index=["icustay_id", "hour"], columns="name", values="value", aggfunc="sum"
    )
    df_avg = df[~(idx_min | idx_sum)].pivot_table(
        index=["icustay_id", "hour"], columns="name", values="value", aggfunc="mean"
    )
    df = pd.concat([df_min, df_sum, df_avg], axis=1, sort=False)

    print("Round...")
    df = df.round(decimals=2)

    print("Save...")
    df.to_pickle(hp.data_dir + "states_actions.pkl")
    df.to_csv(hp.data_dir + "states_actions.csv", index=True)
