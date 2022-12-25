# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Sebastiano Barbieri, UNSW.                                     +
#  All rights reserved. This file is part of the Health Gym, and is released under the   +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#  as part of this package.                                                              +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pickle
import sys
from pdb import set_trace as bp

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lib.utils import save_obj
from omegaconf import DictConfig
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def custom_round(x, base=0.1):
    return base * round(x.astype(float) / base)


@hydra.main(config_path="config", config_name="config")
def main(hp: DictConfig):
    print("Load...")
    df = pd.read_pickle(hp.dir.data + "states_actions.pkl")

    print("Re-index...")
    ## add rows (hours) where no measurements were taken, up until max hour of stay
    # s_index = df.reset_index().groupby('icustay_id')['hour'].max()
    # icustay_ids = np.concatenate([np.repeat(index, value + 1) for index, value in s_index.items()])
    # hours = np.concatenate([np.arange(0, value + 1) for index, value in s_index.items()])
    # idx = pd.MultiIndex.from_arrays([icustay_ids, hours], names=['icustay_id', 'hour'])

    # Alternative: fixed length of hours for all stays
    idx = pd.MultiIndex.from_product(
        [
            df.index.unique(level=0).sort_values(),
            df.index.unique(level=1).sort_values(),
        ],
        names=["icustay_id", "hour"],
    )
    df = df.reindex(idx).reset_index()
    df.index.name = df.columns.name = None

    print("Measurement columns...")
    df["urine_m"] = df["urine"].notna().astype("int")
    df["ALT_AST_m"] = (df["ALT"].notna() | df["AST"].notna()).astype(
        "int"
    )  # usually measured together
    df["FiO2_m"] = df["FiO2"].notna().astype("int")
    df["GCS_total_m"] = df["GCS_total"].notna().astype("int")
    df["PO2_m"] = df["PO2"].notna().astype("int")
    df["lactic_acid_m"] = df["lactic_acid"].notna().astype("int")
    df["serum_creatinine_m"] = df["serum_creatinine"].notna().astype("int")

    print("Fill missing values...")
    df[["fluid_boluses", "vasopressors"]] = df[
        ["fluid_boluses", "vasopressors"]
    ].fillna(0)
    median_values = df.median()
    df.update(df.groupby(["icustay_id"]).fillna(method="ffill"))
    df.update(df.groupby(["icustay_id"]).fillna(method="bfill"))
    df = df.fillna(median_values)

    print("Bin categorical columns...")
    # labels = ['none', 'low', 'medium', 'high']
    bins_fluid_boluses = [0, 250, 500, 1000, 1e6]
    df["fluid_boluses"] = pd.cut(
        df["fluid_boluses"],
        right=False,
        bins=bins_fluid_boluses,
        labels=bins_fluid_boluses[:-1],
    )
    print(df["fluid_boluses"].value_counts())

    bins_vasopressors = [0, 1e-6, 8.4, 20.28, 1e6]
    df["vasopressors"] = pd.cut(
        df["vasopressors"],
        right=False,
        bins=bins_vasopressors,
        labels=bins_vasopressors[:-1],
    )
    print(df["vasopressors"].value_counts())

    bins_FiO2 = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    df["FiO2"] = pd.cut(df["FiO2"], right=False, bins=bins_FiO2, labels=bins_FiO2[:-1])
    print(df["FiO2"].value_counts())

    df["GCS_total"] = round(df["GCS_total"]).astype(int)
    print(df["GCS_total"].value_counts())

    print("Save...")
    df.to_pickle(hp.dir.data + "data_real.pkl")
    df.to_csv(hp.dir.data + "data_real.csv", index=False)

    print("Load data types...")
    dtype = {
        "index": "int32",
        "name": "str",
        "type": "str",
        "num_classes": "int32",
        "embedding_size": "int32",
        "include": "bool",
    }
    data_types = pd.read_csv(
        hp.dir.config + "data_types.csv",
        usecols=dtype.keys(),
        dtype=dtype,
        index_col="index",
    )
    data_types["index_start"] = 0
    data_types["index_end"] = 0
    id_real = data_types["type"] == "real"
    data_types = pd.concat(
        [data_types[id_real], data_types[~id_real]], ignore_index=True
    )

    print("Keep included variables...")
    included_columns = data_types.loc[data_types["include"], "name"].to_list()
    df = df[["icustay_id", "hour"] + included_columns]

    print("Transform columns...")
    current_index = 0
    transforms_dict = {}
    plotted = False
    for index, row in data_types.iterrows():
        if row["include"]:
            # real transforms
            if row["type"] == "real":
                df[row["name"]], lambda_bc = boxcox(df[row["name"]] + 1)
                minmax_scaler = MinMaxScaler(feature_range=(0, 1))
                df[row["name"]] = minmax_scaler.fit_transform(
                    df[row["name"]].values.reshape(-1, 1)
                )
                transforms_dict[row["name"]] = {
                    "lambda_bc": lambda_bc,
                    "minmax_scaler": minmax_scaler,
                }
            # plot distributions
            if row["type"] != "real" and not plotted:
                print("Plot distributions...")
                fig, ax = plt.subplots(
                    ncols=sum(data_types["include"]), nrows=1, figsize=(40, 20)
                )
                for i_plt, r_plt in data_types.iterrows():
                    if r_plt["include"]:
                        ax[i_plt].hist(df[r_plt["name"]].values, bins=100, density=True)
                        ax[i_plt].title.set_text(r_plt["name"])
                fig.savefig(hp.dir.data + "distributions.png")
                plotted = True
            # dummify categorical
            if row["type"] == "cat" or row["type"] == "bin":
                columns_before = df.columns
                df = pd.get_dummies(df, prefix=row["name"], columns=[row["name"]])
                columns_after = df.columns
                dummy_columns = [
                    x.split("_")[-1] for x in columns_after if x not in columns_before
                ]
                transforms_dict[row["name"]] = {"dummy_columns": dummy_columns}
            # columns start and end indices
            data_types.at[index, "index_start"] = current_index
            current_index = current_index + row["num_classes"]
            data_types.at[index, "index_end"] = current_index

    print("Save...")
    df.to_pickle(hp.dir.data + "data_real_transformed.pkl")
    df.to_csv(hp.dir.data + "data_real_transformed.csv", index=False)
    save_obj(transforms_dict, hp.dir.data + "transforms_dict.pkl")
    data_types.to_csv(hp.dir.config + "data_types.csv", index=True, index_label="index")


if __name__ == "__main__":
    # Disable logging for this script
    sys.argv.append("hydra.run.dir=.")
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra/job_logging=disabled")
    sys.argv.append("hydra/hydra_logging=disabled")

    main()
