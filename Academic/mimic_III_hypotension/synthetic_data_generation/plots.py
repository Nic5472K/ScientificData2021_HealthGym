# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Sebastiano Barbieri, UNSW.                                     +
#  All rights reserved. This file is part of the Health Gym, and is released under the   +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#  as part of this package.                                                              +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import sys
from pdb import set_trace as bp

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig


def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)


@hydra.main(config_path="config", config_name="config")
def main(hp: DictConfig):
    plt.rcParams.update({"font.size": 35})

    print("Load...")
    df_real = pd.read_pickle(hp.dir.data + "data_real.pkl")
    df_fake = pd.read_pickle(hp.dir.data + "data_fake.pkl")
    df_real = df_real[df_fake.columns]

    print("Rename columns...")
    mapper = {
        "MAP": "MAP [mmHg]",
        "diastolic_bp": "Diastolic BP [mmHg]",
        "systolic_bp": "Systolic BP [mmHg]",
        "urine": "Urine [mL]",
        "ALT": "ALT [IU/L]",
        "AST": "AST [IU/L]",
        "PO2": "PO2 [mmHg]",
        "lactic_acid": "Lactic Acid [mmol/L]",
        "serum_creatinine": "Serum Creatinine [mg/dL]",
        "fluid_boluses": "Fluid Boluses [mL]",
        "vasopressors": "Vasopressors [mcg/kg/min]",
        "FiO2": "FiO2",
        "GCS_total": "GCS (Total)",
        "urine_m": "Urine (M)",
        "ALT_AST_m": "ALT/AST (M)",
        "FiO2_m": "FiO2 (M)",
        "GCS_total_m": "GCS (M)",
        "PO2_m": "PO2 (M)",
        "lactic_acid_m": "Lactic Acid (M)",
        "serum_creatinine_m": "Serum Creatinine (M)",
    }
    df_real.rename(columns=mapper, inplace=True)
    df_fake.rename(columns=mapper, inplace=True)
    df_real["Type"] = "Real"
    df_fake["Type"] = "Synthetic"
    df_all = pd.concat([df_fake, df_real], ignore_index=True)
    all_cols = mapper.values()

    mapper_fluid_boluses = {"0": "<250", "250": "<500", "500": "<1000", "1000": "≥1000"}
    df_all["Fluid Boluses [mL]"] = (
        df_all["Fluid Boluses [mL]"]
        .astype(str)
        .astype("category")
        .map(mapper_fluid_boluses)
    )
    df_all["Fluid Boluses [mL]"].cat.reorder_categories(
        mapper_fluid_boluses.values(), inplace=True
    )

    mapper_vasopressors = {
        "0.0": "0",
        "1e-06": "<8.4",
        "8.4": "<20.28",
        "20.28": "≥20.28",
    }
    df_all["Vasopressors [mcg/kg/min]"] = (
        df_all["Vasopressors [mcg/kg/min]"]
        .astype(str)
        .astype("category")
        .map(mapper_vasopressors)
    )
    df_all["Vasopressors [mcg/kg/min]"].cat.reorder_categories(
        mapper_vasopressors.values(), inplace=True
    )

    mapper_FiO2 = {
        "0.0": "<.2",
        "0.2": ".2",
        "0.3": ".3",
        "0.4": ".4",
        "0.5": ".5",
        "0.6": ".6",
        "0.7": ".7",
        "0.8": ".8",
        "0.9": ".9",
        "1.0": "1.0",
    }
    df_all["FiO2"] = df_all["FiO2"].astype(str).astype("category").map(mapper_FiO2)
    df_all["FiO2"].cat.reorder_categories(mapper_FiO2.values(), inplace=True)

    mapper_GCS = {
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
        "8": "8",
        "9": "9",
        "10": "10",
        "11": "11",
        "12": "12",
        "13": "13",
        "14": "14",
        "15": "15",
    }
    df_all["GCS (Total)"] = (
        df_all["GCS (Total)"].astype(str).astype("category").map(mapper_GCS)
    )
    df_all["GCS (Total)"].cat.reorder_categories(mapper_GCS.values(), inplace=True)

    mapper_bin = {"0": "False", "1": "True"}
    for col in [
        "Urine (M)",
        "ALT/AST (M)",
        "FiO2 (M)",
        "GCS (M)",
        "PO2 (M)",
        "Lactic Acid (M)",
        "Serum Creatinine (M)",
    ]:
        df_all[col] = df_all[col].astype(str).astype("category").map(mapper_bin)
        df_all[col].cat.reorder_categories(mapper_bin.values(), inplace=True)

    ###############################
    #### ### Distributions  ### ###
    ###############################

    print("Plot distributions...")
    ncols = 4
    fig, ax = plt.subplots(ncols=ncols, nrows=5, figsize=(45, 40))
    for ix, col in enumerate(all_cols):
        i, j = ix // ncols, ix - (ix // ncols) * ncols
        if ix < 9:
            print(col)
            plot = sns.kdeplot(
                x=df_all[col].astype(float).values,
                hue=df_all["Type"],
                fill=True,
                ax=ax[i, j],
            )
            if col == "MAP [mmHg]":
                ax[i, j].set(xlim=[10, 140])
            elif col == "Diastolic BP [mmHg]":
                ax[i, j].set(xlim=[10, 120])
            elif col == "Systolic BP [mmHg]":
                ax[i, j].set(xlim=[30, 200])
            elif col == "Urine [mL]":
                ax[i, j].set(xlim=[0, 600])
            elif col == "ALT [IU/L]":
                ax[i, j].set(xlim=[0, 300])
            elif col == "AST [IU/L]":
                ax[i, j].set(xlim=[0, 300])
            elif col == "PO2 [mmHg]":
                ax[i, j].set(xlim=[30, 200])
            elif col == "Lactic Acid [mmol/L]":
                ax[i, j].set(xlim=[0, 5])
            elif col == "Serum Creatinine [mg/dL]":
                ax[i, j].set(xlim=[0, 8])
        else:
            print(col)
            plot = sns.histplot(
                x=df_all[col],
                hue=df_all["Type"],
                fill=True,
                multiple="dodge",
                stat="probability",
                shrink=0.8,
                alpha=0.3,
                linewidth=0,
                ax=ax[i, j],
            )
            if col == "Urine (M)":
                move_legend(ax[i, j], "lower right")

        plot.legend_.set_title(None)
        ax[i, j].yaxis.set_visible(False)
        ax[i, j].set(xlabel=col)

    fig.tight_layout()
    fig.savefig(hp.dir.data + "distributions_comparison.png")

    ###############################
    #### ### Boxplots       ### ###
    ###############################

    print("Plot boxplots...")

    df_all_id = df_all.copy()
    num_people = df_all_id.shape[0] / 48
    df_all_id["patient_id"] = np.repeat(np.arange(num_people), 48)
    # difference with next row
    df_all_id["MAP Difference [mmHg]"] = df_all_id.groupby("patient_id")[
        "MAP [mmHg]"
    ].diff(periods=-1)
    df_all_id["MAP Difference [mmHg]"] = -df_all_id["MAP Difference [mmHg]"]

    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(25, 40))
    sns.boxplot(
        x="Fluid Boluses [mL]", y="MAP [mmHg]", hue="Type", data=df_all_id, ax=ax[0]
    )
    sns.boxplot(
        x="Vasopressors [mcg/kg/min]",
        y="MAP [mmHg]",
        hue="Type",
        data=df_all_id,
        ax=ax[1],
    )
    fig.tight_layout()
    fig.savefig(hp.dir.data + "map_comparison.png")

    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(25, 40))
    sns.boxplot(
        x="Fluid Boluses [mL]",
        y="MAP Difference [mmHg]",
        hue="Type",
        data=df_all_id,
        ax=ax[0],
    )
    sns.boxplot(
        x="Vasopressors [mcg/kg/min]",
        y="MAP Difference [mmHg]",
        hue="Type",
        data=df_all_id,
        ax=ax[1],
    )
    fig.tight_layout()
    fig.savefig(hp.dir.data + "map_diff_comparison.png")

    ###############################
    #### ### Correlations   ### ###
    ###############################

    print("Plot correlations...")

    df_all_codes = df_all.copy()
    for col in [
        "Fluid Boluses [mL]",
        "Vasopressors [mcg/kg/min]",
        "FiO2",
        "GCS (Total)",
        "Urine (M)",
        "ALT/AST (M)",
        "FiO2 (M)",
        "GCS (M)",
        "PO2 (M)",
        "Lactic Acid (M)",
        "Serum Creatinine (M)",
    ]:
        df_all_codes[col] = df_all_codes[col].cat.codes

    mapper = {
        "MAP [mmHg]": "MAP",
        "Diastolic BP [mmHg]": "Diastolic BP",
        "Systolic BP [mmHg]": "Systolic BP",
        "Urine [mL]": "Urine",
        "ALT [IU/L]": "ALT",
        "AST [IU/L]": "AST",
        "PO2 [mmHg]": "PO2",
        "Lactic Acid [mmol/L]": "Lactic Acid",
        "Serum Creatinine [mg/dL]": "Serum Creatinine",
        "Fluid Boluses [mL]": "Fluid Boluses",
        "Vasopressors [mcg/kg/min]": "Vasopressors",
    }
    df_all_codes.rename(columns=mapper, inplace=True)

    real_matrix = (
        df_all_codes[df_all_codes["Type"] == "Real"]
        .drop(columns=["Type"])
        .astype(float)
        .corr()
    )
    fake_matrix = (
        df_all_codes[df_all_codes["Type"] == "Synthetic"]
        .drop(columns=["Type"])
        .astype(float)
        .corr()
    )
    mask = np.triu(np.ones_like(real_matrix, dtype=bool))

    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(25, 40))
    with sns.axes_style("white"):
        for i, (matrix, data_type) in enumerate(
            zip([fake_matrix, real_matrix], ["Synthetic", "Real"])
        ):
            sns.heatmap(
                matrix,
                cmap="coolwarm",
                mask=mask,
                vmin="-1",
                vmax="1",
                linewidths=0.5,
                square=True,
                ax=ax[i],
            ).set_title("Correlation Matrix: " + data_type + " Data")

    fig.tight_layout(pad=3.0)
    fig.savefig(hp.dir.data + "correlation_comparison.png")


if __name__ == "__main__":
    # Disable logging for this script
    sys.argv.append("hydra.run.dir=.")
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra/job_logging=disabled")
    sys.argv.append("hydra/hydra_logging=disabled")

    main()
