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
from scipy.spatial.distance import cdist


@hydra.main(config_path="config", config_name="config")
def main(hp: DictConfig):
    plt.rcParams.update({"font.size": 35})

    print("Load...")
    df_real = pd.read_pickle(hp.dir.data + "data_real_transformed.pkl")
    df_fake = pd.read_pickle(hp.dir.data + "data_fake_transformed.pkl")

    df_real = df_real.drop(columns=["icustay_id", "hour"]).astype(float)
    df_fake = df_fake.astype(float)

    df_real = df_real.sample(n=10000, random_state=123)

    ###############################
    #### Membership disclosure  ###
    ###############################

    print("Computing pairwise Euclidean distances...")
    pairwise_distances = cdist(df_real.values, df_fake.values, "euclidean")
    min_distances = pairwise_distances.min(axis=1)
    print("Overall minimum: {}".format(min_distances.min()))

    print("Plot distance distribution...")
    fig, ax = plt.subplots(figsize=(9, 10))
    plot = sns.histplot(
        x=min_distances, fill=True, stat="probability", alpha=0.3, linewidth=0, ax=ax
    )
    ax.yaxis.set_visible(False)
    ax.set(xlabel="Minimum Distances")

    fig.tight_layout()
    fig.savefig(hp.dir.data + "membership_disclosure.png")


if __name__ == "__main__":
    # Disable logging for this script
    sys.argv.append("hydra.run.dir=.")
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra/job_logging=disabled")
    sys.argv.append("hydra/hydra_logging=disabled")

    main()
