# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Sebastiano Barbieri, UNSW.                                     +
#  All rights reserved. This file is part of the Health Gym, and is released under the   +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#  as part of this package.                                                              +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pickle
from pdb import set_trace as bp

import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.utils.data as utils
from lib.utils import save_obj
from lib.wgan_gradient_penalty import WGAN_GP, correlation
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config")
def main(hp: DictConfig):
    original_cwd = hydra.utils.get_original_cwd() + "/"

    print("Load...")
    df = pd.read_pickle(original_cwd + hp.dir.data + "data_real_transformed.pkl")
    df = df.drop(["icustay_id", "hour"], axis=1)

    print("Load data types...")
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
        original_cwd + hp.dir.config + "data_types.csv",
        usecols=dtype.keys(),
        dtype=dtype,
        index_col="index",
    )
    data_types = data_types[data_types["include"]]

    print("Create data loaders and tensors...")
    data = df.values
    data = data.reshape(
        (-1, hp.max_sequence_length, max(data_types["index_end"]))
    )  # (3910, 48, 54)
    data = torch.from_numpy(data).float()
    correlation_real = correlation(data)
    num_patients = data.shape[0]
    data = utils.TensorDataset(data, torch.full((num_patients, 1, 1), hp.max_sequence_length))
    trn_loader = utils.DataLoader(
        data, batch_size=hp.batch_size, shuffle=True, drop_last=True
    )

    print("Train model...")
    mlflow.set_tracking_uri(original_cwd + hp.dir.mlruns)
    wgan_gp = WGAN_GP(hp, data_types, correlation_real)

    #artifacts_dir = "../../../mlruns/0/c93934a39c414ab1abf056672917a7b4/artifacts/"
    #wgan_gp.load_model(
        #artifacts_dir + "models_generator/48/",
        #artifacts_dir + "models_discriminator/48/",
    #)    
    
    with mlflow.start_run():
        mlflow.log_params(hp)
        data_types.to_csv("data_types.csv", index=False)
        mlflow.log_artifact("data_types.csv")
        wgan_gp.train(trn_loader)


if __name__ == "__main__":
    main()
