# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Sebastiano Barbieri, UNSW.                                     +
#  All rights reserved. This file is part of the Health Gym, and is released under the   +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#  as part of this package.                                                              +
#  Based in part on WGAN code from                                                       +
#  https://github.com/Zeleni9/                                                           +
#  https://github.com/EmilienDupont/                                                     +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
from pdb import set_trace as bp

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lib.utils import get_device
from torch import autograd
from tqdm import tqdm


def correlation(x, eps = 1e-8):
    """
    Compute correlation matrix of vectors in the last dimension of x
    """
    last_dim = x.shape[-1]
    x = x.reshape((-1, last_dim))
    x = x - x.mean(dim = 0, keepdim = True)
    x = x / torch.clamp(x.norm(dim = 0, keepdim = True), min = eps)
    correlation_matrix = x.transpose(0, 1) @ x
    return correlation_matrix


class Generator(nn.Module):
    def __init__(self, data_types):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, max(data_types["index_end"]))
        self.leakyReLU = nn.LeakyReLU(0.1)
        # output activation function
        self.output_activations = []
        max_real = max(data_types.loc[data_types["type"] == "real", "index_end"])
        self.output_activations.append(lambda x: torch.sigmoid(x[..., 0:max_real]))
        for index, row in data_types.iterrows():
            if row["type"] != "real":
                idxs = row["index_start"]
                idxe = row["index_end"]
                # need to get values: https://stackoverflow.com/questions/28014953/capturing-value-instead-of-reference-in-lambdas
                self.output_activations.append(
                    lambda x, idxs=idxs, idxe=idxe: torch.softmax(
                        x[..., idxs:idxe], dim=-1
                    )
                )

    def forward(self, x, seq_lengths, total_length):
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths=seq_lengths.squeeze(), batch_first=True, enforce_sorted=False
        )
        x, _ = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.leakyReLU(self.linear1(x))
        x = self.leakyReLU(self.linear2(x))
        x = self.linear3(x)
        x_list = [f(x) for f in self.output_activations]
        x = torch.cat(x_list, dim=-1)
        if min(seq_lengths) < total_length:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths=seq_lengths.squeeze(), batch_first=True, enforce_sorted=False
            )
            x, _ = nn.utils.rnn.pad_packed_sequence(
                x, batch_first=True, total_length=total_length
            )
        return x


class Discriminator(nn.Module):
    def __init__(self, data_types):
        super().__init__()
        self.max_real = max(data_types.loc[data_types["type"] == "real", "index_end"])
        # embedding layers
        self.embedding_layers = nn.ModuleList()
        self.soft_embedding = []
        for index, row in data_types.iterrows():
            if row["type"] != "real":
                self.embedding_layers.append(
                    nn.Embedding(row["num_classes"], row["embedding_size"])
                )
                idxs = row["index_start"]
                idxe = row["index_end"]
                self.soft_embedding.append(
                    lambda x, W, idxs=idxs, idxe=idxe: x[..., idxs:idxe] @ W
                )
        # others
        self.linear1 = nn.Linear(sum(data_types["embedding_size"]), 128)
        self.linear2 = nn.Linear(128, 128)
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )
        self.linear3 = nn.Linear(256, 1)
        self.leakyReLU = nn.LeakyReLU(0.1)

    def forward(self, x, seq_lengths):
        x_list = [x[..., 0 : self.max_real]] + [
            f(x, embedding_layer.weight)
            for f, embedding_layer in zip(self.soft_embedding, self.embedding_layers)
        ]
        x = torch.cat(x_list, dim=-1)
        x = self.leakyReLU(self.linear1(x))
        x = self.leakyReLU(self.linear2(x))
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths=seq_lengths.squeeze(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.rnn(x)
        hidden = torch.cat((hidden[0], hidden[1]), dim=-1)
        y = self.linear3(hidden)
        return y


class WGAN_GP:
    def __init__(self, hp, data_types, correlation_real = None):
        self.max_sequence_length = hp.max_sequence_length
        self.batch_size = hp.batch_size
        self.epochs = hp.epochs
        self.G_iter = hp.G_iter
        self.gp_weight = hp.gp_weight
        self.curriculum_lengths = hp.curriculum_lengths
        self.variable_length = hp.variable_length
        self.curriculum_length = self.max_sequence_length
        self.seq_lengths = None
        self.num_steps = 0

        self.device = get_device()
        self.G = Generator(data_types).to(self.device)
        self.D = Discriminator(data_types).to(self.device)
        self.D_opt = optim.Adam(self.D.parameters(), lr=1e-3, betas=(0.9, 0.99))
        self.G_opt = optim.Adam(self.G.parameters(), lr=1e-3, betas=(0.9, 0.99))

        # Correlation
        if correlation_real is not None:
            self.correlation_real = correlation_real.to(self.device)

        # Losses
        self.G_loss = None
        self.D_loss = None
        self.gradient_penalty = None
        self.gradient_norm = None
        self.correlation_loss = None
        
    def generate_data(self, num_samples, to_np=False):
        if self.seq_lengths is None:
            self.seq_lengths = torch.tensor(self.max_sequence_length).repeat(
                num_samples
            )
        z = torch.rand((num_samples, max(self.seq_lengths), 128)).to(self.device)
        data_fake = self.G(z, self.seq_lengths, self.curriculum_length)
        data_fake = data_fake.data.cpu().numpy() if to_np else data_fake
        return data_fake

    def _critic_train_iteration(self, data_real):
        """ """
        # Discriminator values for real and fake data
        data_real = data_real.to(self.device)
        data_fake = self.generate_data(self.batch_size)
        D_real = self.D(data_real, self.seq_lengths)
        D_fake = self.D(data_fake, self.seq_lengths)

        # Get gradient penalty
        with torch.backends.cudnn.flags(enabled=False):
            gradient_penalty = self._gradient_penalty(data_real, data_fake)

        # Calculate loss and optimize
        self.D_opt.zero_grad()
        D_loss = D_fake.mean() - D_real.mean() + gradient_penalty
        D_loss.backward()
        self.D_opt.step()

        # Record loss
        self.D_loss += D_loss.item()
        self.gradient_penalty += gradient_penalty.item()

    def _generator_train_iteration(self):
        """ """        
        # Discriminator value for fake data
        data_fake = self.generate_data(self.batch_size)
        D_fake = self.D(data_fake, self.seq_lengths)

        # Correlation loss
        correlation_loss = self._correlation_loss(data_fake)

        # Calculate loss and optimize
        self.G_opt.zero_grad()
        G_loss = -D_fake.mean() + correlation_loss
        G_loss.backward()
        self.G_opt.step()

        # Record loss
        self.G_loss += G_loss.item()
        self.correlation_loss += correlation_loss.item()

    def _correlation_loss(self, data_fake):
        # L1 distance between correlation matrices of real and fake data
        correlation_fake = correlation(data_fake)
        criterion = nn.L1Loss(reduction = "mean")
        return criterion(correlation_fake, self.correlation_real)

    def _gradient_penalty(self, data_real, data_fake):
        # Calculate interpolation
        alpha = torch.rand((self.batch_size, 1, 1)).to(self.device)
        alpha = alpha.expand_as(data_real)
        interpolated = alpha * data_real + (1 - alpha) * data_fake

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated, self.seq_lengths)

        # Calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(prob_interpolated).to(self.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Flatten to easily take norm per example in batch
        gradients = gradients.view(self.batch_size, -1)

        # Record gradient norm
        self.gradient_norm += gradients.norm(2, dim=1).mean().item()

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def train(self, trn_loader):
        for curriculum_step in range(len(self.curriculum_lengths)):
            self.curriculum_length = self.curriculum_lengths[curriculum_step]
            for epoch in tqdm(range(self.epochs)):
                # Zero Losses
                self.G_loss = 0
                self.D_loss = 0
                self.gradient_penalty = 0
                self.gradient_norm = 0
                self.correlation_loss = 0
                # Train
                for batch_idx, (data_real, batch_seq_lengths) in enumerate(trn_loader):
                    self.seq_lengths = torch.minimum(
                        batch_seq_lengths, torch.tensor(self.curriculum_length)
                    )
                    if self.variable_length:
                        self.seq_lengths = (
                            (self.seq_lengths * torch.rand(self.seq_lengths.shape))
                            .ceil()
                            .int()
                        )

                    # get random sequences of length seq_lengths
                    start_idx = (
                        (
                            (self.max_sequence_length - self.seq_lengths)
                            * torch.rand(self.seq_lengths.shape)
                        )
                        .floor()
                        .int()
                    )
                    idx = start_idx.repeat(
                        1, self.curriculum_length, data_real.shape[-1]
                    )
                    idx = idx + torch.arange(0, self.curriculum_length).unsqueeze(
                        0
                    ).unsqueeze(2)
                    idx = idx.clip(max=self.max_sequence_length - 1)
                    data_real = torch.gather(data_real, 1, idx)

                    self.num_steps += 1
                    self._critic_train_iteration(data_real)
                    if self.num_steps % self.G_iter == 0:
                        self._generator_train_iteration()
                # Log losses
                mlflow_step = curriculum_step * self.epochs + epoch
                if (mlflow_step % 10) == 0:
                    mlflow.log_metric("G", self.G_loss, step=mlflow_step)
                    mlflow.log_metric("D", self.D_loss, step=mlflow_step)
                    mlflow.log_metric(
                        "gradient_penalty", self.gradient_penalty, step=mlflow_step
                    )
                    mlflow.log_metric(
                        "gradient_norm", self.gradient_norm, step=mlflow_step
                    )
                    mlflow.log_metric("correlation_loss", self.correlation_loss, step=mlflow_step)

            # Save the trained parameters
            self.save_model()

    def save_model(self):
        uri_generator = "models_generator/" + str(self.curriculum_length)
        mlflow.pytorch.log_state_dict(self.G.state_dict(), uri_generator)
        uri_discriminator = "models_discriminator/" + str(self.curriculum_length)
        mlflow.pytorch.log_state_dict(self.D.state_dict(), uri_discriminator)
        print(
            "Models saved to "
            + mlflow.get_artifact_uri(uri_generator)
            + " and "
            + mlflow.get_artifact_uri(uri_discriminator)
        )

    def load_model(self, G_model_path, D_model_path):
        self.G.load_state_dict(mlflow.pytorch.load_state_dict(G_model_path))
        self.D.load_state_dict(mlflow.pytorch.load_state_dict(D_model_path))
        print("Generator model loaded from {}".format(G_model_path))
        print("Discriminator model loaded from {}".format(D_model_path))
