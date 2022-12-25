###===>>>++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Copyright (c) 2021. by Nicholas Kuo & Sebastiano Babieri, UNSW.                     +
# All rights reserved. This file is part of the Health Gym, and is released under the +
# "MIT Lisence Agreement". Please see the LICENSE file that should have been included +
# as part of this package.                                                            +
###===###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

###===>>>
# This is the 5th of all files for WGAN on Sepsis
import  numpy               as np

import  torch
import  torch.nn            as  nn
import  torch.nn.functional as  F

###===>>>
# (Part 1)      An "explicit" LSTM module
# This was needed because my older version of PyTorch
# cannot run RNNs through the autograd.grad function for
# calculating the gradient penalty
class MyLSTM(nn.Module):

    ###===###
    def __init__(self, ID, HD):
        super().__init__()

        #---
        self.ID = ID
        self.HD = HD

        #---
        self.i2h = nn.Linear(ID, HD * 4)
        self.h2h = nn.Linear(HD, HD * 4)

    ###===###
    def forward(self, x0):

        Q_k = torch.zeros(x0.shape[0], self.HD).cuda()
        S_k = torch.zeros(x0.shape[0], self.HD).cuda()

        Q_all = []

        for QStr in range(x0.shape[1]):

            X_k = x0[:, QStr, :]
            
            F_i, I_i, A_i, O_i = self.i2h(X_k).chunk(4, dim = 1)
            F_h, I_h, A_h, O_h = self.h2h(Q_k).chunk(4, dim = 1)

            F_k = torch.sigmoid(F_i + F_h)
            I_k = torch.sigmoid(I_i + I_h)
            A_k = torch.tanh(   A_i + A_h)
            O_k = torch.sigmoid(O_i + O_h)                

            S_k = F_k * S_k + I_k * A_k
            Q_k = O_k * torch.tanh(S_k)
            
            Q_all.append(Q_k.unsqueeze(1))

        Q_all = torch.cat(Q_all, dim = 1)

        return Q_all, (Q_k, S_k)

###===>>>
# (Part 2)      The generator network
class Generator(nn.Module):

    ###===###
    def __init__(self,
                 Hyper006_ID, Hyper007_HD,
                 data_types):
        super().__init__()

        #---
        ID, HD = Hyper006_ID, Hyper007_HD

        # The following is a pair of LSTM for bi-directions
        # _f for forward & _r for reverse
        self.rnn_f = MyLSTM(ID, HD)
        self.rnn_r = MyLSTM(ID, HD)

        # The LSTM output is then followed by 3 linear operators
        # with the leakyReLU activation
        self.linear1 = nn.Linear(2 * HD, HD)
        self.linear2 = nn.Linear(    HD, HD)
        self.linear3 = nn.Linear(HD, max(data_types["index_end"]))
        self.leakyReLU = nn.LeakyReLU(0.1)

        #---
        # The corresponding activation of each output dimension
        self.output_activations = []

        # The float values will be treated with sigmoid
        # (hence why we min-maxed the N2 and LN variables in file A001)
        max_real = max(data_types.loc[data_types["type"] == "real", "index_end"])
        self.output_activations.append(lambda x: torch.sigmoid(x[..., 0:max_real]))

        # Whereas the non-float values will be softmaxed across
        # their corresponding dimensions
        for index, row in data_types.iterrows():
            if row["type"] != "real":
                idxs = row["index_start"]
                idxe = row["index_end"]

                self.output_activations.append(
                    lambda x, idxs=idxs, idxe=idxe: torch.softmax(
                        x[..., idxs:idxe], dim=-1
                    )
                )

    ###===###
    def forward(self, x0):
        
        #---
        # this is for the forward LSTM
        x0_f = x0
        # and we feed the reversed signal in the backward LSTM
        x0_r = x0.flip(dims = [1])

        #---
        x1_f, _ = self.rnn_f(x0_f)
        x1_r, _ = self.rnn_r(x0_r)
        x1 = torch.cat((x1_f, x1_r), dim = 2)

        #---
        x2 = self.leakyReLU(self.linear1(x1))
        x3 = self.leakyReLU(self.linear2(x2))
        x4 = self.linear3(x3)

        #---
        x_list = [f(x4) for f in self.output_activations]
        out = torch.cat(x_list, dim=-1)

        return out

###===>>>
# (Part 3)      The discriminator network
class Discriminator(nn.Module):
    
    ###===###
    def __init__(self,
                 Hyper007_HD,
                 data_types):
        super().__init__()

        #---
        # the output dimension (OD) of the discriminator is simply 1
        HD, OD = Hyper007_HD, 1

        #---
        # this helps us to locate those float variables
        self.max_real = max(data_types.loc[data_types["type"] == "real", "index_end"])

        # and non-float variables will be treated with embedding
        self.embedding_layers = nn.ModuleList()

        # specifically soft-embeddings
        # thus making extrapolation in the gradient penalty possible
        self.soft_embedding = []

        # which simply refers to applying sets of (fuzzy) one-hot vectors to
        # the trainable embedding weights
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

        #---
        # first we use fully connected dense layers to mix the data
        self.linear1 = nn.Linear(sum(data_types["embedding_size"]), HD)
        self.linear2 = nn.Linear(HD, HD)

        # then we feed treated signal to a bi-direction LSTM
        self.rnn_f = MyLSTM(HD, HD)
        self.rnn_r = MyLSTM(HD, HD)

        # and rate the realness of the data
        self.linear3 = nn.Linear(2 * HD, OD)

        # with intermediate layers transformed by leakyReLU
        self.leakyReLU = nn.LeakyReLU(0.1)

    ###===###
    def forward(self, x0):
        
        #---
        # The non-float variables are embedded here 
        x_list = [x0[..., 0 : self.max_real]] + [
            f(x0, embedding_layer.weight)
            for f, embedding_layer in zip(self.soft_embedding, self.embedding_layers)
        ]
        x1 = torch.cat(x_list, dim=-1)

        #---
        x2 = self.leakyReLU(self.linear1(x1))
        x3 = self.leakyReLU(self.linear2(x2))

        x3_f = x3
        x3_r = x3.flip(dims = [1])

        _, (x4_f, _) = self.rnn_f(x3_f)
        _, (x4_r, _) = self.rnn_r(x3_r)
        
        x4 = torch.cat((x4_f, x4_r), dim = 1)
        
        out = self.linear3(x4)
        return out







