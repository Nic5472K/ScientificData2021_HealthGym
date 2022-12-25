import  numpy                   as      np
import  pandas                  as      pd
import  itertools
import  random
import  matplotlib.pyplot       as      plt
import  os

import  torch

###===>>>
def ExecuteB006(wgan_gp, Hyper002_Epochs):

    CurFolder = './Z002_Parameters/'+'Epoch_'+str(Hyper002_Epochs)+'/'
    if not os.path.exists(CurFolder):
        os.mkdir(CurFolder)

    D_params = wgan_gp.D.state_dict()
    G_params = wgan_gp.G.state_dict()

    torch.save(D_params,
               CurFolder + 'B002_D_StateDict_Epoch'+str(Hyper002_Epochs))
    torch.save(G_params,
               CurFolder + 'B002_G_StateDict_Epoch'+str(Hyper002_Epochs))
