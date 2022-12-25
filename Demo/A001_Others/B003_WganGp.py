import  numpy                   as      np
import  pandas                  as      pd
import  itertools
import  random
import  yaml

import  torch
import  torch.nn            as      nn
import  torch.nn.functional as      F
import  torch.optim         as      optim
from    torch               import  autograd

#---
from    A001_Others.B003zC001_Models    import *

###===>>>
def correlation(x, eps = 1e-8):
    last_dim = x.shape[-1]
    x = x.reshape((-1, last_dim))
    x = x - x.mean(dim = 0, keepdim = True)
    x = x / torch.clamp(x.norm(dim = 0, keepdim = True), min = eps)
    correlation_matrix = x.transpose(0, 1) @ x

    ###===>>>
    return correlation_matrix

###===>>>
def LoadPreTrain(content = [False, 'G_SD', 'D_SD', 0]):

    Continue = content[0]
    Load_From = './Z002_Parameters/Epoch_' + str(content[3]) + '/'
    G_SD = Load_From + content[1]
    D_SD = Load_From + content[2]

    if Continue:
        G_SD = torch.load(G_SD)
        D_SD = torch.load(D_SD)

        ###===>>>
        return G_SD, D_SD

    else:
        ###===>>>
        return 0, 0

#<<<===>>>
class ExecuteB003:

    ###===>>>
    def __init__(self,
                 All_Trainable_Data,
                 Hyper001_BatchSize, Hyper002_Epochs,
                 Hyper003_G_iter, Hyper004_GP_Lambda, Hyper005_C_Lambda,
                 Hyper006_ID, Hyper007_HD,
                 Hyper008_LR, Hyper009_Betas,
                 data_types,
                 continue_info = [False, 'G_SD', 'D_SD', 0]
                 ):
        super().__init__()
        
        #---
        self.batch_size = Hyper001_BatchSize
        self.epochs     = Hyper002_Epochs
        self.G_iter     = Hyper003_G_iter
        self.gp_weight  = Hyper004_GP_Lambda
        self.c_weight   = Hyper005_C_Lambda
        self.ID         = Hyper006_ID
        self.HD         = Hyper007_HD
        self.lr         = Hyper008_LR
        self.betas      = Hyper009_Betas

        #---
        self.correlation_real = correlation(All_Trainable_Data)

        #---
        if torch.cuda.is_available():
            self.CUDA = True
        else:
            self.CUDA = False

        #---
        self.G = Generator(
                    Hyper006_ID, Hyper007_HD,
                    data_types)
        self.D = Discriminator(
                    Hyper007_HD,
                    data_types)

        if self.CUDA:
            self.G = self.G.cuda()
            self.D = self.D.cuda()

            self.correlation_real = self.correlation_real.cuda()

        #---
        G_SD, D_SD = LoadPreTrain(continue_info)

        if G_SD != 0:
            self.G.load_state_dict(G_SD)
            self.D.load_state_dict(D_SD)

            self.PreviousEpoch = continue_info[3]

        else:
            self.PreviousEpoch = 0

        #---
        self.D_opt = optim.Adam(self.D.parameters(), lr = self.lr, betas = self.betas)
        self.G_opt = optim.Adam(self.G.parameters(), lr = self.lr, betas = self.betas)

    ###===>>>
    def generate_data(self, seq_len, num_samples = None):

        if num_samples is None:
            num_samples = self.batch_size
        
        #---
        z = torch.rand((num_samples, seq_len, self.ID)).cuda()
        data_fake = self.G(z)

        ###===>>>
        return data_fake

    ###===>>>
    def _critic_train_iteration(self, data_real):
        #---
        data_real = data_real
        data_fake = self.generate_data(data_real.shape[1], data_real.shape[0])
        D_real = self.D(data_real)
        D_fake = self.D(data_fake)

        #---
        with torch.backends.cudnn.flags(enabled=False):
            gradient_penalty = self._gradient_penalty(data_real, data_fake)

        #---
        self.D_opt.zero_grad()
        D_loss = D_fake.mean() - D_real.mean() + gradient_penalty
        D_loss.backward()
        self.D_opt.step()

        ###===>>>
        return D_loss.item(), gradient_penalty.item()

    ###===>>>
    def _generator_train_iteration(self, seq_len):
        #---
        data_fake = self.generate_data(seq_len)
        D_fake = self.D(data_fake)

        correlation_loss = self._correlation_loss(data_fake)

        #---
        self.G_opt.zero_grad()
        G_loss = -D_fake.mean() + self.c_weight * correlation_loss
        G_loss.backward()
        self.G_opt.step()

        ###===>>>
        return G_loss.item()

    ###===>>>
    def _correlation_loss(self, data_fake):
        correlation_fake = correlation(data_fake)
        criterion = nn.L1Loss(reduction = "mean")

        ###===>>>
        return criterion(correlation_fake, self.correlation_real)

    ###===>>>
    def _gradient_penalty(self, data_real, data_fake):

        #---
        alpha = torch.rand((self.batch_size, 1, 1)).cuda()
        alpha = alpha.expand_as(data_real)
        interpolated = alpha * data_real + (1 - alpha) * data_fake

        #---
        prob_interpolated = self.D(interpolated)

        #---
        gradients = autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(prob_interpolated).cuda(),
            create_graph=True,
            retain_graph=True,
        )[0]

        #---
        gradients = gradients.view(self.batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        ###===>>>
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    ###===>>>
    def train(self, All_loader):

        #---
        All_Length = list(All_loader.keys())

        All_Length.sort()

        #---
        for epoch in range(self.epochs - self.PreviousEpoch):

            for Ltr in range(len(All_Length)):

                Cur_Len = All_Length[Ltr]

                Cur_loader = All_loader[Cur_Len]

                print("###===>>>")

                #---
                for batch_idx, (data_real, _) in enumerate(Cur_loader):

                    data_real = data_real.cuda()

                    for itr in range(self.G_iter):
                        D_Loss, GP = self._critic_train_iteration(data_real)

                    G_Loss = self._generator_train_iteration(seq_len = Cur_Len)

                #---
                print("###===###")
                print("Epoch: \t{}".format(self.PreviousEpoch + epoch + 1))
                print("Loader Len: \t{}".format(Cur_Len))
                print("---" * 3)
                print("D_Loss: \t{}".format(D_Loss))
                print("GP: \t\t{}".format(    GP))
                print("G_Loss: \t{}".format(G_Loss))
                print("")









                    
