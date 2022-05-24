# _*_ coding:   utf-8 _*_
# @Time     :    10:29 AM
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from DenseFeatureExtractors import ASPP_module,Feature_extractors

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):
        super(GCNLayer, self).__init__()
        self.A = A
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        # 第一层GCN
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        nodes_count = self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        self.mask = torch.ceil(self.A * 0.00001)

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, model='normal'):

        # softmax归一化 
        H = self.BN(H)
        H_xx1 = self.GCN_liner_theta_1(H)
        e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        zero_vec = -9e15 * torch.ones_like(e)
        A = torch.where(self.mask > 0, e, zero_vec) + self.I
        if model != 'normal': A = torch.clamp(A, 0.1)
        A = F.softmax(A, dim=1)
        output = self.Activition(torch.mm(A, self.GCN_liner_out_1(H)))

        return output, A



def LDA_Process(self,curr_labels):
    '''
    :param curr_labels: height * width
    :return:
    '''
    curr_labels=np.reshape(curr_labels,[-1])
    idx=np.where(curr_labels!=0)[0]
    x=self.x_flatt[idx]
    y=curr_labels[idx]
    lda = LinearDiscriminantAnalysis()
    lda.fit(x,y-1)
    X_new = lda.transform(self.x_flatt)
    return np.reshape(X_new,[self.height, self.width,-1])

class ASPPGCN(nn.Module):
    def __init__(self, height, width, changel, class_count, Q, A,
                 model='normal'):
        super(ASPPGCN, self).__init__()

        self.class_count = class_count 

        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.model = model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # Normalize by column Q

        layers_count = 2
        # Spectra Transformation Sub-Network
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),
                                            nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        self.CNN_denoise_1 = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise_1.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128))
                self.CNN_denoise_1.add_module('CNN_denoise_Conv' + str(i),
                                            nn.Conv2d(128, 64, kernel_size=(1, 1)))
                self.CNN_denoise_1.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise_1.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(64), )
                self.CNN_denoise_1.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(64, 64, kernel_size=(1, 1)))
                self.CNN_denoise_1.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        self.CNN_denoise_2 = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise_2.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(64))
                self.CNN_denoise_2.add_module('CNN_denoise_Conv' + str(i),
                                              nn.Conv2d(64, 32, kernel_size=(1, 1)))
                self.CNN_denoise_2.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise_2.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(32), )
                self.CNN_denoise_2.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(32, 32, kernel_size=(1, 1)))
                self.CNN_denoise_2.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        self.CNN_denoise_3 = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise_3.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(32))
                self.CNN_denoise_3.add_module('CNN_denoise_Conv' + str(i),
                                              nn.Conv2d(32, 16, kernel_size=(1, 1)))
                self.CNN_denoise_3.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise_3.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(16), )
                self.CNN_denoise_3.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(16, 16, kernel_size=(1, 1)))
                self.CNN_denoise_3.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        self.CNN_denoise_4 = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise_4.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(16))
                self.CNN_denoise_4.add_module('CNN_denoise_Conv' + str(i),
                                              nn.Conv2d(16, 3, kernel_size=(1, 1)))
                self.CNN_denoise_4.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise_4.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(3), )
                self.CNN_denoise_4.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(3, 3, kernel_size=(1, 1)))
                self.CNN_denoise_4.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
        # Atrous Conv
        self.dense_features = Feature_extractors(3, 16)

        # Pixel-level Convolutional Sub-Network
        self.ASPP = ASPP_module(2048, 256, 16)
        self.conv1 = nn.Conv2d(1280, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(112, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       # nn.BatchNorm2d(64),
                                       # nn.ReLU(),
                                       nn.Conv2d(64, 64, kernel_size=1, stride=1))

        # Superpixel-level Graph Sub-Network
        self.GCN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 128, self.A))
            else:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 64, self.A))

        # Softmax layer
        self.Softmax_linear = nn.Sequential(nn.Linear(128, self.class_count))

    def forward(self, x):
        '''
        :param x: H*W*C
        :return: probability_map
        '''
        (h, w, c) = x.shape

        # 先去除噪声
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        noise = torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x = noise  # 直连

        clean_x_flatten = clean_x.reshape([h * w, -1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)


        noise1 = self.CNN_denoise_1(torch.unsqueeze(noise.permute([2, 0, 1]), 0))
        noise1 = self.CNN_denoise_2(noise1)
        noise1 = self.CNN_denoise_3(noise1)
        noise1 = self.CNN_denoise_4(noise1)
        noise1 = torch.squeeze(noise1, 0).permute([1, 2, 0])
        clean_x_1 = noise1
        hx = clean_x_1
        CNN_result,low_level_features = self.dense_features(torch.unsqueeze(hx.permute([2, 0, 1]), 0))  # spectral-spatial convolution
        CNN_result = self.ASPP(CNN_result)
        CNN_result = self.conv1(CNN_result)
        CNN_result = self.bn1(CNN_result)
        CNN_result = self.relu(CNN_result)
        CNN_result = F.interpolate(CNN_result, size=(int(math.ceil(x.size()[-2] / 4)),
                                   int(math.ceil(x.size()[-2] / 4))), mode='bilinear', align_corners=True)
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        CNN_result = torch.cat((CNN_result, low_level_features), dim=1)
        CNN_result = self.last_conv(CNN_result)
        CNN_result = F.interpolate(CNN_result, size=(int(x.size()[-2]),
                                int(x.size()[-2])), mode='bilinear', align_corners=True)

        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])

        # CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])


        H = superpixels_flatten
        if self.model == 'normal':
            for i in range(len(self.GCN_Branch)): H, _ = self.GCN_Branch[i](H)
        else:
            for i in range(len(self.GCN_Branch)): H, _ = self.GCN_Branch[i](H, model='smoothed')

        GCN_result = torch.matmul(self.Q, H)

        # fusion
        Y = torch.cat([GCN_result, CNN_result], dim=-1)
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)
        return Y
