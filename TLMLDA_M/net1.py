# -*- coding: utf-8 -*-
"""
Created on Fri May 14 20:07:59 2021

@author: ZHUANG
"""
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, in_dim=1582, hidden_size=500, out_dim=1582):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=1200),  #1200
            nn.BatchNorm1d(1200,affine=True,track_running_stats=True), 
            nn.ELU(),
            nn.Dropout(),

            nn.Linear(in_features=1200, out_features=900),    #900
            nn.BatchNorm1d(900,affine=True,track_running_stats=True),
            nn.ELU(),        
            nn.Dropout(),

            nn.Linear(in_features=900, out_features=hidden_size),   #hidden_size=500
            nn.BatchNorm1d(hidden_size,affine=True,track_running_stats=True),
            nn.ELU(),                                                                                  
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=900),
            nn.BatchNorm1d(900,affine=True,track_running_stats=True),            
            nn.Sigmoid(),
            nn.Dropout(),
                      
            nn.Linear(in_features=900, out_features=1200),
            nn.BatchNorm1d(1200,affine=True,track_running_stats=True),
            nn.Sigmoid(),
            nn.Dropout(),
                                           
            nn.Linear(in_features=1200, out_features=out_dim),
            nn.BatchNorm1d(out_dim,affine=True,track_running_stats=True),
            nn.Sigmoid(),          
        )
    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return encoder_out, decoder_out 

class my_net(nn.Module):    
    def __init__(self, num_classes,hidden_size):
        super(my_net, self).__init__()
        self.feature_layers = AutoEncoder()
        self.DNN = nn.Sequential(
            nn.Linear(hidden_size, 600),  #600
            nn.BatchNorm1d(600,affine=True,track_running_stats=True),
            nn.Sigmoid(),
            
            nn.Linear(600, 256),   #256      
            nn.Sigmoid(),              
            )  
        self.cls_fc = nn.Sequential(
            nn.Linear(256, num_classes), 
            )
    def forward(self, source, target):
        source_en_out, source_de_out = self.feature_layers(source)
        s_dnn_out = self.DNN(source_en_out)
        s_p = self.cls_fc(s_dnn_out)
        
        if self.training ==True:
            target_en_out, target_de_out = self.feature_layers(target)           
            t_dnn_out = self.DNN(target_en_out)
            t_p = self.cls_fc(t_dnn_out)
        else:
            target_en_out = 0
            target_de_out = 0
            t_dnn_out = 0
            t_p = 0

        return source_en_out, source_de_out, s_dnn_out, s_p, target_en_out, target_de_out, t_dnn_out, t_p




















