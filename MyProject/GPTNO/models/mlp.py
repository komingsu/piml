import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn

class MultiLayerPerceptronClass(nn.Module):
    """_summary_
    MultiLayerPerceptron
    
    This class is a simple implementation of a multi-layer perceptron (MLP) using PyTorch
    """
    def __init__(
        self,
        name       = 'mlp',
        x_dim      = 784,
        h_dim_list = [256,256],
        y_dim      = 10,
        actv       = nn.ReLU(),
        p_drop     = 0.0,
        batch_norm = False,
        residual   = False,
        list_output= False
    ):
        """
            Initialize MLP
        """
        super(MultiLayerPerceptronClass,self).__init__()
        self.name       = name
        self.x_dim      = x_dim
        self.h_dim_list = h_dim_list
        self.y_dim      = y_dim
        self.actv       = actv
        self.p_drop     = p_drop
        self.resiudal   = residual
        self.batch_norm = batch_norm
        self.list_output= list_output
        
        # Declare layers
        self.layer_list = []
        h_dim_prev = self.x_dim
        for h_dim in self.h_dim_list:
            # dense -> batchnorm -> actv -> dropout
            self.layer_list.append(nn.Linear(h_dim_prev,h_dim))
            if self.batch_norm: self.layer_list.append(nn.BatchNorm1d(num_features=h_dim))
            self.layer_list.append(self.actv)
            if self.p_drop: self.layer_list.append(nn.Dropout1d(p=self.p_drop))
            h_dim_prev = h_dim
        self.layer_list.append(nn.Linear(h_dim_prev,self.y_dim))
        
        # Define net
        self.net = nn.Sequential()
        self.layer_names = []
        for l_idx,layer in enumerate(self.layer_list):
            layer_name = "%s_%02d"%(type(layer).__name__.lower(),l_idx)
            self.layer_names.append(layer_name)
            self.net.add_module(layer_name,layer)
        
        # Initialize parameters
        self.init_param(VERBOSE=False)
        
    def init_param(self,VERBOSE=False):
        """
            Initialize parameters
        """
        for m_idx,m in enumerate(self.modules()):
            if VERBOSE:
                print ("[%02d]"%(m_idx))
            if isinstance(m,nn.BatchNorm1d): # init BN
                nn.init.constant_(m.weight,1.0)
                nn.init.constant_(m.bias,0.0)
            elif isinstance(m, nn.Linear):  # Init dense
                if isinstance(self.actv, (nn.ReLU, nn.GELU, nn.LeakyReLU)):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif isinstance(self.actv, (nn.Sigmoid, nn.Tanh)):
                    nn.init.xavier_normal_(m.weight)
                elif isinstance(self.actv, nn.ELU):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu', a=self.actv.alpha)
                nn.init.zeros_(m.bias)
        
    def forward(self,x):
        """
            Forward propagate
        """
        intermediate_output_list = []
        for idx, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear) and self.residual and x.shape == layer(x).shape:
                x = x + layer(x)  # Apply residual connection if shapes match
            else:
                x = layer(x)
            intermediate_output_list.append(x)
        # Final output
        final_output = x
        if self.list_output:
            return final_output, intermediate_output_list
        else:
            return final_output
        