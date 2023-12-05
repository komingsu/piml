import sys, os
import torch
import torch.nn as nn
import math
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LayerPerceptron(torch.nn.Module):
    def __init__(
        self,
        in_dim     = None,
        out_dim    = None,
        w_init     = False,
        b_init     = False,
        act        = nn.Tanh(),
    ):
        """
            Initialize LayerPerceptron
        """
        super(LayerPerceptron,self).__init__()
        self.dense      = nn.Linear(in_features=in_dim,
                                    out_features=out_dim,
                                    dtype=torch.float)

        self.activation = act
        # Initialize parameters
        self.init_param(w_init, b_init)

    def init_param(self, w_init, b_init):
        """
            Initialize parameters
        """
        if w_init:
            nn.init.constant_(self.dense.weight, w_init)
            if b_init:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.dense.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                nn.init.zeros_(self.dense.bias)
        else:
            nn.init.kaiming_normal_(self.dense.weight,a=math.sqrt(5))
            if b_init:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.dense.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                nn.init.zeros_(self.dense.bias)

    def forward(self,x):
        """
            Forward propagate
        """
        if self.activation is not None:
            out = self.activation(self.dense(x))
        else:
            out = self.dense(x)
        return out
    
class ResidualLayerPerceptron(torch.nn.Module):
    def __init__(
        self,
        in_dim     = None,
        out_dim    = None,
        w_init     = False,
        b_init     = False,
        act        = nn.Tanh(),
    ):
        """
            Initialize ResidualLayerPerceptron
        """
        super(ResidualLayerPerceptron,self).__init__()

        if in_dim != out_dim:
            raise ValueError("in_dim of ResBlock should be equal of out_dim, but got in_dim: {}, "
                             "out_dim: {}".format(in_dim, out_dim))

        self.dense      = nn.Linear(in_features=in_dim,
                                    out_features=out_dim,
                                    dtype=torch.float)

        self.activation = act
        # Initialize parameters
        self.init_param(w_init, b_init)

    def init_param(self, w_init, b_init):
        """
            Initialize parameters
        """
        if w_init:
            nn.init.constant_(self.dense.weight, w_init)
            if b_init:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.dense.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                nn.init.zeros_(self.dense.bias)
        else:
            nn.init.kaiming_normal_(self.dense.weight,a=math.sqrt(5))
            if b_init:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.dense.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                nn.init.zeros_(self.dense.bias)

    def forward(self,x):
        """
            Forward propagate
        """
        out = self.activation(self.dense(x)+x)
        return out

class MLPSequential(torch.nn.Module):
    def __init__(
        self,
        in_dim     = None,
        out_dim    = None,
        layers     = None,
        neurons    = None,
        residual   = False,
        w_init     = False,
        b_init     = False,
        act        = nn.Tanh(),
    ):
        """
            Initialize MLPSequential
        """
        super(MLPSequential,self).__init__()
        if layers < 3:
            raise ValueError("MLPSequential have at least 3 layers, but got layers: {}".format(layers))

        # Define net
        self.net = nn.Sequential()
        self.net.add_module("input", LayerPerceptron(in_dim=in_dim,
                                                           out_dim=neurons,
                                                           w_init=w_init,
                                                           b_init=b_init,
                                                           act=act,
                                                           ))
        for idx in range(layers - 2):
            if residual:
                self.net.add_module(f"res_{idx+1:02d}", ResidualLayerPerceptron(in_dim=neurons,
                                                                                     out_dim=neurons,
                                                                                     w_init=w_init,
                                                                                     b_init=b_init,
                                                                                     act=act,
                                                                                     ))
            else:
                self.net.add_module(f"mlp_{idx+1:02d}", LayerPerceptron(in_dim=neurons,
                                                                        out_dim=neurons,
                                                                        w_init=w_init,
                                                                        b_init=b_init,
                                                                        act=act,
                                                                        ))

        self.net.add_module("output", LayerPerceptron(in_dim=neurons,
                                                      out_dim=out_dim,
                                                      w_init=w_init,
                                                      b_init=b_init,
                                                      act=None,
                                                      ))       


    def forward(self,x):
        """
            Forward propagate
        """
        out = self.net(x)
        return out
    
class InputScaleNet(torch.nn.Module):
    def __init__(
        self,
        scales     = [],
        centers    = None,
    ):
        """
            Initialize InputScaleNet
        """
        super(InputScaleNet,self).__init__()
        self.scales     = torch.from_numpy(np.array(scales)).type(torch.float)
        if centers is None:
            self.centers = torch.zeros_like(self.scales, dtype=torch.float)
        else:
            self.centers = torch.from_numpy(np.array(centers)).type(torch.float)
            
        self.centers = self.centers.to(device)
        self.scales = self.scales.to(device)

    def forward(self,x):
        """
            Forward propagate
        """
        out = torch.mul(x - self.centers, self.scales)
        return out
    
class MultiScaleMLPSequential(torch.nn.Module):
    def __init__(
        self,
        in_dim     = None,
        out_dim    = None,
        layers     = None,
        neurons    = None,
        residual   = False, # use residual | [True, False]
        w_init     = False, # constant std | (ex 0.1, 0.01)
        b_init     = False, # use bias | [True, False]
        act        = nn.Tanh(),
        subnets    = None,  # subnet(multi scale) number
        amp        = 1.0,   # amplification factor of input
        base_scale = 2.0,   # base scale factor
        in_scale   = None,  # scale factor of input (ex [x,y,t])
        in_center  = None,  # Center position of coordinate translation (ex [x,y,t])
        vec_scen   = 4,     # number of latent vector scenarios
        vec_size   = 16,    # size of latent vector
    ):
        """
            Initialize MultiScaleMLPSequential
        """
        super(MultiScaleMLPSequential,self).__init__()
        self.subnets = subnets
        self.scale_coef = torch.Tensor([amp * (base_scale**i) for i in range(self.subnets)])
        self.num_scenarios = vec_scen
        self.latent_size = vec_size
        self.latent_vec = torch.from_numpy(np.random.randn(vec_scen, vec_size) / np.sqrt(vec_size)).type(torch.float)
        in_dim += self.latent_size

        # Define MultiScaleMLP
        self.msnet = nn.Sequential()
        for i in range(self.subnets):
            self.msnet.add_module(f"Scale_{i+1}_Net",MLPSequential(in_dim=in_dim,
                                                               out_dim=out_dim,
                                                               layers=layers,
                                                               neurons=neurons,
                                                               residual=residual,
                                                               w_init=w_init,
                                                               b_init=b_init,
                                                               act=act,
                                                               ))
        if in_scale:
            self.in_scale = InputScaleNet(in_scale, in_center)
        else:
            self.in_scale = torch.nn.Identity()
        
        self.latent_vec = self.latent_vec.to(device)

    def forward(self,x):
        """
            Forward propagate
        """
        x = self.in_scale(x)

        batch_size = x.shape[0]
        latent_vectors = self.latent_vec.view(self.num_scenarios, 1, self.latent_size).repeat(1,batch_size//self.num_scenarios,1).view(-1,self.latent_size)
        x = torch.concat([x, latent_vectors], axis=1)
        
        out = 0
        for i in range(self.subnets):
            x_s = x * self.scale_coef[i]
            out = out + self.msnet[i](x_s)
        return out
    
# def calc_latent_init(latent_size, latent_vector_ckpt, mode, num_scenarios):
#     if mode == "pretrain":
#         latent_init = np.random.randn(num_scenarios, latent_size) / np.sqrt(latent_size)
#     else:
#         latent_norm = np.mean(np.linalg.norm(latent_vector_ckpt, axis=1))
#         print("check mean latent vector norm: ", latent_norm)
#         latent_init = np.zeros((num_scenarios, latent_size))
#     latent_vector = torch.from_numpy(latent_init).type(torch.float)
#     return latent_vector

def calc_latent_init(latent_size, latent_vector_ckpt, mode, num_scenarios):
    latent_init = np.random.randn(num_scenarios, latent_size) / np.sqrt(latent_size)
    latent_vector = torch.from_numpy(latent_init).type(torch.float)
    return latent_vector


class LatentMLPSequential(torch.nn.Module):
    def __init__(
        self,
        in_dim     = None,
        out_dim    = None,
        layers     = None,
        neurons    = None,
        residual   = False, # use residual | [True, False]
        w_init     = False, # constant std | (ex 0.1, 0.01)
        b_init     = False, # use bias | [True, False]
        act        = nn.Tanh(),
        in_scale   = None,  # scale factor of input (ex [x,y,t])
        in_center  = None,  # Center position of coordinate translation (ex [x,y,t])
        vec_scen   = 4,     # number of latent vector scenarios
        vec_size   = 16,    # size of latent vector
    ):
        """
            Initialize MultiScaleMLPSequential
        """
        super(LatentMLPSequential,self).__init__()
        self.num_scenarios = vec_scen
        self.latent_size = vec_size
        self.latent_vec = torch.from_numpy(np.random.randn(vec_scen, vec_size) / np.sqrt(vec_size)).type(torch.float)
        self.latent_vec.requires_grad = True
        in_dim += self.latent_size

        # Define MultiScaleMLP
        self.lvnet = MLPSequential(in_dim=in_dim,
                                   out_dim=out_dim,
                                   layers=layers,
                                   neurons=neurons,
                                   residual=residual,
                                   w_init=w_init,
                                   b_init=b_init,
                                   act=act,
                                  )
        
        self.latent_vec = self.latent_vec.to(device)

    def forward(self,x):
        """
            Forward propagate
        """
        batch_size = x.shape[0]
        latent_vectors = self.latent_vec.view(self.num_scenarios, 1, self.latent_size).repeat(1,batch_size//self.num_scenarios,1).view(-1,self.latent_size)
        x = torch.concat([x, latent_vectors], axis=1)
        
        out = self.lvnet(x)
        return out