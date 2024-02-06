import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn

from th_operator import calc_grad

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
        p_drop     = 0.2,
        batch_norm = True
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
        self.batch_norm = batch_norm
        
        # Declare layers
        self.layer_list = []
        h_dim_prev = self.x_dim
        for h_dim in self.h_dim_list:
            # dense -> batchnorm -> actv -> dropout
            self.layer_list.append(nn.Linear(h_dim_prev,h_dim))
            if self.batch_norm: self.layer_list.append(nn.BatchNorm1d(num_features=h_dim))
            self.layer_list.append(self.actv)
            self.layer_list.append(nn.Dropout1d(p=self.p_drop))
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
            elif isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
                nn.init.zeros_(m.bias)
        
    def forward(self,x):
        """
            Forward propagate
        """
        intermediate_output_list = []
        for idx, layer in enumerate(self.net):
            x = layer(x)
            intermediate_output_list.append(x)
        # Final output
        final_output = x
        return final_output,intermediate_output_list
    

class GradientLayer(nn.Module):
    """
    Custom layer to compute derivatives for the steady Navier-Stokes equation using PyTorch.
    # model = Create_Model()
    # gradient_layer = GradientLayer(model)
    """
    def __init__(self, model):
        super(GradientLayer, self).__init__()
        self.model = model

    def forward(self, xyt):
        """
        Computing derivatives for the steady Navier-Stokes equation.
        Args:
            xy: input variable.
        Returns:
            psi: stream function.
            p_grads: pressure and its gradients.
            u_grads: u and its gradients.
            v_grads: v and its gradients.
        """
        x, y = xyt[..., 0], xyt[..., 1]
        x.requires_grad_(True)
        y.requires_grad_(True)
        # Combine x and y and predict u, v, p
        u_v_p, _ = self.model(th.stack([x, y], dim=-1))
        u, v, p = u_v_p[..., 0], u_v_p[..., 1], u_v_p[..., 2]
        
        # First derivatives
        u_x = calc_grad(u.sum(), x)
        u_y = calc_grad(u.sum(), y)
        v_x = calc_grad(v.sum(), x)
        v_y = calc_grad(v.sum(), y)
        p_x = calc_grad(p.sum(), x)
        p_y = calc_grad(p.sum(), y)
        # u_x = th.autograd.grad(u.sum(), x, create_graph=True)[0]
        # v_x = th.autograd.grad(v.sum(), x, create_graph=True)[0]
        # u_y = th.autograd.grad(u.sum(), y, create_graph=True)[0]
        # v_y = th.autograd.grad(v.sum(), y, create_graph=True)[0]
        # p_x = th.autograd.grad(p.sum(), x, create_graph=True)[0]
        # p_y = th.autograd.grad(p.sum(), y, create_graph=True)[0]

        # Second derivatives
        u_xx = calc_grad(u_x.sum(), x)
        u_yy = calc_grad(u_y.sum(), y)
        v_xx = calc_grad(v_x.sum(), x)
        v_yy = calc_grad(v_y.sum(), y)
        # u_xx = th.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        # u_yy = th.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        # v_xx = th.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        # v_yy = th.autograd.grad(v_y.sum(), y, create_graph=True)[0]

        p_grads = (p, p_x, p_y)
        u_grads = (u, u_x, u_y, u_xx, u_yy)
        v_grads = (v, v_x, v_y, v_xx, v_yy)

        return p_grads, u_grads, v_grads

class PINN(nn.Module):
    def __init__(self, network, rho=1.0, mu=0.01):
        super(PINN, self).__init__()
        self.network = network
        self.rho = rho
        self.mu = mu
        self.grads = GradientLayer(self.network)
    
    def forward(self, inputs):
        # Unpack inputs
        # Equation input: xy_eqn
        # Boundary Condition: xy_in, xy_out, xy_w1, xy_w2, xy_circle
        xy_eqn, xy_in, xy_out, xy_w1, xy_w2, xy_circle = inputs
        
        # compute gradients relative to equation
        p_grads, u_grads, v_grads = self.grads(xy_eqn)
        _, p_x, p_y = p_grads
        u, u_x, u_y, u_xx, u_yy = u_grads
        v, v_x, v_y, v_xx, v_yy = v_grads
        
        # compute equation loss
        u_eqn =  u*u_x + v*u_y + p_x/self.rho - self.mu*(u_xx + u_yy) / self.rho
        v_eqn =  u*v_x + v*v_y + p_y/self.rho - self.mu*(v_xx + v_yy) / self.rho
        uv_eqn = u_x + v_y
        
        u_eqn = u_eqn.unsqueeze(-1)
        v_eqn = v_eqn.unsqueeze(-1)
        uv_eqn = uv_eqn.unsqueeze(-1)
        uv_eqn = th.cat([u_eqn, v_eqn, uv_eqn], dim=1)
        
        # compute gradients relative to boundary condition
        p_r, u_grads_r, v_grads_r = self.grads(xy_out)
        uv_out = th.cat([p_r[0].unsqueeze(-1), p_r[0].unsqueeze(-1), p_r[0].unsqueeze(-1)], dim=1)

        p_l, u_grads_l, v_grads_l = self.grads(xy_w1)
        uv_w1 = th.cat([u_grads_l[0].unsqueeze(-1), v_grads_l[0].unsqueeze(-1), p_l[2].unsqueeze(-1)], dim=1)

        p_l, u_grads_l, v_grads_l = self.grads(xy_w2)
        uv_w2 = th.cat([u_grads_l[0].unsqueeze(-1), v_grads_l[0].unsqueeze(-1), p_l[2].unsqueeze(-1)], dim=1)

        p_l, u_grads_l, v_grads_l = self.grads(xy_circle)
        uv_circle = th.cat([u_grads_l[0].unsqueeze(-1), v_grads_l[0].unsqueeze(-1), u_grads_l[0].unsqueeze(-1)], dim=1)

        p_inn, u_inn, v_inn = self.grads(xy_in)
        uv_in = th.cat([u_inn[0].unsqueeze(-1), v_inn[0].unsqueeze(-1), u_inn[0].unsqueeze(-1)], dim=1)

        # build the PINN model for the steady Navier-Stokes equation
        return uv_eqn, uv_in, uv_out, uv_w1, uv_w2, uv_circle
    
    