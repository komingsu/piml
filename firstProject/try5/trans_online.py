import numpy as np
import time
from pyDOE import lhs
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import scipy.io
import random

# Setup GPU for training (use tensorflow v1.9 for CuDNNLSTM)
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

class PINN_laminar_flow:
    # Initialize the class
    def __init__(self, Collo, INLET, OUTLET, WALL, uv_layers, lb, ub, ExistModel=0, uvDir=''):

        # Count for callback function
        self.count=0

        # Bounds
        self.lb = lb
        self.ub = ub

        # Mat. properties
        self.rho = 1.0
        self.mu = 0.02

        # Collocation point
        self.x_c = Collo[:, 0:1]
        self.y_c = Collo[:, 1:2]

        self.x_INLET = INLET[:, 0:1]
        self.y_INLET = INLET[:, 1:2]
        self.u_INLET = INLET[:, 2:3]
        self.v_INLET = INLET[:, 3:4]

        self.x_OUTLET = OUTLET[:, 0:1]
        self.y_OUTLET = OUTLET[:, 1:2]

        self.x_WALL = WALL[:, 0:1]
        self.y_WALL = WALL[:, 1:2]

        # Define layers
        self.uv_layers = uv_layers

        self.loss_rec = []

        # Initialize NNs
        if ExistModel== 0 :
            self.uv_weights, self.uv_biases = self.initialize_NN(self.uv_layers)
        else:
            print("Loading uv NN ...")
            self.uv_weights, self.uv_biases = self.load_NN(uvDir, self.uv_layers)

        # tf placeholders
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])

        self.x_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.x_WALL.shape[1]])
        self.y_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.y_WALL.shape[1]])

        self.x_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.x_OUTLET.shape[1]])
        self.y_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.y_OUTLET.shape[1]])

        self.x_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.x_INLET.shape[1]])
        self.y_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.y_INLET.shape[1]])
        self.u_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.u_INLET.shape[1]])
        self.v_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.v_INLET.shape[1]])

        self.x_c_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_c_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])

        # tf graphs
        self.u_pred, self.v_pred, self.p_pred, _, _, _ = self.net_uv(self.x_tf, self.y_tf)
        self.f_pred_u, self.f_pred_v, self.f_pred_s11, self.f_pred_s22, self.f_pred_s12, \
            self.f_pred_p = self.net_f(self.x_c_tf, self.y_c_tf)
        self.u_WALL_pred, self.v_WALL_pred, _, _, _, _ = self.net_uv(self.x_WALL_tf, self.y_WALL_tf)
        self.u_INLET_pred, self.v_INLET_pred, _, _, _, _ = self.net_uv(self.x_INLET_tf, self.y_INLET_tf)
        _, _, self.p_OUTLET_pred, _, _, _ = self.net_uv(self.x_OUTLET_tf, self.y_OUTLET_tf)

        self.loss_f = tf.reduce_mean(tf.square(self.f_pred_u)) \
                      + tf.reduce_mean(tf.square(self.f_pred_v))\
                      + tf.reduce_mean(tf.square(self.f_pred_s11))\
                      + tf.reduce_mean(tf.square(self.f_pred_s22))\
                      + tf.reduce_mean(tf.square(self.f_pred_s12))\
                      + tf.reduce_mean(tf.square(self.f_pred_p))
        self.loss_WALL = tf.reduce_mean(tf.square(self.u_WALL_pred)) \
                       + tf.reduce_mean(tf.square(self.v_WALL_pred))
        self.loss_INLET = tf.reduce_mean(tf.square(self.u_INLET_pred-self.u_INLET_tf)) \
                         + tf.reduce_mean(tf.square(self.v_INLET_pred-self.v_INLET_tf))
        self.loss_OUTLET = tf.reduce_mean(tf.square(self.p_OUTLET_pred))

        self.loss = self.loss_f + 2*(self.loss_WALL + self.loss_INLET + self.loss_OUTLET)

        # Optimizer for solution
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                var_list=self.uv_weights + self.uv_biases,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 100000,
                                                                         'maxfun': 100000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1*np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
                                                          var_list=self.uv_weights + self.uv_biases)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)



    def callback(self, loss):
        self.count = self.count+1
        self.loss_rec.append(loss)
        print('{} th iterations, Loss: {}'.format(self.count, loss))


    def train(self, iter, learning_rate):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   self.x_WALL_tf: self.x_WALL, self.y_WALL_tf: self.y_WALL,
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET,
                   self.learning_rate: learning_rate}

        loss_WALL = []
        loss_f = []
        loss_INLET = []
        loss_OUTLET = []

        for it in range(iter):                                                                                                                                                                                                                                                                                                                                                                                                             

            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e' %
                      (it, loss_value))

            loss_WALL.append(self.sess.run(self.loss_WALL, tf_dict))
            loss_f.append(self.sess.run(self.loss_f, tf_dict))
            self.loss_rec.append(self.sess.run(self.loss, tf_dict))
            loss_INLET.append(self.sess.run(self.loss_INLET, tf_dict))
            loss_OUTLET.append(self.sess.run(self.loss_OUTLET, tf_dict))

        return loss_WALL, loss_INLET, loss_OUTLET, loss_f, self.loss

    def train_bfgs(self):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   self.x_WALL_tf: self.x_WALL, self.y_WALL_tf: self.y_WALL,
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, x_star, y_star):
        u_star = self.sess.run(self.u_pred, {self.x_tf: x_star, self.y_tf: y_star})
        v_star = self.sess.run(self.v_pred, {self.x_tf: x_star, self.y_tf: y_star})
        p_star = self.sess.run(self.p_pred, {self.x_tf: x_star, self.y_tf: y_star})
        return u_star, v_star, p_star

    def getloss(self):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   self.x_WALL_tf: self.x_WALL, self.y_WALL_tf: self.y_WALL,
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET}

        loss_f = self.sess.run(self.loss_f, tf_dict)
        loss_WALL = self.sess.run(self.loss_WALL, tf_dict)
        loss_INLET = self.sess.run(self.loss_INLET, tf_dict)
        loss = self.sess.run(self.loss, tf_dict)
        loss_OUTLET = self.sess.run(self.loss_OUTLET, tf_dict)

        return loss_WALL, loss_INLET, loss_OUTLET, loss_f, loss


def postProcess(xmin, xmax, ymin, ymax, field_FLUENT, field_MIXED, s=2, alpha=0.5, marker='o'):

    [x_FLUENT, y_FLUENT, u_FLUENT, v_FLUENT, p_FLUENT] = field_FLUENT
    [x_MIXED, y_MIXED, u_MIXED, v_MIXED, p_MIXED] = field_MIXED

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(7, 4))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    # Plot MIXED result
    cf = ax[0, 0].scatter(x_MIXED, y_MIXED, c=u_MIXED, alpha=alpha-0.1, edgecolors='none', cmap='rainbow', marker=marker, s=int(s))
    ax[0, 0].axis('square')
    for key, spine in ax[0, 0].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 0].set_title(r'$u$ (m/s)')
    fig.colorbar(cf, ax=ax[0, 0], fraction=0.046, pad=0.04)

    cf = ax[1, 0].scatter(x_MIXED, y_MIXED, c=v_MIXED, alpha=alpha-0.1, edgecolors='none', cmap='rainbow', marker=marker, s=int(s))
    ax[1, 0].axis('square')
    for key, spine in ax[1, 0].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1, 0].set_title(r'$v$ (m/s)')
    fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)

    cf = ax[2, 0].scatter(x_MIXED, y_MIXED, c=p_MIXED, alpha=alpha, edgecolors='none', cmap='rainbow', marker=marker, s=int(s), vmin=-0.25, vmax=4.0)
    ax[2, 0].axis('square')
    for key, spine in ax[2, 0].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[2, 0].set_xticks([])
    ax[2, 0].set_yticks([])
    ax[2, 0].set_xlim([xmin, xmax])
    ax[2, 0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[2, 0].set_title('Pressure (Pa)')
    fig.colorbar(cf, ax=ax[2, 0], fraction=0.046, pad=0.04)

    # Plot FLUENT result
    cf = ax[0, 1].scatter(x_FLUENT, y_FLUENT, c=u_FLUENT, alpha=alpha, edgecolors='none', cmap='rainbow', marker=marker, s=s)
    ax[0, 1].axis('square')
    for key, spine in ax[0, 1].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 1].set_title(r'$u$ (m/s)')
    fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)

    cf = ax[1, 1].scatter(x_FLUENT, y_FLUENT, c=v_FLUENT, alpha=alpha, edgecolors='none', cmap='rainbow', marker=marker, s=s)
    ax[1, 1].axis('square')
    for key, spine in ax[1, 1].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1, 1].set_title(r'$v$ (m/s)')
    fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)

    cf = ax[2, 1].scatter(x_FLUENT, y_FLUENT, c=p_FLUENT, alpha=alpha, edgecolors='none', cmap='rainbow', marker=marker, s=s, vmin=-0.25, vmax=4.0)
    ax[2, 1].axis('square')
    for key, spine in ax[2, 1].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[2, 1].set_xticks([])
    ax[2, 1].set_yticks([])
    ax[2, 1].set_xlim([xmin, xmax])
    ax[2, 1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[2, 1].set_title('Pressure (Pa)')
    fig.colorbar(cf, ax=ax[2, 1], fraction=0.046, pad=0.04)

    plt.savefig('./uvp.png', dpi=300)
    plt.close('all')

def preprocess(dir='FenicsSol.mat'):
    '''
    Load reference solution from Fenics or Fluent
    '''
    data = scipy.io.loadmat(dir)

    X = data['x']
    Y = data['y']
    P = data['p']
    vx = data['vx']
    vy = data['vy']

    x_star = X.flatten()[:, None]
    y_star = Y.flatten()[:, None]
    p_star = P.flatten()[:, None]
    vx_star = vx.flatten()[:, None]
    vy_star = vy.flatten()[:, None]

    return x_star, y_star, vx_star, vy_star, p_star

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))


if __name__ == "__main__":

    # Visualize the collocation points
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.scatter(XY_c[:,0:1], XY_c[:,1:2], marker='o', alpha=0.1 ,color='blue')
    plt.scatter(WALL[:,0:1], WALL[:,1:2], marker='o', alpha=0.2 , color='green')
    plt.scatter(OUTLET[:, 0:1], OUTLET[:, 1:2], marker='o', alpha=0.2, color='orange')
    plt.scatter(INLET[:, 0:1], INLET[:, 1:2], marker='o', alpha=0.2, color='red')
    plt.show()

    with tf.device('/device:GPU:0'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        # Train from scratch
        # model = PINN_laminar_flow(XY_c, INLET, OUTLET, WALL, uv_layers, lb, ub)

        # Load trained neural network
        model = PINN_laminar_flow(XY_c, INLET, OUTLET, WALL, uv_layers, lb, ub, ExistModel = 1, uvDir = 'uvNN.pickle')

        start_time = time.time()
        loss_WALL, loss_INLET, loss_OUTLET, loss_f, loss = model.train(iter=10000, learning_rate=5e-4)
        model.train_bfgs()
        print("--- %s seconds ---" % (time.time() - start_time))

        # Save neural network
        model.save_NN('uvNN.pickle')

        # Save loss history
        with open('loss_history.pickle', 'wb') as f:
            pickle.dump(model.loss_rec, f)

        # Load fluent result
        [x_FLUENT, y_FLUENT, u_FLUENT, v_FLUENT, p_FLUENT] = preprocess(dir='../FluentReferenceMu002/FluentSol.mat')
        field_FLUENT = [x_FLUENT, y_FLUENT, u_FLUENT, v_FLUENT, p_FLUENT]

        # Get mixed-form PINN prediction
        x_PINN = np.linspace(0, 1.1, 251)
        y_PINN = np.linspace(0, 0.41, 101)
        x_PINN, y_PINN = np.meshgrid(x_PINN, y_PINN)
        x_PINN = x_PINN.flatten()[:, None]
        y_PINN = y_PINN.flatten()[:, None]
        dst = ((x_PINN-0.2)**2+(y_PINN-0.2)**2)**0.5
        x_PINN = x_PINN[dst >= 0.05]
        y_PINN = y_PINN[dst >= 0.05]
        x_PINN = x_PINN.flatten()[:, None]
        y_PINN = y_PINN.flatten()[:, None]
        u_PINN, v_PINN, p_PINN = model.predict(x_PINN, y_PINN)
        field_MIXED = [x_PINN, y_PINN, u_PINN, v_PINN, p_PINN]

        # Plot the comparison of u, v, p
        postProcess(xmin=0, xmax=1.1, ymin=0, ymax=0.41, field_FLUENT=field_FLUENT, field_MIXED=field_MIXED, s=3, alpha=0.5)
