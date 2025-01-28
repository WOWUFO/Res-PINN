import numpy as onp
import jax.numpy as np
import jax
from jax import random, grad, vmap, jit, jacfwd, jacrev
from jax.example_libraries import optimizers
from jax.experimental.ode import odeint
from jax.experimental.jet import jet
from jax.nn import relu
from jax import config
from jax import lax
from jax.flatten_util import ravel_pytree
import itertools
from functools import partial
from torch.utils import data
from tqdm import trange

import scipy.io
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


# Define MLP
def MLP(layers, L_x=1.0, L_y=1.0, M_t=1, M_x=1, M_y=1, activation=relu):
    # Define input encoding function
    def input_encoding(t, x, y):
        k_t = np.power(10.0, np.arange(0, M_t + 1))
        w_x = 2.0 * np.pi / L_x
        w_y = 2.0 * np.pi / L_y
        k_x = np.arange(1, M_x + 1)
        k_y = np.arange(1, M_y + 1)
        k_xx, k_yy = np.meshgrid(k_x, k_y)
        k_xx = k_xx.flatten()
        k_yy = k_yy.flatten()
        out = np.hstack([1, k_t * t,
                         np.cos(k_x * w_x * x), np.cos(k_y * w_y * y),
                         np.sin(k_x * w_x * x), np.sin(k_y * w_y * y),
                         np.cos(k_xx * w_x * x) * np.cos(k_yy * w_y * y),
                         np.cos(k_xx * w_x * x) * np.sin(k_yy * w_y * y),
                         np.sin(k_xx * w_x * x) * np.cos(k_yy * w_y * y),
                         np.sin(k_xx * w_x * x) * np.sin(k_yy * w_y * y)])
        return out

    # Define the neural net
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = 1.0 / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev * random.normal(k1, (d_in, d_out))
            b = np.zeros(d_out)
            return W, b

        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params

    def apply(params, inputs):
        t = inputs[0]
        x = inputs[1]
        y = inputs[2]
        H = input_encoding(t, x, y)
        for W, b in params[:-1]:
            outputs = np.dot(H, W) + b
            H = activation(outputs)
        W, b = params[-1]
        outputs = np.dot(H, W) + b
        return outputs

    return init, apply


def ResNet(layers, L_x=1.0, L_y=1.0, M_t=1, M_x=1, M_y=1, activation=relu):
    # Define input encoding function
    def input_encoding(t, x, y):
        k_t = np.power(10.0, np.arange(0, M_t + 1))
        w_x = 2.0 * np.pi / L_x
        w_y = 2.0 * np.pi / L_y
        k_x = np.arange(1, M_x + 1)
        k_y = np.arange(1, M_y + 1)
        k_xx, k_yy = np.meshgrid(k_x, k_y)
        k_xx = k_xx.flatten()
        k_yy = k_yy.flatten()
        out = np.hstack([1, k_t * t,
                         np.cos(k_x * w_x * x), np.cos(k_y * w_y * y),
                         np.sin(k_x * w_x * x), np.sin(k_y * w_y * y),
                         np.cos(k_xx * w_x * x) * np.cos(k_yy * w_y * y),
                         np.cos(k_xx * w_x * x) * np.sin(k_yy * w_y * y),
                         np.sin(k_xx * w_x * x) * np.cos(k_yy * w_y * y),
                         np.sin(k_xx * w_x * x) * np.sin(k_yy * w_y * y)])
        return out

    def init(rng_key):
        def init_layer(key, d_in, d_out):
            glorot_stddev = 1.0 / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev * random.normal(key, (d_in, d_out))
            b = np.zeros(d_out)
            return W, b

        keys = random.split(rng_key, len(layers))
        params = []
        for i in range(len(layers) - 1):
            key = keys[i]
            d_in = layers[i]
            d_out = layers[i + 1]
            W, b = init_layer(key, d_in, d_out)
            params.append((W, b))
        return params

    def apply(params, inputs):
        t = inputs[0]
        x = inputs[1]
        y = inputs[2]
        H = input_encoding(t, x, y)  # Shape: (d0,)

        W, b = params[0]
        H = activation(np.dot(H, W) + b)  # Shape: (d,)
        H_input = H

        for i in range(1, len(params) - 1):
            W, b = params[i]
            H = activation(np.dot(H, W) + b)  # Shape: (d,)
            if i % 2 == 1:
                H += H_input
                H_input = H
        W, b = params[-1]
        outputs = np.dot(H, W) + b
        return outputs

    return init, apply

class DataGenerator(data.Dataset):
    def __init__(self, t0, t1, n_t=10, n_x=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.t0 = t0
        self.t1 = t1 + 0.01 * t1
        self.n_t = n_t
        self.n_x = n_x
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        batch = self.__data_generation(subkey)
        return batch

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        subkeys = random.split(key, 2)
        t_r = random.uniform(subkeys[0], shape=(self.n_t,), minval=self.t0, maxval=self.t1).sort()
        x_r = random.uniform(subkeys[1], shape=(self.n_x, 2), minval=0.0, maxval=2.0 * np.pi)
        batch = (t_r, x_r)
        return batch

# Define the model
class PINN:
    def __init__(self, key, w_exact, layers, M_t, M_x, M_y, state0, t0, t1, n_t, x_star, y_star):

        self.w_exact = w_exact

        self.M_t = M_t
        self.M_x = M_x
        self.M_y = M_y

        # grid
        self.n_t = n_t
        self.n_x = n_x
        self.t0 = t0
        self.t1 = t1
        eps = 0.01 * t1
        self.t = np.linspace(self.t0, self.t1 + eps, n_t)
        self.x_star = x_star
        self.y_star = y_star

        # initial state
        self.state0 = state0

        self.M = np.triu(np.ones((n_t, n_t)), k=1).T

        self.init, self.apply = ResNet(layers, L_x=2 * np.pi, L_y=2 * np.pi, M_t=M_t, M_x=M_x, M_y=M_y,
                                                activation=np.tanh)
        params = self.init(rng_key=key)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3,
                                                                                                       decay_steps=10000,
                                                                                                       decay_rate=0.9))
        self.opt_state = self.opt_init(params)
        _, self.unravel = ravel_pytree(params)

        self.u0_pred_fn = vmap(vmap(self.u_net, (None, None, None, 0)), (None, None, 0, None))
        self.v0_pred_fn = vmap(vmap(self.v_net, (None, None, None, 0)), (None, None, 0, None))
        self.w0_pred_fn = vmap(vmap(self.vorticity_net, (None, None, None, 0)), (None, None, 0, None))
        self.u_pred_fn = vmap(vmap(vmap(self.u_net, (None, None, None, 0)), (None, None, 0, None)),
                              (None, 0, None, None))
        self.v_pred_fn = vmap(vmap(vmap(self.v_net, (None, None, None, 0)), (None, None, 0, None)),
                              (None, 0, None, None))
        self.w_pred_fn = vmap(vmap(vmap(self.vorticity_net, (None, None, None, 0)), (None, None, 0, None)),
                              (None, 0, None, None))
        self.r_pred_fn = vmap(vmap(self.residual_net, (None, None, 0, 0)), (None, 0, None, None))

        # Logger
        self.itercount = itertools.count()

        self.loss_log = []
        self.loss_ics_log = []
        self.loss_u0_log = []
        self.loss_v0_log = []
        self.loss_w0_log = []
        self.loss_bcs_log = []
        self.loss_res_w_log = []
        self.loss_res_c_log = []
        self.l2_error_log = []

    def neural_net(self, params, t, x, y):
        z = np.stack([t, x, y])
        outputs = self.apply(params, z)
        u = outputs[0]
        v = outputs[1]
        return u, v

    def u_net(self, params, t, x, y):
        u, _ = self.neural_net(params, t, x, y)
        return u

    def v_net(self, params, t, x, y):
        _, v = self.neural_net(params, t, x, y)
        return v

    def vorticity_net(self, params, t, x, y):
        u_y = grad(self.u_net, argnums=3)(params, t, x, y)
        v_x = grad(self.v_net, argnums=2)(params, t, x, y)
        w = v_x - u_y
        return w

    def residual_net(self, params, t, x, y):

        u, v = self.neural_net(params, t, x, y)

        u_x = grad(self.u_net, argnums=2)(params, t, x, y)
        v_y = grad(self.v_net, argnums=3)(params, t, x, y)

        w_t = grad(self.vorticity_net, argnums=1)(params, t, x, y)
        w_x = grad(self.vorticity_net, argnums=2)(params, t, x, y)
        w_y = grad(self.vorticity_net, argnums=3)(params, t, x, y)

        w_xx = grad(grad(self.vorticity_net, argnums=2), argnums=2)(params, t, x, y)
        w_yy = grad(grad(self.vorticity_net, argnums=3), argnums=3)(params, t, x, y)

        res_w = w_t + u * w_x + v * w_y - nu * (w_xx + w_yy)
        res_c = u_x + v_y

        return res_w, res_c

    @partial(jit, static_argnums=(0,))
    def residuals_and_weights(self, params, batch):
        t_r, x_r = batch
        loss_u0, loss_v0, loss_w0 = self.loss_ics(params)
        L_0 = 1e5 * (loss_u0 + loss_v0 + loss_w0)
        res_w_pred, res_c_pred = self.r_pred_fn(params, t_r, x_r[:, 0], x_r[:, 1])
        L_t = np.mean(res_w_pred ** 2 + 100 * res_c_pred ** 2, axis=1)
        return L_0, L_t

    @partial(jit, static_argnums=(0,))
    def loss_ics(self, params):
        # Compute forward pass
        u0_pred = self.u0_pred_fn(params, 0.0, self.x_star, self.y_star)
        v0_pred = self.v0_pred_fn(params, 0.0, self.x_star, self.y_star)
        w0_pred = self.w0_pred_fn(params, 0.0, self.x_star, self.y_star)
        # Compute loss
        loss_u0 = np.mean((u0_pred - self.state0[0, :, :]) ** 2)
        loss_v0 = np.mean((v0_pred - self.state0[1, :, :]) ** 2)
        loss_w0 = np.mean((w0_pred - self.state0[2, :, :]) ** 2)
        return loss_u0, loss_v0, loss_w0

    @partial(jit, static_argnums=(0,))
    def loss_res(self, params, batch):
        t_r, x_r = batch
        # Compute forward pass
        res_w_pred, res_c_pred = self.r_pred_fn(params, t_r, x_r[:, 0], x_r[:, 1])
        # Compute loss
        loss_res_w = np.mean(res_w_pred ** 2)
        loss_res_c = np.mean(res_c_pred ** 2)
        return loss_res_w, loss_res_c

    @partial(jit, static_argnums=(0,))
    def loss(self, params, batch):

        L_0, L_t= self.residuals_and_weights(params, batch)
        # Compute loss
        loss = np.mean(L_t + L_0)
        return loss

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params):
        w_pred = self.w_pred_fn(params, t_star[:num_step], x_star, y_star)
        l2_error = np.linalg.norm(w_pred - self.w_exact) / np.linalg.norm(self.w_exact)
        return l2_error

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, batch)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, dataset, nIter=10000):
        res_data = iter(dataset)
        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            batch = next(res_data)
            self.current_count = next(self.itercount)
            self.opt_state = self.step(self.current_count, self.opt_state, batch)

            if it % 1000 == 0:
                params = self.get_params(self.opt_state)

                l2_error_value = self.compute_l2_error(params)

                loss_value = self.loss(params, batch)

                loss_u0_value, loss_v0_value, loss_w0_value = self.loss_ics(params)
                loss_res_w_value, loss_res_c_value = self.loss_res(params, batch)

                self.l2_error_log.append(l2_error_value)
                self.loss_log.append(loss_value)
                self.loss_u0_log.append(loss_u0_value)
                self.loss_v0_log.append(loss_v0_value)
                self.loss_w0_log.append(loss_w0_value)
                self.loss_res_w_log.append(loss_res_w_value)
                self.loss_res_c_log.append(loss_res_c_value)

                pbar.set_postfix({'l2 error': l2_error_value,
                                  'Loss': loss_value,
                                  'loss_u0': loss_u0_value,
                                  'loss_v0': loss_v0_value,
                                  'loss_w0': loss_w0_value,
                                  'loss_res_w': loss_res_w_value,
                                  'loss_res_c': loss_res_c_value,
                                 })


data = np.load('NS.npy', allow_pickle=True).item()
# Test data
sol = data['sol']

t_star = data['t']
x_star = data['x']
y_star = data['y']
nu = data['viscosity']

# downsampling
sol = sol
x_star = x_star
y_star = y_star

# Create PINNs model
key = random.PRNGKey(1234)

u0 = data['u0']
v0 = data['v0']
w0 = data['w0']
state0 = np.stack([u0, v0, w0])
M_t = 2
M_x = 5
M_y = 5
d0 = 2 * M_x + 2 * M_y + 4 * M_x * M_y + M_t + 2
layers = [d0, 128, 128, 128, 128, 2]
# layers = [d0, 128, 128, 128, 128, 128, 128, 128, 128, 2]

num_step = 10
t0 = 0.0
t1 = t_star[num_step]
n_t = 32

# Create data set
n_x = 256
dataset = DataGenerator(t0, t1, n_t, n_x)

N = 20
w_pred_list = []
params_list = []
losses_list = []

for k in range(N):
    # Initialize model
    print('Final Time: {}'.format((k + 1) * t1))
    w_exact = sol[num_step * k: num_step * (k + 1), :, :]
    model = PINN(key, w_exact, layers, M_t, M_x, M_y, state0, t0, t1, n_t, x_star, y_star)
    model.train(dataset, nIter=100000)

    # Store
    params = model.get_params(model.opt_state)
    w_pred = model.w_pred_fn(params, t_star[:num_step], x_star, y_star)
    w_pred_list.append(w_pred)
    flat_params, _ = ravel_pytree(params)
    params_list.append(flat_params)
    losses_list.append([model.l2_error_log,
                        model.loss_log,
                        model.loss_u0_log,
                        model.loss_v0_log,
                        model.loss_w0_log,
                        model.loss_res_w_log,
                        model.loss_res_c_log, ])

    np.save('Res_w_pred_list.npy', w_pred_list)
    np.save('Res_params_list.npy', params_list)
    np.save('Res_losses_list.npy', losses_list)

    # error
    w_preds = np.vstack(w_pred_list)
    error = np.linalg.norm(w_preds - sol[:num_step * (k + 1), :, :]) / np.linalg.norm(sol[:num_step * (k + 1), :, :])
    print('Relative l2 error: {:.3e}'.format(error))

    params = model.get_params(model.opt_state)
    u0_pred = model.u0_pred_fn(params, t1, x_star, y_star)
    v0_pred = model.v0_pred_fn(params, t1, x_star, y_star)
    w0_pred = model.w0_pred_fn(params, t1, x_star, y_star)
    state0 = np.stack([u0_pred, v0_pred, w0_pred])