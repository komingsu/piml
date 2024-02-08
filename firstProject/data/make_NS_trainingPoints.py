import numpy as np
import scipy.io as sio

# Data file destination


# Task parameters
u0 = 0.1 # inlet flow velocity
rho = 5000 # density
mu = 0.5 # viscosity

# Samples
num_points_per_step = 10000  # number of spatial points per time step

# Data boundary
## Domain
x_ini, x_f, y_ini, y_f = -4, 16, -4, 4

## Time
T = 20  # total time in seconds
Delta_t = 0.1  # time step in seconds
num_time_steps = int(T / Delta_t)  # number of time steps
time_steps = np.linspace(0, T, num=num_time_steps)

## Circle
Cx, Cy = 0, 0
a, b = 1, 1

def generate_boundary_points(num_points, boundary_func, time_steps):
    xyt_points = []
    for t in time_steps:
        points = boundary_func(num_points)
        t_col = np.full((points.shape[0], 1), t)
        xyt = np.hstack((points, t_col))
        xyt_points.append(xyt)
    return np.vstack(xyt_points)

def circle_boundary(num_points):
    # Generate points for the circle boundary
    theta = np.random.uniform(0, 2*np.pi, num_points)
    x = Cx + a * np.cos(theta)
    y = Cy + b * np.sin(theta)
    return np.vstack((x, y)).T

def wall_boundary(num_points, x_range, y_value):
    # Generate points for wall boundaries (top or bottom)
    x = np.random.uniform(x_range[0], x_range[1], num_points)
    y = np.full(num_points, y_value)
    return np.vstack((x, y)).T

def inlet_outlet_boundary(num_points, y_range, x_value):
    # Generate points for inlet or outlet boundaries
    y = np.random.uniform(y_range[0], y_range[1], num_points)
    x = np.full(num_points, x_value)
    return np.vstack((x, y)).T

# Generating boundary points for each boundary condition and time step
xyt_circle = generate_boundary_points(num_points_per_step, circle_boundary, time_steps)
xyt_w1 = generate_boundary_points(num_points_per_step, lambda num_points: wall_boundary(num_points, (x_ini, x_f), y_ini), time_steps)
xyt_w2 = generate_boundary_points(num_points_per_step, lambda num_points: wall_boundary(num_points, (x_ini, x_f), y_f), time_steps)
xyt_in = generate_boundary_points(num_points_per_step, lambda num_points: inlet_outlet_boundary(num_points, (y_ini, y_f), x_ini), time_steps)
xyt_out = generate_boundary_points(num_points_per_step, lambda num_points: inlet_outlet_boundary(num_points, (y_ini, y_f), x_f), time_steps)

# Combine all training points
x_train = {
    "xyt_circle": xyt_circle,
    "xyt_w1": xyt_w1,
    "xyt_w2": xyt_w2,
    "xyt_in": xyt_in,
    "xyt_out": xyt_out,
    # Add equation collocation points if needed
}

