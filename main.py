import numpy as np
import matplotlib.pyplot as plt

from src.RBF import phi_matrix_f, fo_phi_f, so_phi_f

nodes = 100
distance = 1
dx = distance/(nodes-1)
_x = np.linspace(0, distance, nodes)
#f = np.sin(np.pi * _x)
c = 0.5
n_steps = 200
dt = 1e-7
time = n_steps * dt
elastic_modulus = 1
density = 2
initial_velocity = 1
k = ((elastic_modulus*dt**2)/density)
_y = [ 0, 0.47675, -0.62681, 0.66253, -0.55683, 0.08213, -0.052714, 0.020251, -0.0069143, 0.0016148]
