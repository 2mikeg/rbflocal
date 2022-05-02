import numpy as np
from numpy.linalg import inv
import matplotlib.pylab as plt


class RBF:
    def __init__(self, space_vector, c: float, nodes: int):

        self.space_vector = space_vector
        self.c = c
        self.nodes = nodes

    def phi_matrix_f(nodes: int, space_vector: np.array, c: float) -> np.array:

        phi = np.zeros((nodes, nodes))

        for row in range(nodes):
            for col in range(nodes):
                r = np.linalg.norm(space_vector[row]-space_vector[col])
                phi[row, col] = np.sqrt(r**2 + c**2)

        return phi


    def fo_phi_f(space_vector: np.array, nodes: int, c: float) -> np.array:

        fo_phi = np.zeros((nodes, nodes))
        for row in range(nodes):
            for col in range(nodes):
                r = np.linalg.norm(space_vector[row] - space_vector[col])
                fo_phi[row, col] = ((r**2+c**2)**(-1/2))*(space_vector[row]-space_vector[col])

        return fo_phi


    def so_phi_f(space_vector: np.array, nodes: int, c: float) -> np.array:

        so_phi = np.zeros((nodes, nodes))
        for row in range(nodes):
            for col in range(nodes):
                r = np.linalg.norm(space_vector[row] - space_vector[col])
                so_phi[row, col] = c**2/((c**2+r**2)**(3/2))

        return so_phi

    def train(self, initial_velocity: float, n_steps: int, dt: float, _y: list, k: float):

                #uj0 = np.sin(np.pi * self.space_vector)
        uj0  = np.zeros(self.nodes)
        uj0[0] = _y[0]
        uj0[-1] = 0
        uj1 = np.zeros(self.nodes)
        uj_1 = np.zeros(self.nodes)
        h = np.zeros((self.nodes, n_steps+1))  # Matrix where the solution is stored after iteration

        # h[:, 0] = uj0


        uj_1 = uj0 - initial_velocity*dt

        for j in range(0, n_steps):
            if j == 0:
                _phi = self.phi_matrix_f(self.nodes, self.space_vector)
                alpha = np.linalg.solve(_phi, uj0)
                function_rbf = np.dot(_phi, alpha)

                _so_phi = self.so_phi_f(self.space_vector, self.nodes)
                so_rbf = np.dot(_so_phi, alpha)
                so_function_analytical = -np.sin(self.space_vector)

                #k_so_rbf = np.dot(k, so_rbf)
                k_so_rbf = k * so_rbf
                uj1 = k_so_rbf + uj0
                h[:, j+1] = uj1[:]
                uj_1 = uj0
                uj0 = uj1

            if j > 0:
                if j < len(_y):
                    uj0[0] = _y[j]
                else:
                    uj0[0] = 0

                uj0[-1] = 0
                _phi = self.phi_matrix_f(self.nodes, self.space_vector)
                alpha = np.linalg.solve(_phi, uj0)
                function_rbf = np.dot(_phi, alpha)

                _so_phi = self.so_phi_f(self.space_vector, self.nodes)
                so_rbf = np.dot(_so_phi, alpha)
                so_function_analytical = -np.sin(self.space_vector)

                #k_so_rbf = np.dot(k, so_rbf)
                k_so_rbf = k * so_rbf
                uj1 = k_so_rbf + (2*uj0)-uj_1
                h[:, j+1] = uj1[:]
                uj_1 = uj0
                uj0 = uj1

        return h