''' Módulo que implementa um perceptron '''

import random
import numpy as np

from datetime import datetime


class MLP:
    ''' Classe implementação da rede MLP '''
    def __init__(self, input_matrix, results_vector, layer_size, learning_rate, n_entries):
        population = [x * 0.1 for x in range(1, 10)] * 4

        self.layer_count = len(layer_size) - 1
        self.shape = layer_size

        self.error = []
        self.seasons = 0
        self.learning_rate = learning_rate

        self.results_vector = results_vector
        self.input_matrix = [[-1] + line for line in input_matrix]

        self.weights = []
        self._layer_input = []
        self._layer_output = []

        for (layer1, layer2) in zip(layer_size[:-1], layer_size[1:]):
            self.weights.append([random.sample(population, layer1 + 1)] * layer2)
            self._weight_delta = self.weights[:]

    def training(self, epsilon):
        ''' Treinamento de uma rede mlp '''
        while True:
            prev_error = self.calc_eqm(self.weights[-1][-1])

            for i, sample in zip(range(0, len(self.results_vector)), self.input_matrix):
                self.train_epoch(sample, i)

            self.seasons += 1

            cur_error = self.calc_eqm(self.weights[-1][-1])

            eqm = abs(cur_error - prev_error)

            if eqm <= epsilon:
                break

    def run(self, entry):
        '''Run the network based on the input data'''

        # Clear out the previous intermediate values lists
        self._layer_input = []
        self._layer_output = []

        layer_output = []

        # Run it
        for index in range(self.layer_count):
            # Determinate layer input
            if index == 0:
                layer_input = entry
            else:
                layer_input = self._layer_output[-1]

            layer_weights = self.weights[index][:]

            if index == self.layer_count - 1:
                for weights in layer_weights:
                    out = sum([weight * entry for weight, entry in zip(weights, layer_input)])
                    layer_output.append(self.sgm(out))
            else:
                pass

            self._layer_input.append(layer_input)
            self._layer_output.append(layer_output)

        return self._layer_output[-1]

    def train_epoch(self, entry, index_r):
        '''This method trains network for one epoch'''
        delta = 0

        # First run the network
        self.run(entry)

        # Calculate our deltas
        for index in reversed(range(self.layer_count)):
            layer_weights = self.weights[index][:]

            if index == self.layer_count - 1:
                # Compare to the target values
                output_delta = self.results_vector[index_r] - self._layer_output[index][-1]
                delta = output_delta * self.sgm(self._layer_input[index], True)

                error = self.calc_eqm(self.weights[-1][-1])
                self.error.append(error)
            else:
                # Compare to the following layer's delta                
                delta_pullback = self.weights[index - 1].T.dot(delta[-1])
                delta = delta_pullback[:-1, :] * self.sgm(self._layer_input[index], True)

            for weights in layer_weights:
                if index == 0:
                    partial = self.learning_rate + (self.learning_rate * delta)
                    weights = [a + partial * b for a, b in zip(weights, entry)]
                else:
                    partial = self.learning_rate - (self.learning_rate * delta)
                    weights = [a + partial * b for a, b in zip(weights, entry)]


    def sgm(self, value, derivative=False):
        ''' Calcula a função logistica '''
        if not derivative:
            if type(value) is int:
                return 1 / (1 + np.exp(-value))
            else:
                return 1 / (1 + np.exp(-value[-1]))
        else:
            out = self.sgm(value)
            return out * (1.0 - out)

    def calc_eqm(self, weights):
        ''' Calcula o erro médio '''
        error = 0
        len_input = len(self.input_matrix)
        for i in range(0, len(self.results_vector)):
            escalar = self.calc_scalar(weights, self.input_matrix[i])
            error += ((self.results_vector[i] - escalar) ** 2)
        return error / len_input

    def calc_scalar(self, weights, input_matrix):
        ''' Calcula o escalar entre os pesos e as entradas '''
        return sum([weight * entrie for weight, entrie in zip(weights, input_matrix)])
