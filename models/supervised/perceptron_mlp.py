import random
import numpy as np


class MLP:
    """
    Classe que implementa uma rede neural do tipo Perceptron Multicamadas (MLP).
    """

    def __init__(self, input_matrix, results_vector, layer_size, learning_rate, n_entries=None):
        """
        Inicializa a MLP com pesos aleatórios e estrutura da rede.
        :param input_matrix: dados de entrada (lista de listas)
        :param results_vector: saídas esperadas
        :param layer_size: lista com a quantidade de neurônios em cada camada
        :param learning_rate: taxa de aprendizado
        :param n_entries: parâmetro não utilizado, mas mantido por compatibilidade
        """
        self.layer_count = len(layer_size) - 1
        self.shape = layer_size

        self.error = []
        self.seasons = 0
        self.learning_rate = learning_rate

        self.results_vector = results_vector
        self.input_matrix = [[-1] + line for line in input_matrix]  # Adiciona bias -1

        self.weights = []
        self._layer_input = []
        self._layer_output = []

        # Inicialização dos pesos com valores aleatórios pequenos
        for i in range(self.layer_count):
            in_size = layer_size[i] + 1  # +1 para o bias
            out_size = layer_size[i + 1]
            layer_weights = [[random.uniform(-0.5, 0.5) for _ in range(in_size)] for _ in range(out_size)]
            self.weights.append(layer_weights)

        # Inicializa o delta dos pesos com zeros
        self._weight_delta = [np.zeros((len(layer), len(layer[0]))) for layer in self.weights]

    def sigmoid(self, x):
        """
        Função de ativação sigmoid.
        """
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, y):
        """
        Derivada da função sigmoid.
        """
        return y * (1.0 - y)

    def training(self, epsilon=0.01, max_epochs=10000):
        """
        Treina a rede até que o erro seja menor que epsilon ou o número de épocas seja atingido.
        """
        for epoch in range(max_epochs):
            total_error = 0
            for i, input_vector in enumerate(self.input_matrix):
                # Forward pass
                outputs = [np.array(input_vector)]
                for layer_index, layer_weights in enumerate(self.weights):
                    input_with_bias = np.append(-1, outputs[-1]) if layer_index > 0 else outputs[-1]
                    input_array = np.array(layer_weights) @ input_with_bias
                    activated_output = self.sigmoid(input_array)
                    outputs.append(activated_output)

                # Backward pass (retropropagação)
                expected = np.array(self.results_vector[i])
                deltas = [expected - outputs[-1]]
                total_error += np.sum(deltas[0] ** 2) / 2

                # Calcula deltas para as camadas ocultas
                for j in reversed(range(1, len(self.weights))):
                    error = np.dot(np.array(self.weights[j]).T, deltas[0])
                    derivative = self.sigmoid_derivative(outputs[j])
                    deltas.insert(0, error[:-1] * derivative)  # Ignora bias

                # Atualiza os pesos
                for j in range(len(self.weights)):
                    input_layer = np.append(-1, outputs[j])  # Inclui o bias
                    for k in range(len(self.weights[j])):
                        self.weights[j][k] = (
                            np.array(self.weights[j][k])
                            + self.learning_rate * deltas[j][k] * input_layer
                        ).tolist()

            self.error.append(total_error)
            self.seasons += 1

            if total_error < epsilon:
                break
