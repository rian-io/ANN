''' Módulo que implementa um perceptron '''

from datetime import datetime
import random

class Hebb:
    ''' Classe implementação do perceptron '''

    def __init__(self, input_matrix, results_vector, learning_rate, n_entries):
        population = [x for x in range(1, 10)] * n_entries

        self.result = []
        self.seasons = 0
        self.learning_rate = learning_rate

        self.results_vector = results_vector
        self.input_matrix = [[-1] + line for line in input_matrix]

        self.weights = random.sample(population, n_entries + 1)
        self.initial_weights = self.weights[:]

    def training_hebb(self):
        ''' Treinamento do perceptron pela regra de Hebb '''
        while True:
            erro = False

            for i, sample in zip(range(0, len(self.results_vector)), self.input_matrix):
                output_sum = sum([weight * entrie for weight, entrie in zip(self.weights, sample)])

                output = self.signal_function(output_sum)

                if output != self.results_vector[i]:
                    partial = self.learning_rate * (self.results_vector[i] - output)
                    self.weights = [a + partial * b for a, b in zip(self.weights, sample)]
                    erro = True

            self.seasons += 1

            if not erro:
                break

    def classify(self, input_matrix):
        ''' Classifica uma amostra '''
        input_matrix = [[-1] + sample for sample in input_matrix]
        for sample in input_matrix:
            output_sum = sum([weight * entrie for weight, entrie in zip(self.weights, sample)])
            self.result.append(self.signal_function(output_sum))

    def signal_function(self, somatoria=0):
        ''' Implmentação da função sinal (degrau bipolar) '''
        return 1 if somatoria >= 0 else -1

    def save_results(self, nome_arquivo, contador=0):
        ''' Exibe os pesos sinátpticos e o número de épocas do perceptron '''
        data_file = open(file=nome_arquivo, mode='a', encoding='utf8')

        content = '''
Teste {0}

Pesos Iniciais: {1}
Pesos Finais: {2}
Épocas: {3}
Resultado da classificação: {4}
\n\n'''.format(contador, self.initial_weights, self.weights, self.seasons, self.result)

        data_file.write(content)
        data_file.close()
