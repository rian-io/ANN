''' Módulo de acesso e conversão do arquivo com a matriz de dados '''

import os
from models.supervised.perceptron_hebb import Hebb
from models.supervised.perceptron_adaline import Adaline
from models.supervised.perceptron_mlp import MLP

def obter_dados(nome_arquivo, tipo_acesso=1):
    '''
    Acessa um arquivo de texto contendo os dados necessários
    para o treino e teste de uma rede neural
    '''
    # Abre o o arquivo em modo de leirura e armazena o conteúdo em uma string
    data_file = open(nome_arquivo, 'r')
    content = data_file.read()
    data_file.close()

    # Quebra o conteúdo por linha criando um vetor de strings
    split_content = content.split('\n')

    # Quebra cada string do vetor por espaços,
    # transformando cada linha em um vetor de letras e adiciona o vetor a uma matriz
    matriz = [line.split(' ') for line in split_content]

    # Transforma cada letra da matriz em um número do tipo float
    input_matrix = [[float(y) for y in x] for x in matriz]

    if tipo_acesso == 1:
        result_matrix = [line[-1] for line in input_matrix]
        input_matrix = [line[0:-1] for line in input_matrix]
        return input_matrix, result_matrix
    else:
        return input_matrix

def clean_results():
    '''
        path_0 = './perceptron_mlp_result.txt'
        if os.path.isfile(path_0):
            os.remove(path_0)
    '''
    path_1 = './perceptron_hebb_result.txt'
    if os.path.isfile(path_1):
        os.remove(path_1)

    path_2 = './perceptron_adaline_result.txt'
    if os.path.isfile(path_2):
        os.remove(path_2)


if __name__ == '__main__':
    clean_results()
    
    INPUT_MATRIX_TREINO, RESULT_MATRIZ = obter_dados('perceptron_hebb_treino.txt')
    INPUT_MATRIX_TESTE = obter_dados('perceptron_hebb_teste.txt', 2)

    for i in range(1, 6):
        PERCEPTRON = Hebb(INPUT_MATRIX_TREINO, RESULT_MATRIZ, 0.01, 3)
        PERCEPTRON.training_hebb()
        PERCEPTRON.classify(INPUT_MATRIX_TESTE)
        PERCEPTRON.save_results('perceptron_hebb_result.txt', i)

    INPUT_MATRIX_TREINO, RESULT_MATRIZ = obter_dados('perceptron_adaline_treino.txt')
    INPUT_MATRIX_TESTE = obter_dados('perceptron_adaline_teste.txt', 2)

    for i in range(1, 6):
        PERCEPTRON = Adaline(INPUT_MATRIX_TREINO, RESULT_MATRIZ, 0.0025, 4)
        if i < 3:
            PERCEPTRON.training_adaline(0.00001, True)
        else:
            PERCEPTRON.training_adaline(0.00001)
        PERCEPTRON.classify(INPUT_MATRIX_TESTE)
        PERCEPTRON.save_results('perceptron_adaline_result.txt', i)

    INPUT_MATRIX_TREINO, RESULT_MATRIZ = obter_dados('perceptron_mlp_treino.txt')
    INPUT_MATRIX_TESTE = obter_dados('perceptron_mlp_teste.txt', 2)
    for i in range(1, 6):
        PERCEPTRON = MLP(INPUT_MATRIX_TREINO, RESULT_MATRIZ, (4, i*5, 1), 0.0025, 4)
        PERCEPTRON.training(0.00001)
        
        #PERCEPTRON.classify(INPUT_MATRIX_TESTE)
        #PERCEPTRON.save_results('perceptron_hebb_result.txt', i)
