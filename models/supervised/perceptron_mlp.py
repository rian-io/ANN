"""
mlp_ocr_refactored.py

Refactored Multilayer Perceptron (MLP) implementation from scratch using NumPy.
- Code identifiers and functions are in English.
- Comments and explanations are in Portuguese (as requested).
- The network is constructed from a list describing the number of neurons in each layer.
- Hidden activation can be chosen among 'relu', 'leaky_relu', 'elu', 'sigmoid'.
- Output activation uses softmax (suitable for multi-class classification + cross-entropy).
- Includes training (forward/backward separated), prediction and evaluation metrics.
"""

import numpy as np
from typing import List, Tuple, Callable, Dict, Union, Optional
from numpy import floating


# -----------------------------
# Activation functions & utils
# -----------------------------

def relu(Z: np.ndarray) -> np.ndarray:
    """ReLU activation.
    Retorna max(0, Z).
    """
    return np.maximum(0, Z)


def relu_derivative(Z: np.ndarray) -> np.ndarray:
    """Derivada da ReLU em relação a Z.
    Retorna 1 para Z>0, 0 caso contrário.
    """
    return (Z > 0).astype(float)


def leaky_relu(Z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU: pequena inclinação para Z<0"""
    return np.where(Z > 0, Z, alpha * Z)


def leaky_relu_derivative(Z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Derivada da Leaky ReLU"""
    d = np.ones_like(Z)
    d[Z < 0] = alpha
    d[Z == 0] = 0.5 * (1 + alpha)  # valor arbitrário para Z==0 (pouco relevante)
    return d


def elu(Z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Exponential Linear Unit (ELU)"""
    return np.where(Z > 0, Z, alpha * (np.exp(Z) - 1.0))


def elu_derivative(Z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Derivada do ELU"""
    return np.where(Z > 0, 1.0, alpha * np.exp(Z))


def sigmoid(Z: np.ndarray) -> np.ndarray:
    """Sigmoid activation"""
    return 1.0 / (1.0 + np.exp(-Z))


def sigmoid_derivative(Z: np.ndarray) -> np.ndarray:
    """Derivada da sigmoid: sigma * (1 - sigma). Recebe Z (logits) e calcula via sigma(Z)."""
    s = sigmoid(Z)
    return s * (1 - s)


def softmax(Z: np.ndarray) -> np.ndarray:
    """Softmax estável numericamente.
    Espera Z com shape (n_classes, n_examples) e retorna mesma shape com probabilidades.
    """
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z_shift)
    return expZ / np.sum(expZ, axis=0, keepdims=True)


# Map activation names to functions and derivatives
_ACTIVATIONS: Dict[str, Tuple[Callable, Callable]] = {
    'relu': (relu, relu_derivative),
    'leaky_relu': (leaky_relu, leaky_relu_derivative),
    'elu': (elu, elu_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative)
}


# -----------------------------
# Metrics (evaluation)
# -----------------------------

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Acurácia simples: proporção de rótulos corretos.
    y_true e y_pred são vetores de shape (n_examples,)
    """
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Matriz de confusão (n_classes x n_classes)"""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, Union[np.ndarray, float, floating]]:
    """Calcula precision, recall e f1 por classe e micro/macro averages."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    fp = np.sum(cm, axis=0).astype(float) - tp
    fn = np.sum(cm, axis=1).astype(float) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)

    # macro averages
    macro_p = np.mean(precision)
    macro_r = np.mean(recall)
    macro_f1 = np.mean(f1)

    # micro averages
    total_tp = np.sum(tp)
    total_fp = np.sum(fp)
    total_fn = np.sum(fn)
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) != 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) != 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) != 0 else 0.0

    return {
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1,
        'micro_precision': micro_p,
        'micro_recall': micro_r,
        'micro_f1': micro_f1
    }


# -----------------------------
# MLP class
# -----------------------------

class MLP:
    """
    Multilayer Perceptron implemented with NumPy.
    Comentários em Português conforme solicitado.
    """

    def __init__(self, architecture: List[int], learning_rate: float = 0.01, hidden_activation: str = 'relu', seed: Optional[int] = None):
        """
        Constructor.
        - architecture: vector com número de neurônios por camada (entrada ... saída).
          Exemplo: [784, 128, 10] -> input=784, hidden=128, output=10.
        - learning_rate: taxa de aprendizado (padrão 0.01).
        - hidden_activation: 'relu' | 'leaky_relu' | 'elu' | 'sigmoid'.
        - seed: opcional, para reprodutibilidade.
        """
        if seed is not None:
            np.random.seed(seed)

        # Validations
        if len(architecture) < 2:
            raise ValueError("Architecture must contain at least input and output sizes.")

        if hidden_activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{hidden_activation}'. Choose from {_ACTIVATIONS.keys()}")

        # Arquitetura da rede (usar numpy arrays internamente posteriormente)
        self.architecture = list(architecture)
        self.learning_rate = learning_rate
        self.hidden_activation_name = hidden_activation
        self.hidden_activation, self.hidden_activation_derivative = _ACTIVATIONS[hidden_activation]

        # Inicializa pesos e biases (W shapes: (n_next, n_current), b shapes: (n_next,1))
        self.weights = []
        self.biases = []
        for i in range(len(self.architecture) - 1):
            in_size = self.architecture[i]
            out_size = self.architecture[i + 1]
            # He / Xavier initialization could be used; aqui usamos He-like init
            W = np.random.randn(out_size, in_size) * np.sqrt(2. / max(1, in_size))
            b = np.zeros((out_size, 1))
            self.weights.append(W)
            self.biases.append(b)

    # -----------------------------
    # Architecture modification helpers
    # -----------------------------

    def set_input_size(self, n_input: int):
        """Define o número de neurônios na camada de entrada e reinit a rede."""
        # Atualiza arquitetura e reinicializa parâmetros
        self.architecture[0] = int(n_input)
        self._reinitialize_parameters()

    def set_hidden_layers(self, hidden_layers: List[int]):
        """Recebe vetor com tamanhos das camadas intermediárias e reinicializa rede."""
        self.architecture = [self.architecture[0]] + [int(x) for x in hidden_layers] + [self.architecture[-1]]
        self._reinitialize_parameters()

    def set_output_size(self, n_output: int):
        """Define o número de neurônios na camada final e reinicializa rede."""
        self.architecture[-1] = int(n_output)
        self._reinitialize_parameters()

    def _reinitialize_parameters(self):
        """Reinicializa pesos e biases com base na arquitetura atual."""
        self.weights = []
        self.biases = []
        for i in range(len(self.architecture) - 1):
            in_size = self.architecture[i]
            out_size = self.architecture[i + 1]
            W = np.random.randn(out_size, in_size) * np.sqrt(2. / max(1, in_size))
            b = np.zeros((out_size, 1))
            self.weights.append(W)
            self.biases.append(b)

    # -----------------------------
    # Forward pass
    # -----------------------------

    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation (vetorizado).
        - X: shape (n_features, n_examples)
        Retorna:
        - Zs: lista de Z (pre-activations) por camada (não inclui entrada)
        - As: lista de A (ativations) por camada incluindo entrada (As[0] = X)
        """
        As = [X]
        Zs = []

        A = X
        for i in range(len(self.weights)):
            W = self.weights[i]
            b = self.biases[i]
            Z = W @ A + b  # shape: (n_next, n_examples)
            Zs.append(Z)

            # Para camadas ocultas: aplicar a ativação escolhida
            if i < len(self.weights) - 1:
                if self.hidden_activation_name == 'leaky_relu':
                    A = leaky_relu(Z)
                elif self.hidden_activation_name == 'elu':
                    A = elu(Z)
                elif self.hidden_activation_name == 'sigmoid':
                    A = sigmoid(Z)
                else:
                    A = relu(Z)
            else:
                # última camada: softmax
                A = softmax(Z)
            As.append(A)

        return Zs, As

    # -----------------------------
    # Backward pass (vectorized)
    # -----------------------------

    def backward(self, Zs: List[np.ndarray], As: List[np.ndarray], Y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backpropagation vetorizado para atualização de parâmetros.
        - Zs: lista de pre-activations (n_layers-1)
        - As: lista de activations incluindo entrada (len = n_layers)
        - Y: labels one-hot shape (n_output, n_examples)
        Retorna grad_W, grad_b (listas com mesmos shapes de weights/biases)
        """
        m = Y.shape[1]
        L = len(self.weights)

        grad_W = [np.zeros_like(W) for W in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        # dZ for the last layer (softmax + cross-entropy simplification)
        A_last = As[-1]  # (n_output, m)
        dZ = A_last - Y  # (n_output, m)
        grad_W[L - 1] = (1.0 / m) * (dZ @ As[L - 1].T)
        grad_b[L - 1] = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)

        # propagate backwards
        dA_prev = dZ
        for l in range(L - 2, -1, -1):
            Z = Zs[l]
            A_prev = As[l]
            W_next = self.weights[l + 1]

            # compute dA for layer l+1 backpropagated to layer l
            dA = W_next.T @ dA_prev

            # compute dZ depending on activation
            if self.hidden_activation_name == 'leaky_relu':
                dZ = dA * leaky_relu_derivative(Z)
            elif self.hidden_activation_name == 'elu':
                dZ = dA * elu_derivative(Z)
            elif self.hidden_activation_name == 'sigmoid':
                dZ = dA * sigmoid_derivative(Z)
            else:
                dZ = dA * relu_derivative(Z)

            grad_W[l] = (1.0 / m) * (dZ @ A_prev.T)
            grad_b[l] = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)

            dA_prev = dZ

        return grad_W, grad_b

    # -----------------------------
    # Parameter update
    # -----------------------------

    def _update_parameters(self, grad_W: List[np.ndarray], grad_b: List[np.ndarray]):
        """Atualiza pesos e bias com gradientes calculados."""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_W[i]
            self.biases[i] -= self.learning_rate * grad_b[i]

    # -----------------------------
    # Training loop
    # -----------------------------

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 1000, batch_size: Optional[int] = None, verbose: bool = True):
        """
        Treina a rede.
        - X: input shape (n_features, n_examples)
        - Y: one-hot labels shape (n_output, n_examples)
        - epochs: número de épocas
        - batch_size: None -> full-batch, ou int para mini-batch
        """
        n_examples = X.shape[1]
        if batch_size is None:
            batch_size = n_examples  # full-batch by default
        
        assert isinstance(batch_size, int), "batch_size must be an integer"

        for epoch in range(1, epochs + 1):
            permutation = np.random.permutation(n_examples)
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]

            epoch_loss = 0.0

            for start in range(0, n_examples, batch_size):
                end = min(start + batch_size, n_examples)
                X_batch = X_shuffled[:, start:end]
                Y_batch = Y_shuffled[:, start:end]

                Zs, As = self.forward(X_batch)
                grad_W, grad_b = self.backward(Zs, As, Y_batch)
                self._update_parameters(grad_W, grad_b)

                # compute batch loss (cross-entropy)
                probs = As[-1]
                batch_loss = -np.sum(Y_batch * np.log(probs + 1e-9)) / X_batch.shape[1]
                epoch_loss += batch_loss * (X_batch.shape[1] / n_examples)

            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.6f}")

    # -----------------------------
    # Prediction & evaluation
    # -----------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades (softmax) para cada amostra.
        X shape: (n_features, n_examples)
        """
        _, As = self.forward(X)
        return As[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna rótulos preditos (inteiros) shape (n_examples,)"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=0)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, object]:
        """
        Avalia a rede e devolve métricas (accuracy, confusion matrix, precision/recall/f1)
        - y_true: vetor de inteiros shape (n_examples,)
        """
        y_pred = self.predict(X)
        acc = accuracy(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, n_classes=self.architecture[-1])
        prf = precision_recall_f1(y_true, y_pred, n_classes=self.architecture[-1])
        return {
            'accuracy': acc,
            'confusion_matrix': cm,
            'prf': prf
        }

    # -----------------------------
    # Utility: save / load parameters (simple)
    # -----------------------------

    def save_parameters(self, filepath: str):
        """Salva pesos e vieses em arquivo numpy .npz"""
        np.savez(filepath, weights=self.weights, biases=self.biases, architecture=self.architecture)

    def load_parameters(self, filepath: str):
        """Carrega parâmetros salvos em arquivo .npz"""
        data = np.load(filepath, allow_pickle=True)
        self.weights = [np.array(x) for x in data['weights']]
        self.biases = [np.array(x) for x in data['biases']]
        self.architecture = list(data['architecture'])

# End of file
