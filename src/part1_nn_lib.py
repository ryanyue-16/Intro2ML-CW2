import numpy as np

class LinearLayer:


    def __init__(self, n_in: int, n_out: int):
        #y = X W + b, X: (batch_size, n_in) W: (n_in, n_out) b: (n_out,)
        
        try:
            self._W = xavier_init(n_in, n_out)  
        except NameError:
            
            limit = np.sqrt(6.0 / (n_in + n_out))
            self._W = np.random.uniform(-limit, limit, size=(n_in, n_out))
        self._b = np.zeros((n_out,), dtype=self._W.dtype)

        # Caches / grads (set during forward/backward)
        self._cache_current = None          # will store inputs X
        self._grad_W_current = np.zeros_like(self._W)
        self._grad_b_current = np.zeros_like(self._b)

        # Save dims (handy for checks/debugging)
        self.n_in = n_in
        self.n_out = n_out

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        #Vectorized forward pass

        # Cache inputs needed for gradient computation
        self._cache_current = inputs

        # Affine transform: XW + b (broadcast b across batch dim)
        return inputs @ self._W + self._b

    def backward(self, grad_outputs: np.ndarray) -> np.ndarray:
        
        #Vectorized backward pass given dL/dY = grad_outputs (B, n_out).

        X = self._cache_current  

        
        # dL/dW = X^T * dL/dY
        self._grad_W_current = X.T @ grad_outputs  

        # dL/db = sum over batch of dL/dY
        self._grad_b_current = np.sum(grad_outputs, axis=0) 

        # Gradient w.r.t. inputs: dL/dX = dL/dY * W^T
        grad_inputs = grad_outputs @ self._W.T  
        return grad_inputs

    def update_params(self, learning_rate: float) -> None:
        
        #gradient descent using stored grads.
        self._W -= learning_rate * self._grad_W_current
        self._b -= learning_rate * self._grad_b_current
        
class SigmoidLayer:
    def __init__(self):
        self._cache_current = None  

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # numerically stable sigmoid
        out = np.empty_like(x, dtype=np.float64)
        pos = x >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[neg])
        out[neg] = ex / (1.0 + ex)
        return out

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        s = self._sigmoid(inputs)
        self._cache_current = s
        return s

    def backward(self, grad_outputs: np.ndarray) -> np.ndarray:
        s = self._cache_current
        # dσ/dx = σ(x)(1-σ(x))
        return grad_outputs * s * (1.0 - s)

    def update_params(self, learning_rate: float) -> None:
        # no learnable parameters
        return


class ReluLayer:
    
    def __init__(self):
        self._cache_current = None  # cache inputs to make the mask

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._cache_current = inputs
        return np.maximum(inputs, 0)

    def backward(self, grad_outputs: np.ndarray) -> np.ndarray:
        x = self._cache_current
        mask = (x > 0).astype(grad_outputs.dtype)
        return grad_outputs * mask

    def update_params(self, learning_rate: float) -> None:
        # no learnable parameters
        return        



class MultiLayerNetwork:
   
    def __init__(self, input_dim: int, neurons, activations):
        if neurons is None or len(neurons) == 0:
            raise ValueError("`neurons` must be a non-empty list of output sizes.")

        
        if activations is None:
            activations = [None] * len(neurons)
        if len(activations) != len(neurons):
            raise ValueError("`activations` must have the same length as `neurons`.")

        self._layers = []  
        n_in = int(input_dim)

        def _make_activation(act):
            if act is None:
                return None
            if not isinstance(act, str):
                # allow passing a layer instance (optional flexibility)
                return act
            a = act.lower()
            if a in ("relu",):
                return ReluLayer()
            if a in ("sigmoid", "logistic"):
                return SigmoidLayer()
            if a in ("identity", "linear", "none"):
                return None
            raise ValueError(f"Unknown activation '{act}'")

        for n_out, act in zip(neurons, activations):
            # linear layer
            self._layers.append(LinearLayer(n_in=n_in, n_out=int(n_out)))
            # optional activation
            act_layer = _make_activation(act)
            if act_layer is not None:
                self._layers.append(act_layer)
            n_in = int(n_out)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        x = inputs
        for layer in self._layers:
            x = layer(x)  # delegates to each layer's forward
        return x

    def backward(self, grad_outputs: np.ndarray) -> np.ndarray:
        g = grad_outputs
        for layer in reversed(self._layers):
            g = layer.backward(g)  # each layer returns grad wrt its inputs
        return g

    def update_params(self, learning_rate: float) -> None:
        for layer in self._layers:
            # activation layers have a no-op update_params; safe to call
            layer.update_params(learning_rate)

