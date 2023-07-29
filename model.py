from dataclasses import dataclass
import numpy as np
import math
from scipy.special import erf

@dataclass
class GPT2Config():
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int = None
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    summary_type: str = "cls_index"
    summary_use_proj: bool = True
    summary_activation: str = None
    summary_proj_to_labels: bool = True
    summary_first_dropout: float = 0.1
    scale_attn_weights: bool = True
    use_cache: bool = True
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False

class Base:
    def __init__(self):
        pass
    
    def process_weights(self, weights):
        weights_copy = {}
        for key, value in weights.items():
            key_split = key.split('.')
            base_key = key_split[0]
            remaining_key = key_split[1:]
            if len(key_split) >= 2:
                if base_key not in weights_copy:
                    weights_copy[base_key] = {}
                weights_copy[base_key]['.'.join(remaining_key)] = value
            else:
                assert self.__dict__[key].shape == value.shape
                self.__dict__[key] = value
        return weights_copy

    def __init__weights__(self, weights):
        weights_copy = self.process_weights(weights)
        for key, value in weights_copy.items():
            self.__dict__[key].__init__weights__(value)

class List(Base):
    def __init__(self, items):
        self.items = items

    def __init__weights__(self, weights):
        weights_copy = self.process_weights(weights)
        weights_copy = sorted([[int(k), v] for k,v in weights_copy.items()])
        for key, value in weights_copy:
            self.items[key].__init__weights__(value)
    
    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def __call__(self, *args):
        for item in self.items:
            if type(args) != tuple:
                args = (args,)
            args = item(*args)
        return args


class GPT2Model(Base):
    def __init__(self, config: GPT2Config):
        self.config = config

        self.wte = Embedding(self.config.vocab_size, self.config.n_embd)
        self.wpe = Embedding(self.config.n_positions, self.config.n_embd)

        self.h = List([GPT2Layer(config) for _ in range(self.config.n_layer)])
        self.ln_f = LayerNorm(self.config.hidden_size)
        self.lm_head = Linear(self.config.n_embd, self.config.vocab_size, bias_present=False)
    
    def __call__(self, input_ids):
        input_embeddings = self.wte(input_ids) + self.wpe(np.tile(np.arange(input_ids.shape[-1]), (input_ids.shape[:-1] + (1, ) )))
        hidden_states = self.h(input_embeddings)
        return self.lm_head(hidden_states)
    
    def __init__weights__(self, weights):
        super().__init__weights__(weights)
        self.lm_head.weight = self.wte.weight.T


class GPT2Layer(Base):
    def __init__(self, config):
        self.config = config

        self.ln_1 = LayerNorm(self.config.n_embd)
        self.attn = Attention(self.config.n_embd, self.config.n_head)
        self.ln_2 = LayerNorm(self.config.n_embd)
        self.mlp = TransformerMLP(self.config.n_embd, 4 * self.config.n_embd)
    
    def __call__(self, inputs):
        inputs = inputs + self.attn(self.ln_1(inputs))[0]
        inputs = inputs + self.mlp(self.ln_2(inputs))
        return inputs


class TransformerMLP(Base):
    def __init__(self, n_outer, n_inner):
        self.n_outer = n_outer
        self.n_inner = n_inner

        self.c_fc = Linear(self.n_outer, self.n_inner)
        self.c_proj = Linear(self.n_inner, self.n_outer)
    
    def __call__(self, inp):
        return self.c_proj(gelu(self.c_fc(inp)))
    

class Linear(Base):
    def __init__(self, n_in, n_out, bias_present=True):
        self.n_in = n_in
        self.n_out = n_out
        self.bias_present = bias_present

        self.weight = np.ndarray((n_in, n_out), dtype=np.float32)
        self.bias = np.ndarray((n_out), dtype=np.float32) if self.bias_present else None
    
    def __call__(self, inp):
        inp = np.matmul(inp, self.weight)
        if self.bias_present:
            inp += self.bias
        return inp

class LayerNorm(Base):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

        self.weight = np.ndarray((self.hidden_size), dtype=np.float32)
        self.bias = np.ndarray((self.hidden_size), dtype=np.float32)

    def __call__(self, inp):
        mean = np.mean(inp, axis=-1, keepdims=True)
        std = np.std(inp, axis=-1, keepdims=True)

        return (self.weight/std)*(inp - mean) + self.bias

class Attention(Base):
    def __init__(self, n_embd, n_head):
        self.n_embd = n_embd
        self.n_head = n_head

        self.c_attn = Linear(self.n_embd, 3 * self.n_embd)
        self.c_proj = Linear(self.n_embd, self.n_embd)
    
    def __call__(self, inputs):
        c_attn = self.c_attn(inputs)
        batch_size = inputs.shape[0]
        seq_length = inputs.shape[1]

        causal_mask =  np.tile(np.expand_dims(np.triu(np.ones(seq_length), k=1), (0, 1)), ((batch_size, self.n_head, 1, 1)))

        # c_attn_reshaped = c_attn.reshape(batch_size, seq_length, self.n_head, 3, self.n_embd // self.n_head).transpose(3, 0, 2, 1, 4)
        # q, k, v  = c_attn_reshaped
        # c_attn_reshaped = c_attn.reshape(batch_size, seq_length, self.n_head, self.n_embd // self.n_head, 3).transpose(4, 0, 2, 1, 3)
        # q, k, v = c_attn_reshaped
        # c_attn_reshaped = c_attn.reshape(batch_size, seq_length, 3, self.n_embd // self.n_head, self.n_head).transpose(2, 0, 4, 1, 3)
        # q, k, v = c_attn_reshaped
        c_attn_reshaped = c_attn.reshape(batch_size, seq_length, 3, self.n_head, self.n_embd // self.n_head).transpose(2, 0, 3, 1, 4)
        q, k, v = c_attn_reshaped
        # c_attn_reshaped = c_attn.reshape(batch_size, seq_length, self.n_embd // self.n_head, 3, self.n_head).transpose(3, 0, 4, 1, 2)
        # q, k, v = c_attn_reshaped
        # c_attn_reshaped = c_attn.reshape(batch_size, seq_length, self.n_embd // self.n_head, self.n_head, 3).transpose(4, 0, 3, 1, 2)
        # q, k, v = c_attn_reshaped
        # (b, h, l, d)

        matmul = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.n_embd // self.n_head)
        matmul = np.ma.array(matmul, mask=causal_mask).filled(fill_value=-np.inf)
        weights = softmax(matmul, axis=-1)
        # (b, h, l, l)

        weighted_sum = np.matmul(weights, v)
        # (b, h, l, d)

        concatenated_weighted_sum=weighted_sum.transpose(0,2,1,3).reshape(batch_size, seq_length, -1)

        return self.c_proj(concatenated_weighted_sum), weights


class Embedding(Base):
    def __init__(self, vocab_size, n_embd):
        self.vocab_size = vocab_size
        self.n_embd = n_embd

        self.weight = np.ndarray((vocab_size, n_embd), dtype=np.float32)
    
    def __call__(self, inp):
        return self.weight[inp]

def sigmoid(inp):
    return 1 / (1 + np.exp(-inp))

def softmax(inp, axis=-1):
    inp = inp - np.max(inp, axis=axis, keepdims=True)
    inp_exp = np.exp(inp)
    return inp_exp / np.sum(inp_exp, axis=axis, keepdims=True)

def gelu(inp):
    # return inp * sigmoid(1.702 * inp)
    # return inp * phi(inp)
    # return 0.5 * inp * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (inp + 0.044715 * inp**3)))
    return 0.5 * inp * (1.0 + erf(inp / np.sqrt(2.0)))
