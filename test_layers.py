import transformers
from model import Linear, TransformerMLP, GPT2Layer, GPT2Model, gelu
import numpy as np
import torch

def get_models():
    hf_model = transformers.AutoModelForCausalLM.from_pretrained("gpt2", output_hidden_states=True, output_attentions=True)
    params = {k: v.detach().numpy() for k,v in hf_model.named_parameters()}

    gpt2model = GPT2Model(hf_model.config)
    prefix = "transformer."
    selected_weights = {k.split(prefix)[1]: v for k, v in params.items() if k.startswith(prefix)}
    gpt2model.__init__weights__(selected_weights)

    return hf_model, gpt2model

def test_lmhead():
    hf_model, gpt2model = get_models()

    batch_size = 32
    max_len = 64
    random_inputs = torch.randint(0, hf_model.config.vocab_size, (batch_size,max_len))
    model_out = hf_model(input_ids = random_inputs)
    last_hidden_states = model_out.hidden_states[-1].detach().numpy()
    pt_out = model_out.logits.detach().numpy()

    np_out = gpt2model.lm_head(last_hidden_states)
    assert np.allclose(np_out, pt_out)

def test_lmhead_random_inputs():
    hf_model, gpt2model = get_models()

    batch_size = 32
    max_len = 64
    
    random_input = torch.randn((batch_size, max_len, hf_model.config.n_embd))
    pt_out = hf_model.lm_head(random_input).detach().numpy()
    np_out = gpt2model.lm_head(random_input.numpy())
    assert np.allclose(np_out, pt_out, atol=1e-5)

def test_gelu():
    random_input = torch.randn((32, 32))
    pt_out = torch.nn.GELU(approximate='none')(random_input).numpy()
    np_out = gelu(random_input.numpy())

    assert np.allclose(np_out, pt_out)

def test_linear():
    hf_model, gpt2model = get_models()

    inputs = []
    outputs = []
    def hook_fn(m, i, o):
        inputs.append(i)
        outputs.append(o)

    hf_model.transformer.h[0].mlp.c_fc.register_forward_hook(hook_fn)
    
    batch_size = 32
    max_len = 64
    random_inputs = torch.randint(0, hf_model.config.vocab_size, (batch_size,max_len))
    hf_model(input_ids = random_inputs)
    
    pt_out = outputs[0].detach().numpy()

    np_out = gpt2model.h[0].mlp.c_fc(inputs[0][0].detach().numpy())
    assert np.allclose(np_out, pt_out, atol=1e-5)


def test_linear_random_inputs():
    hf_model, gpt2model = get_models()

    batch_size = 32
    max_len = 64
    random_input = torch.randn((batch_size, max_len, hf_model.config.n_embd))
    pt_out = hf_model.transformer.h[0].mlp.c_fc(random_input).detach().numpy()

    np_out = gpt2model.h[0].mlp.c_fc(random_input.numpy())
    assert np.allclose(np_out, pt_out, atol=1e-5)

def test_mlp():
    hf_model, gpt2model = get_models()

    inputs = []
    outputs = []
    def hook_fn(m, i, o):
        inputs.append(i)
        outputs.append(o)

    hf_model.transformer.h[0].mlp.register_forward_hook(hook_fn)
    
    batch_size = 32
    max_len = 64
    random_inputs = torch.randint(0, hf_model.config.vocab_size, (batch_size,max_len))
    hf_model(input_ids = random_inputs)
    
    pt_out = outputs[0].detach().numpy()

    np_out = gpt2model.h[0].mlp(inputs[0][0].detach().numpy())
    assert np.allclose(np_out, pt_out, atol=1e-2)


def test_mlp_random_inputs():
    hf_model, gpt2model = get_models()

    batch_size = 32
    max_len = 64
    random_input = torch.randn((batch_size, max_len, hf_model.config.n_embd))
    pt_out = hf_model.transformer.h[0].mlp(random_input).detach().numpy()

    np_out = gpt2model.h[0].mlp(random_input.numpy())
    assert np.allclose(np_out, pt_out, atol=1e-5)


def test_matmul():
    mat1 = torch.randn((32, 32))
    mat2 = torch.randn((32, 32))
    res = torch.matmul(mat1, mat2).numpy()

    np_mat1 = mat1.numpy()
    np_mat2 = mat2.numpy()
    np_res = np.matmul(np_mat1, np_mat2)

    assert np.allclose(res, np_res, atol=1e-5)

def test_layernorm():
    hf_model, gpt2model = get_models()

    batch_size = 32
    max_len = 64
    random_input = torch.randn((batch_size, max_len, hf_model.config.n_embd))
    pt_out = hf_model.transformer.h[0].ln_1(random_input).detach().numpy()

    np_out = gpt2model.h[0].ln_1(random_input.numpy())
    assert np.allclose(np_out, pt_out, atol=1e-5)

def test_attention():
    hf_model, gpt2model = get_models()

    inputs = []
    outputs = []
    def hook_fn(m, i, o):
        inputs.append(i)
        outputs.append(o)

    hf_model.transformer.h[0].attn.register_forward_hook(hook_fn)

    batch_size = 32
    max_len = 65
    # random_input = torch.randn((batch_size, max_len, hf_model.config.n_embd))
    # pt_out, pt_attention_weights = (t.detach().numpy() for t in hf_model.transformer.h[0].attn(random_input))
    random_inputs = torch.randint(0, hf_model.config.vocab_size, (batch_size,max_len))
    # random_inputs = torch.load("random_inputs.pt")
    pt_attention_weights = hf_model(input_ids=random_inputs).attentions[0].detach().numpy()
    pt_out = outputs[0][0].detach().numpy()

    np_out, np_attention_weights = gpt2model.h[0].attn(inputs[0][0].detach().numpy())
    print(pt_attention_weights[0][0], np_attention_weights[0][0])
    assert np.allclose(pt_attention_weights, np_attention_weights, atol=1e-5)
    assert np.allclose(pt_out, np_out, atol=1e-5)

if __name__ == "__main__":
    test_attention()