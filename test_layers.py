import transformers
from model import Linear, TransformerMLP, GPT2Layer, GPT2Model, gelu
import numpy as np
import torch

def get_models():
    hf_model = transformers.AutoModelForCausalLM.from_pretrained("gpt2", output_attentions=True, output_hidden_states=True)
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
    assert np.allclose(np_out, pt_out)

def test_gelu():
    random_input = torch.randn((32, 32))
    pt_out = torch.nn.GELU(approximate='none')(random_input).numpy()
    np_out = gelu(random_input.numpy())

    assert np.allclose(np_out, pt_out)

def test_linear():
    hf_model, gpt2model = get_models()

    batch_size = 32
    max_len = 64
    random_input = torch.randn((batch_size, max_len, hf_model.config.n_embd))
    hf_model.eval()
    with torch.no_grad():
        pt_out = hf_model.transformer.h[0].mlp.c_fc(random_input).detach().numpy()

    np_out = gpt2model.h[0].mlp.c_fc(random_input.numpy())
    # np_out = hf_model.transformer.h[0].mlp(random_input).detach().numpy()
    assert np.allclose(np_out, pt_out, rtol=1e-1)

def test_mlp():
    hf_model, gpt2model = get_models()

    batch_size = 32
    max_len = 64
    random_input = torch.randn((batch_size, max_len, hf_model.config.n_embd))
    # random_input = torch.load("random_input.pt")
    pt_out = hf_model.transformer.h[0].mlp(random_input).detach().numpy()

    np_out = gpt2model.h[0].mlp(random_input.numpy())
    assert np.allclose(np_out, pt_out)


if __name__ == "__main__":
    test_linear()