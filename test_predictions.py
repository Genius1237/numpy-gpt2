import transformers
from model import GPT2Model
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


def test_argmax():
    hf_model, gpt2model = get_models()

    batch_size = 32
    max_len = 65
    random_input = torch.randint(0, hf_model.config.vocab_size, (batch_size,max_len))
    pt_out = hf_model(random_input)[0].detach().numpy()
    np_out = gpt2model(random_input.numpy())
    assert (np.argmax(pt_out, axis=2) == np.argmax(np_out, axis=2)).all()


def test_topk():
    hf_model, gpt2model = get_models()

    batch_size = 32
    max_len = 65
    random_input = torch.randint(0, hf_model.config.vocab_size, (batch_size,max_len))
    pt_out = hf_model(random_input)[0].detach().numpy()
    np_out = gpt2model(random_input.numpy())
    assert (np.argmax(pt_out, axis=2) == np.argmax(np_out, axis=2)).all()


if __name__ == "__main__":
    test_argmax()