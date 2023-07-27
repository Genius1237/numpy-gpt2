import transformers
from model import Linear, TransformerMLP, GPT2Layer, GPT2Model

def test_linear_init_weights():
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    params = {k: v.detach().numpy() for k,v in model.named_parameters()}

    linear = Linear(model.config.n_embd, model.config.vocab_size)
    prefix = "transformer.ln_f."
    selected_weights = {k.split(prefix)[1]: v for k, v in params.items() if k.startswith(prefix)}
    linear.__init__weights__(selected_weights)

    assert (linear.weight == params[prefix + "weight"]).all()
    assert (linear.bias == params[prefix + "bias"]).all()

def test_transformermlp_init_weights():
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    params = {k: v.detach().numpy() for k,v in model.named_parameters()}

    mlp = TransformerMLP(model.config.n_embd, 4 * model.config.n_embd)
    prefix = "transformer.h.0.mlp."
    selected_weights = {k.split(prefix)[1]: v for k, v in params.items() if k.startswith(prefix)}
    mlp.__init__weights__(selected_weights)

    assert (mlp.c_fc.weight == params[prefix + "c_fc.weight"]).all()
    assert (mlp.c_fc.bias == params[prefix + "c_fc.bias"]).all()
    assert (mlp.c_proj.weight == params[prefix + "c_proj.weight"]).all()
    assert (mlp.c_proj.bias == params[prefix + "c_proj.bias"]).all()

def test_gpt2layer_init_weights():
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    params = {k: v.detach().numpy() for k,v in model.named_parameters()}

    gpt2layer = GPT2Layer(model.config)
    prefix = "transformer.h.0."
    selected_weights = {k.split(prefix)[1]: v for k, v in params.items() if k.startswith(prefix)}
    gpt2layer.__init__weights__(selected_weights)

    assert (gpt2layer.ln_1.weight == params[prefix + "ln_1.weight"]).all()
    assert (gpt2layer.ln_1.bias == params[prefix + "ln_1.bias"]).all()
    assert (gpt2layer.ln_2.weight == params[prefix + "ln_2.weight"]).all()
    assert (gpt2layer.ln_2.bias == params[prefix + "ln_2.bias"]).all()
    assert (gpt2layer.attn.c_attn.weight == params[prefix + "attn.c_attn.weight"]).all()
    assert (gpt2layer.attn.c_attn.bias == params[prefix + "attn.c_attn.bias"]).all()
    assert (gpt2layer.attn.c_proj.weight == params[prefix + "attn.c_proj.weight"]).all()
    assert (gpt2layer.attn.c_proj.bias == params[prefix + "attn.c_proj.bias"]).all()
    assert (gpt2layer.mlp.c_fc.weight == params[prefix + "mlp.c_fc.weight"]).all()
    assert (gpt2layer.mlp.c_fc.bias == params[prefix + "mlp.c_fc.bias"]).all()
    assert (gpt2layer.mlp.c_proj.weight == params[prefix + "mlp.c_proj.weight"]).all()
    assert (gpt2layer.mlp.c_proj.bias == params[prefix + "mlp.c_proj.bias"]).all()

def test_gpt2model_init_weights():
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    params = {k: v.detach().numpy() for k,v in model.named_parameters()}

    gpt2model = GPT2Model(model.config)
    prefix = "transformer."
    selected_weights = {k.split(prefix)[1]: v for k, v in params.items() if k.startswith(prefix)}
    gpt2model.__init__weights__(selected_weights)

    for i, gpt2layer in enumerate(gpt2model.h):
        assert (gpt2layer.ln_1.weight == params[prefix + "h." + str(i) + "." + "ln_1.weight"]).all()
        assert (gpt2layer.ln_1.bias == params[prefix + "h." + str(i) + "." + "ln_1.bias"]).all()
        assert (gpt2layer.ln_2.weight == params[prefix + "h." + str(i) + "." + "ln_2.weight"]).all()
        assert (gpt2layer.ln_2.bias == params[prefix + "h." + str(i) + "." + "ln_2.bias"]).all()
        assert (gpt2layer.attn.c_attn.weight == params[prefix + "h." + str(i) + "." + "attn.c_attn.weight"]).all()
        assert (gpt2layer.attn.c_attn.bias == params[prefix + "h." + str(i) + "." + "attn.c_attn.bias"]).all()
        assert (gpt2layer.attn.c_proj.weight == params[prefix + "h." + str(i) + "." + "attn.c_proj.weight"]).all()
        assert (gpt2layer.attn.c_proj.bias == params[prefix + "h." + str(i) + "." + "attn.c_proj.bias"]).all()
        assert (gpt2layer.mlp.c_fc.weight == params[prefix + "h." + str(i) + "." + "mlp.c_fc.weight"]).all()
        assert (gpt2layer.mlp.c_fc.bias == params[prefix + "h." + str(i) + "." + "mlp.c_fc.bias"]).all()
        assert (gpt2layer.mlp.c_proj.weight == params[prefix + "h." + str(i) + "." + "mlp.c_proj.weight"]).all()
        assert (gpt2layer.mlp.c_proj.bias == params[prefix + "h." + str(i) + "." + "mlp.c_proj.bias"]).all()
    
    assert (gpt2model.ln_f.weight == params[prefix + "ln_f.weight"]).all()
    assert (gpt2model.ln_f.bias == params[prefix + "ln_f.bias"]).all()
    assert (gpt2model.lm_head.weight == params[prefix + "wte.weight"].T).all()
    assert (gpt2model.wte.weight == params[prefix + "wte.weight"]).all()
    assert (gpt2model.wpe.weight == params[prefix + "wpe.weight"]).all()

if __name__ == "__main__":
    test_gpt2model_init_weights()