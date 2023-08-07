import transformers
from model import GPT2Model
import numpy as np
import torch
import argparse
import time

def get_models():
    hf_model = transformers.AutoModelForCausalLM.from_pretrained("gpt2", output_hidden_states=True, output_attentions=True)
    params = {k: v.detach().numpy() for k,v in hf_model.named_parameters()}

    gpt2model = GPT2Model(hf_model.config)
    prefix = "transformer."
    selected_weights = {k.split(prefix)[1]: v for k, v in params.items() if k.startswith(prefix)}
    gpt2model.__init__weights__(selected_weights)

    return hf_model, gpt2model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The capital of USA is")
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--hf", action="store_true")
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    hf_model, model = get_models()

    prompt = args.prompt
    tokenized_prompt = tokenizer(prompt).input_ids
    print(prompt, end='\r')
    time_taken = []
    for _ in range(args.num_tokens):
        start = time.time()
        if args.hf:
            tokenized = tokenizer(prompt, return_tensors="pt")
            out = hf_model(tokenized.input_ids).logits
            argmax_token_id = torch.argmax(out, axis=2)[0][-1]
        else:    
            tokenized = tokenizer(prompt, return_tensors="np")
            out = model(tokenized.input_ids)
            argmax_token_id = np.argmax(out, axis=2)[0][-1]
        tokenized_prompt = tokenized_prompt + [argmax_token_id]
        prompt = tokenizer.decode(tokenized_prompt)
        print(prompt, end='\r')
        time_taken.append([_, time.time() - start])
    print()
    print("\n".join([",".join([str(tt) for tt in t]) for t in time_taken]))


if __name__ == "__main__":
    main()