import transformers
from model import GPT2Model
import numpy as np
import torch
import argparse
import time
import pandas as pd
from tqdm import tqdm


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
    parser.add_argument("--num_tokens", type=int, default=1010)
    parser.add_argument("--skip", type=int, default=50)
    parser.add_argument("--hf", action="store_true")
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    hf_model, model = get_models()

    time_taken = []
    for _ in tqdm(range(1, args.num_tokens)):
        if _ % args.skip == 0:
            start = time.time()
            input_ids = torch.randint(0, model.config.vocab_size, (1, _))
            if args.hf:
                hf_model(input_ids)
            else:    
                model(input_ids.numpy())
            time_taken.append([_, time.time() - start])
    pd.DataFrame(time_taken, columns=["idx", "time_taken"]).to_csv(args.output_file, index=None)



if __name__ == "__main__":
    main()