import argparse
import gc
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--aligned-model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--save-path", type=str, default="./lox-model")
parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--k", type=int, default=6)
parser.add_argument("--coef", type=float, default=1.25)
args = parser.parse_args()
print(args)

@torch.no_grad()
def main():
    k = args.k
    coef = args.coef

    tokenizer = AutoTokenizer.from_pretrained(args.aligned_model, use_fast=False)

    # Reduce peak RAM while loading
    aligned_model = AutoModelForCausalLM.from_pretrained(
        args.aligned_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # state_dict() is a shallow mapping of references, so modifying a tensor
    # from aligned_sd modifies the aligned_model in memory.
    aligned_sd = aligned_model.state_dict()
    base_sd = base_model.state_dict()

    for name in tqdm(aligned_sd.keys()):
        wa = aligned_sd[name]
        wb = base_sd[name]

        if wa.ndim > 1:
            if k > 0:
                # Keep same math as original: SVD is done in float32
                dW = wa.float() - wb.float()
                U, S, Vt = torch.linalg.svd(dW, full_matrices=False)
                S[k:] = 0
                m = U @ torch.diag(S) @ Vt

                # Original code effectively ends up back in model dtype when loaded
                updated = (wa.float() + coef * m).to(dtype=wa.dtype)
                wa.copy_(updated)

                del dW, U, S, Vt, m, updated
            else:
                # Keep k=0 behavior as close as possible to original
                m = wa - wb
                updated = wa + coef * m
                wa.copy_(updated)

                del m, updated

        del wa, wb
        gc.collect()

    del base_model, base_sd
    gc.collect()

    aligned_model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

if __name__ == "__main__":
    main()

