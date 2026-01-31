import argparse
import math

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and evaluate OLMo with perplexity.")
    parser.add_argument(
        "--model",
        default="allenai/OLMo-1B-hf",
        help="HF model id or local path (default: allenai/OLMo-1B-hf)",
    )
    parser.add_argument(
        "--dataset",
        default="wikitext",
        help="HF dataset name (default: wikitext)",
    )
    parser.add_argument(
        "--dataset-config",
        default="wikitext-2-raw-v1",
        help="HF dataset config (default: wikitext-2-raw-v1)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Max sequence length for evaluation windows",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1024,
        help="Stride for sliding window evaluation",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (e.g., cpu, cuda). Default: auto",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype (default: auto)",
    )
    return parser.parse_args()


def pick_dtype(dtype_str: str, device: torch.device) -> torch.dtype:
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    # auto
    return torch.float16 if device.type == "cuda" else torch.float32


@torch.no_grad()
def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts,
    max_seq_len: int,
    stride: int,
    device: torch.device,
) -> float:
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    nll_sum = 0.0
    n_tokens = 0

    for start in range(0, input_ids.size(1), stride):
        end = min(start + max_seq_len, input_ids.size(1))
        input_ids_window = input_ids[:, start:end]
        target_ids = input_ids_window.clone()

        # Ignore loss on all tokens except the last (end-start) tokens
        target_ids[:, :-1] = -100

        outputs = model(input_ids_window, labels=target_ids)
        neg_log_likelihood = outputs.loss

        # Number of predicted tokens is window length - 1
        n_tokens_window = input_ids_window.size(1) - 1
        nll_sum += neg_log_likelihood.item() * n_tokens_window
        n_tokens += n_tokens_window

        if end == input_ids.size(1):
            break

    return math.exp(nll_sum / max(n_tokens, 1))


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = pick_dtype(args.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    texts = [t for t in dataset["text"] if t.strip()]

    ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_seq_len=args.max_seq_len,
        stride=args.stride,
        device=device,
    )

    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}/{args.dataset_config} [{args.split}]")
    print(f"Perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()
