"""
Script 3: Generate Embeddings from Prompts using C2S-Scale Model

This script generates embeddings from text prompts using a local C2S-Scale model.
Optimized for large-scale batch processing with memory management.

Usage:
    python 3_gen_embedding.py --prompts_file prompts.csv --output_dir ./embeddings
    python 3_gen_embedding.py --test_mode --num_test_samples 10
"""

import torch
from transformers import GemmaTokenizer, AutoModel
import os
import time
import numpy as np
import pandas as pd
import argparse

from transformers import BitsAndBytesConfig


def load_model_and_tokenizer(model_path):
    """Load model and tokenizer with optimized settings for large batch processing."""
    print("--- Loading Model and Tokenizer ---")
    start_time = time.time()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Multi-GPU inference requires GPUs.")
    
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} GPUs.")

    print(f"Loading GemmaTokenizer from: {model_path}")
    
    tokenizer_file = os.path.join(model_path, "tokenizer.model")
    if not os.path.exists(tokenizer_file):
        raise FileNotFoundError(f"tokenizer.model not found at {tokenizer_file}.")

    tokenizer = GemmaTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Warning: pad_token was not set. Manually set to eos_token.")

    print("Tokenizer loaded successfully.")
    
    print(f"Loading model from: {model_path}")
    print("This may take a few minutes...")
    
    # Optimized quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModel.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
        use_cache=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    model.eval()

    end_time = time.time()
    print(f"Model and tokenizer loaded successfully in {end_time - start_time:.2f} seconds.")
    print("-" * 40)
    return model, tokenizer


def get_embeddings_batched(prompts, model, tokenizer, batch_size=1, max_length=4096):
    """Extract embeddings with optimized memory handling for long sequences."""
    all_embeddings = []
    total_batches = (len(prompts) - 1) // batch_size + 1
    
    log_interval = max(1, total_batches // 100)
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        current_batch_num = i // batch_size + 1
        
        if current_batch_num % log_interval == 0 or current_batch_num == 1:
            progress = (current_batch_num / total_batches) * 100
            print(f"Progress: {progress:.1f}% - Batch {current_batch_num}/{total_batches}")

        # Tokenize
        tokens = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        input_ids = tokens['input_ids'].to(model.device)
        attention_mask = tokens['attention_mask'].to(model.device)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

            last_hidden_states = outputs.hidden_states[-1]
            
            # Masked mean pooling
            expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
            sum_embeddings = (last_hidden_states * expanded_mask).sum(1)
            sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).cpu().half()
            all_embeddings.append(batch_embeddings.numpy())
            
            # Memory cleanup
            del outputs, last_hidden_states, input_ids, attention_mask, tokens
            del expanded_mask, sum_embeddings, sum_mask, batch_embeddings
            
        if current_batch_num % 10 == 0:
            torch.cuda.empty_cache()
    
    return np.vstack(all_embeddings).astype(np.float32)


def smart_batching_by_length(prompts):
    """Sort prompts by length for efficient batching."""
    print("--- Sorting prompts by length for efficient batching ---")
    
    prompt_lengths = []
    for i, prompt in enumerate(prompts):
        if i % 10000 == 0:
            print(f"Sorting progress: {i}/{len(prompts)}")
        length = len(prompt)
        prompt_lengths.append((prompt, length, i))
    
    prompt_lengths.sort(key=lambda x: x[1])
    
    sorted_prompts = [p[0] for p in prompt_lengths]
    lengths = [p[1] for p in prompt_lengths]
    original_indices = [p[2] for p in prompt_lengths]
    
    print(f"\nSorting completed:")
    print(f"  Shortest: {lengths[0]} characters")
    print(f"  Longest: {lengths[-1]} characters")
    print(f"  Median: {np.median(lengths):.0f} characters")
    
    return sorted_prompts, original_indices


def process_large_dataset_chunked(
    model, tokenizer, prompts, batch_size, max_length, 
    output_dir, chunk_size=5000, smart_batching=False
):
    """Process data in chunks with periodic saving to avoid memory issues."""
    os.makedirs(output_dir, exist_ok=True)
    
    total_chunks = (len(prompts) - 1) // chunk_size + 1
    print(f"Total prompts: {len(prompts)}")
    print(f"Processing in {total_chunks} chunks of {chunk_size} samples each")
    print(f"Smart batching: {'ENABLED' if smart_batching else 'DISABLED'}")
    print("=" * 60)
    
    if smart_batching:
        print("\n--- Starting smart batching preprocessing ---")
        sorted_prompts, original_indices = smart_batching_by_length(prompts)
        print("Smart batching preprocessing completed.")
        print("=" * 60)
    else:
        sorted_prompts = prompts
        original_indices = list(range(len(prompts)))
    
    all_embeddings_sorted = []
    
    for chunk_idx in range(0, len(sorted_prompts), chunk_size):
        chunk_end = min(chunk_idx + chunk_size, len(sorted_prompts))
        chunk_prompts = sorted_prompts[chunk_idx:chunk_end]
        chunk_num = chunk_idx // chunk_size + 1
        
        print(f"\n{'='*60}")
        print(f"CHUNK {chunk_num}/{total_chunks}: Processing samples {chunk_idx} to {chunk_end}")
        print(f"{'='*60}")
        
        chunk_start_time = time.time()
        
        embeddings = get_embeddings_batched(
            chunk_prompts, model, tokenizer, batch_size, max_length
        )
        
        chunk_time = time.time() - chunk_start_time
        samples_per_sec = len(chunk_prompts) / chunk_time
        
        chunk_file = os.path.join(
            output_dir, 
            f"embeddings_chunk_sorted_{chunk_idx:06d}_{chunk_end:06d}.npy"
        )
        np.save(chunk_file, embeddings)
        all_embeddings_sorted.append(embeddings)
        
        print(f"\nChunk {chunk_num} completed:")
        print(f"  - Time: {chunk_time:.2f}s")
        print(f"  - Speed: {samples_per_sec:.2f} samples/sec")
        print(f"  - Saved to: {chunk_file}")
        print(f"  - Shape: {embeddings.shape}")
        
        remaining_samples = len(sorted_prompts) - chunk_end
        estimated_time_remaining = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0
        print(f"  - Estimated time remaining: {estimated_time_remaining/3600:.2f} hours")
        
        del embeddings
        torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print("All chunks processed!")
    print(f"{'='*60}")
    
    # Merge all embeddings
    print("\nMerging all chunks...")
    merged_embeddings_sorted = np.vstack(all_embeddings_sorted)
    
    if smart_batching:
        print("Restoring original order...")
        merged_embeddings = np.zeros_like(merged_embeddings_sorted)
        for sorted_idx, original_idx in enumerate(original_indices):
            merged_embeddings[original_idx] = merged_embeddings_sorted[sorted_idx]
        
        merged_file = os.path.join(output_dir, "all_embeddings_merged.npy")
        np.save(merged_file, merged_embeddings)
        print(f"\nMerged embeddings (original order) saved to: {merged_file}")
        print(f"  Final shape: {merged_embeddings.shape}")
    else:
        merged_file = os.path.join(output_dir, "all_embeddings_merged.npy")
        np.save(merged_file, merged_embeddings_sorted)
        print(f"\nMerged embeddings saved to: {merged_file}")
        print(f"  Final shape: {merged_embeddings_sorted.shape}")
    
    print(f"  Expected samples: {len(prompts)}")
    
    return merged_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings from prompts using C2S-Scale model."
    )
    
    parser.add_argument(
        '--model_path', 
        type=str,
        default="./model/c2s-scale-gemma-2",
        help="Path to C2S-Scale model"
    )
    parser.add_argument(
        '--prompts_file', 
        type=str,
        default="./results/gdsc_ccle_mechanism_prompts.csv",
        help="Path to prompts CSV file"
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        '--max_length', 
        type=int, 
        default=8192,
        help="Maximum sequence length"
    )
    parser.add_argument(
        '--output_dir', 
        type=str,
        default='./results/embeddings',
        help="Directory to save embeddings"
    )
    parser.add_argument(
        '--chunk_size', 
        type=int, 
        default=5000,
        help="Number of samples per chunk"
    )
    parser.add_argument(
        '--test_mode', 
        action='store_true',
        help="Run in test mode with limited samples"
    )
    parser.add_argument(
        '--num_test_samples', 
        type=int, 
        default=1,
        help="Number of samples for test mode"
    )
    parser.add_argument(
        '--smart_batching', 
        action='store_true',
        help="Sort prompts by length for efficient batching"
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Load prompts
    print("--- Loading Prompts ---")
    try:
        if args.prompts_file.endswith('.csv'):
            df = pd.read_csv(args.prompts_file)
            all_prompts = df['Prompt'].tolist()
        elif args.prompts_file.endswith('.txt'):
            with open(args.prompts_file, 'r') as f:
                all_prompts = [line.strip() for line in f]
        else:
            raise ValueError("Unsupported file format.")
        
        print(f"Loaded {len(all_prompts)} total prompts.")
        
        if args.test_mode:
            print(f"TEST MODE: Using only {args.num_test_samples} samples")
            all_prompts.sort(key=lambda s: len(s.split()), reverse=True)
            all_prompts = all_prompts[:args.num_test_samples]
        
        print("-" * 40)
        
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return

    batch_size = args.batch_size
    print(f"Using batch_size={batch_size}, max_length={args.max_length}")
    print("-" * 40)

    # Process dataset
    start_time = time.time()
    
    if args.test_mode:
        if args.smart_batching:
            sorted_prompts, original_indices = smart_batching_by_length(all_prompts)
            embeddings_sorted = get_embeddings_batched(
                sorted_prompts, model, tokenizer, batch_size, args.max_length
            )
            embeddings = np.zeros_like(embeddings_sorted)
            for sorted_idx, original_idx in enumerate(original_indices):
                embeddings[original_idx] = embeddings_sorted[sorted_idx]
        else:
            embeddings = get_embeddings_batched(
                all_prompts, model, tokenizer, batch_size, args.max_length
            )
        
        test_output = os.path.join(args.output_dir, 'test_embeddings.npy')
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(test_output, embeddings)
        print(f"Test embeddings saved to: {test_output}")
    else:
        process_large_dataset_chunked(
            model, tokenizer, all_prompts,
            batch_size, args.max_length,
            args.output_dir, args.chunk_size,
            smart_batching=args.smart_batching
        )
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {total_time/3600:.2f} hours ({total_time:.2f} seconds)")
    print(f"Average speed: {len(all_prompts)/total_time:.2f} samples/second")
    print("=" * 60)


if __name__ == '__main__':
    main()
