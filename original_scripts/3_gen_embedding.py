import torch
from transformers import GemmaTokenizer, AutoModel
import os
import time
import numpy as np
import pandas as pd
import argparse

from transformers import BitsAndBytesConfig

def load_model_and_tokenizer(model_path):
    print("--- Loading Model and Tokenizer (Optimized for Large Batch Processing) ---")
    start_time = time.time()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Multi-GPU inference requires GPUs.")
    
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} GPUs.")

    print(f"Explicitly loading SLOW GemmaTokenizer to bypass conversion issues...")
    
    tokenizer_file = os.path.join(model_path, "tokenizer.model")
    if not os.path.exists(tokenizer_file):
        raise FileNotFoundError(f"tokenizer.model not found at {tokenizer_file}.")

    tokenizer = GemmaTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Warning: pad_token was not set. Manually set to eos_token.")

    print("Tokenizer loaded successfully as a slow tokenizer.")
    
    print(f"Loading model from local path: {model_path}")
    print("This may take a few minutes...")
    
    # 优化的量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # 使用 float16 而不是 bfloat16
        bnb_4bit_use_double_quant=True,  # 双重量化节省更多内存
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
    """
    针对长序列优化的嵌入提取
    """
    all_embeddings = []
    total_batches = (len(prompts) - 1) // batch_size + 1
    
    # 设置进度更新间隔
    log_interval = max(1, total_batches // 100)  # 每处理1%打印一次
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        current_batch_num = i // batch_size + 1
        
        if current_batch_num % log_interval == 0 or current_batch_num == 1:
            progress = (current_batch_num / total_batches) * 100
            print(f"Progress: {progress:.1f}% - Batch {current_batch_num}/{total_batches}")

        # Tokenize
        tokens = tokenizer(
            batch_prompts, 
            padding=True,  # 使用动态padding而不是填充到max_length
            truncation=True, 
            max_length=max_length,
            return_tensors="pt"
        )
        
        # 只将必要的张量移到GPU
        input_ids = tokens['input_ids'].to(model.device)
        attention_mask = tokens['attention_mask'].to(model.device)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

            last_hidden_states = outputs.hidden_states[-1]
            
            # Masked mean pooling - 在GPU上完成
            expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
            sum_embeddings = (last_hidden_states * expanded_mask).sum(1)
            sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).cpu().half()  # 使用half精度保存
            # 立即转换并添加到列表
            all_embeddings.append(batch_embeddings.numpy())
            
            print(batch_embeddings)
            # 积极的内存清理
            del outputs, last_hidden_states, input_ids, attention_mask, tokens
            del expanded_mask, sum_embeddings, sum_mask, batch_embeddings
            
        # 每10个batch清理一次GPU缓存
        if current_batch_num % 10 == 0:
            torch.cuda.empty_cache()
    
    # 最后拼接所有嵌入
    return np.vstack(all_embeddings).astype(np.float32)

def smart_batching_by_length(prompts):
    """
    按照prompt字符串长度排序，让相似长度的在同一个batch
    避免重复tokenize，直接用字符串长度作为近似
    """
    print("--- Sorting prompts by length for efficient batching ---")
    
    # 计算每个prompt的字符串长度（快速近似）
    prompt_lengths = []
    for i, prompt in enumerate(prompts):
        if i % 10000 == 0:
            print(f"Sorting progress: {i}/{len(prompts)}")
        
        # 使用字符串长度作为近似（比tokenize快得多）
        length = len(prompt)
        prompt_lengths.append((prompt, length, i))  # 保存原始索引
    
    # 按长度排序（从短到长）
    prompt_lengths.sort(key=lambda x: x[1])
    
    sorted_prompts = [p[0] for p in prompt_lengths]
    lengths = [p[1] for p in prompt_lengths]
    original_indices = [p[2] for p in prompt_lengths]  # 保存原始索引
    
    print(f"\nSorting completed:")
    print(f"  Shortest: {lengths[0]} characters")
    print(f"  Longest: {lengths[-1]} characters")
    print(f"  Median: {np.median(lengths):.0f} characters")
    
    return sorted_prompts, original_indices

def process_large_dataset_chunked(model, tokenizer, prompts, batch_size, max_length, output_dir, chunk_size=5000, smart_batching=False):
    """
    分块处理数据，定期保存避免内存问题
    """
    os.makedirs(output_dir, exist_ok=True)
    
    total_chunks = (len(prompts) - 1) // chunk_size + 1
    print(f"Total prompts: {len(prompts)}")
    print(f"Processing in {total_chunks} chunks of {chunk_size} samples each")
    print(f"Smart batching: {'ENABLED' if smart_batching else 'DISABLED'}")
    print("=" * 60)
    
    # 如果启用智能batching，先对所有prompts排序
    if smart_batching:
        print("\n--- Starting smart batching preprocessing ---")
        sorted_prompts, original_indices = smart_batching_by_length(prompts)
        print("Smart batching preprocessing completed.")
        print("=" * 60)
    else:
        sorted_prompts = prompts
        original_indices = list(range(len(prompts)))
    
    # 存储所有embeddings以便后续恢复顺序
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
        
        # 保存分块结果（排序后的）
        chunk_file = os.path.join(output_dir, f"embeddings_chunk_sorted_{chunk_idx:06d}_{chunk_end:06d}.npy")
        np.save(chunk_file, embeddings)
        
        # 保存到内存以便后续恢复顺序
        all_embeddings_sorted.append(embeddings)
        
        print(f"\n✓ Chunk {chunk_num} completed:")
        print(f"  - Time: {chunk_time:.2f}s")
        print(f"  - Speed: {samples_per_sec:.2f} samples/sec")
        print(f"  - Saved to: {chunk_file}")
        print(f"  - Shape: {embeddings.shape}")
        
        # 估算剩余时间
        remaining_samples = len(sorted_prompts) - chunk_end
        estimated_time_remaining = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0
        print(f"  - Estimated time remaining: {estimated_time_remaining/3600:.2f} hours")
        
        del embeddings
        torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print("All chunks processed!")
    print(f"{'='*60}")
    
    # 合并所有embeddings
    print("\nMerging all chunks...")
    merged_embeddings_sorted = np.vstack(all_embeddings_sorted)
    
    # 如果使用了智能batching，恢复原始顺序
    if smart_batching:
        print("Restoring original order...")
        merged_embeddings = np.zeros_like(merged_embeddings_sorted)
        for sorted_idx, original_idx in enumerate(original_indices):
            merged_embeddings[original_idx] = merged_embeddings_sorted[sorted_idx]
        
        # 保存恢复顺序后的文件
        merged_file = os.path.join(output_dir, "all_embeddings_merged.npy")
        np.save(merged_file, merged_embeddings)
        
        print(f"\n✓ Merged embeddings (original order) saved to: {merged_file}")
        print(f"  Final shape: {merged_embeddings.shape}")
    else:
        # 保存排序后的文件
        merged_file = os.path.join(output_dir, "all_embeddings_merged.npy")
        np.save(merged_file, merged_embeddings_sorted)
        
        print(f"\n✓ Merged embeddings saved to: {merged_file}")
        print(f"  Final shape: {merged_embeddings_sorted.shape}")
    
    print(f"  Expected samples: {len(prompts)}")
    
    return merged_file


def merge_embedding_chunks(output_dir, total_samples):
    """
    合并所有分块的嵌入文件
    """
    chunk_files = sorted([f for f in os.listdir(output_dir) if f.startswith('embeddings_chunk_')])
    
    if not chunk_files:
        print("No chunk files found!")
        return
    
    print(f"Found {len(chunk_files)} chunk files. Merging...")
    
    all_embeddings = []
    for chunk_file in chunk_files:
        chunk_path = os.path.join(output_dir, chunk_file)
        embeddings = np.load(chunk_path)
        all_embeddings.append(embeddings)
        print(f"Loaded {chunk_file}: shape {embeddings.shape}")
    
    # 合并
    merged_embeddings = np.vstack(all_embeddings)
    
    # 保存合并后的文件
    merged_file = os.path.join(output_dir, "all_embeddings_merged.npy")
    np.save(merged_file, merged_embeddings)
    
    print(f"\n✓ Merged embeddings saved to: {merged_file}")
    print(f"  Final shape: {merged_embeddings.shape}")
    print(f"  Expected samples: {total_samples}")
    
    assert merged_embeddings.shape[0] == total_samples, "Sample count mismatch!"
    
    return merged_file

def main():
    parser = argparse.ArgumentParser(description="Process large-scale embeddings with long sequences.")
    
    parser.add_argument('--model_path', type=str, 
                       default="/home/hechang/model/c2s-scale-gemma-2")
    parser.add_argument('--prompts_file', type=str, 
                       default="/home/hechang/merged_frame/results/gdsc_ccle_mechanism_prompts_filtered.csv")
    parser.add_argument('--batch_size', type=int, default=1,
                       help="Batch size")
    parser.add_argument('--max_length', type=int, default=8192,
                       help="Maximum sequence length")
    parser.add_argument('--output_dir', type=str, 
                       default='/home/hechang/merged_frame/results/embeddings_chunks',
                       help="Directory to save chunk files")
    parser.add_argument('--chunk_size', type=int, default=5000,
                       help="Number of samples per chunk")
    parser.add_argument('--test_mode', action='store_true',
                       help="Run in test mode with limited samples")
    parser.add_argument('--num_test_samples', type=int, default=1,
                       help="Number of samples for test mode")
    parser.add_argument('--smart_batching', action='store_true',
                       help="Sort prompts by length for efficient batching (RECOMMENDED)")
    
    args = parser.parse_args()
    
    # --- 1. Load Model ---
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # --- 2. Load Prompts ---
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

    # --- 4. Process Dataset ---
    start_time = time.time()
    
    if args.test_mode:
        # 测试模式：直接处理不分块
        if args.smart_batching:
            sorted_prompts, original_indices = smart_batching_by_length(all_prompts)
            embeddings_sorted = get_embeddings_batched(
                sorted_prompts, model, tokenizer, batch_size, args.max_length
            )
            # 恢复原始顺序
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
        # 生产模式：分块处理
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