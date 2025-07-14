import configparser
import subprocess
import re
import argparse
import os
import json
import torch

def load_model_config(config_file="model_configs.json"):
    # Load model configuration file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_file)
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file {config_path} does not exist!")
        return {}
    except json.JSONDecodeError:
        print(f"Configuration file {config_path} format error!")
        return {}

def get_model_config(model, cache_size):
    # Get model configuration based on model name and cache size
    # Load model configurations
    model_configs = load_model_config()
    
    # Determine adapter_path
    if model in ["switch-base-128", "switch-large-128"]:
        adapter_path = ""
    else:
        adapter_path = f"/data/{model}"
    
    # Determine base_model
    if "base" in model:
        base_model = "t5-base"
    elif "large" in model:
        base_model = "t5-large"
    else:
        base_model = "t5-base"  # default value
    
    # Get cache_init_experts
    cache_init_experts = []
    if model in model_configs:
        cache_init_experts = model_configs[model].get("cache_init_experts", [])
    else:
        print(f"Warning: Configuration for model {model} not found, using default values")
    
    # Process experts directly based on cache_size
    if cache_size == 0 or not cache_init_experts:
        top_k_experts = []
    else:
        top_k_experts = []
        for expert_row in cache_init_experts:
            if cache_size <= len(expert_row):
                # Take first cache_size experts
                top_k_experts.append(expert_row[:cache_size])
            else:
                # Pad with zeros to cache_size
                top_k_experts.append(expert_row + [0] * (cache_size - len(expert_row)))
    
    return {
        "adapter_path": adapter_path,
        "base_model": base_model,
        "cache_init_experts": top_k_experts
    }

def parse_output(output: str):
    block_lat = 0
    for line in reversed(output.splitlines()):
        m = re.search(r"BLK AVG: ([\d\.]+) ms", line)
        if m:
            block_lat = float(m[1])
            break

    throughput = .0
    for line in reversed(output.splitlines()):
        m = re.search(r", (\d+) tokens/sec\.", line)
        if m:
            throughput = int(m[1])
            break

    peak_mem_encoder = peak_mem_decoder = 0
    for line in output.splitlines():
        m = re.search(r"MEM usage: (\d+) (\d+)", line)
        if m:
            peak_mem_encoder = int(m[1])
            peak_mem_decoder = int(m[2])
            break

    max_active_experts = 0
    for line in output.splitlines():
        m = re.search(r"Max active experts: (\d+)", line)
        if m:
            max_active_experts = int(m[1])
            break

    cache_hit_rate = 0
    for line in output.splitlines():
        m = re.search(r"Average cache hit rate: ([\d\.]+)", line)
        if m:
            cache_hit_rate = float(m[1])

    return block_lat, throughput, peak_mem_encoder, peak_mem_decoder, max_active_experts, cache_hit_rate


def profile_config(cpp_config, model, method, batch_size, cache_ratio=0, cache_policy="LFU", dataset_name="", use_moecache=True, cache_size=6, fix_cache_size=0, max_val=2, threshold=1.0, dec_in_cache=0.4, dec_out_cache=0.4, print_log=False):
    iterations = 4
    os.makedirs(f"logs/{model}", exist_ok=True)
    exp_name = f"{model}/{model}_{method}_{batch_size}_{cache_ratio}_{cache_policy}_{dataset_name.rsplit('/', 1)[-1]}_{use_moecache}_{cache_size}_{fix_cache_size}_{max_val}_{threshold}_{dec_in_cache}_{dec_out_cache}"
    print(f"Running {exp_name}")
    if method == "GPU-only":
        encoder_fetcher_mode = "0"
        decoder_fetcher_mode = "0"
    elif method == "Pre-gated":
        encoder_fetcher_mode = "1"
        decoder_fetcher_mode = "2"
    elif method == "DeepSpeed":
        encoder_fetcher_mode = "1"
        decoder_fetcher_mode = "1"
    elif method == "Diff-MoE":
        encoder_fetcher_mode = "1"
        decoder_fetcher_mode = "2"
    # elif method == "SE-MoE":
    #     encoder_fetcher_mode = "1"
    #     decoder_fetcher_mode = "2"
    #     iterations = 1

    if "base" in model:
        size_per_expert = 18874368
        num_layer = 6
    elif "large" in model:
        size_per_expert = 33554432
        num_layer = 12
    # elif "xxl" in model:
    #     size_per_expert = 33554432
    #     num_layer = 12
    total_experts = int(re.search(r"\d+", model)[0])

    arena_size = 3 * size_per_expert * total_experts


    # Use new model configuration loading method
    config = get_model_config(model, cache_size)
    adapter_path = config["adapter_path"]
    base_model = config["base_model"]
    cache_init_experts = config["cache_init_experts"]
    
    # Use the processed experts directly
    top_k_experts = cache_init_experts
    
    # Convert 2D array to string format 
    if top_k_experts:
        top_k_experts_str = ','.join(
            ','.join(str(expert) for expert in row) for row in top_k_experts)
    else:
        # Default to all zeros when no configuration is found
        if cache_size > 0:
            # Create default all-zero configuration based on cache_size and num_layer
            if "base" in model:
                num_layer = 12
            elif "large" in model:
                num_layer = 24
            else:
                num_layer = 12  # default
            
            # Create num_layer/2 rows (decoder layers), each with cache_size zeros
            default_experts = [[0] * cache_size for _ in range(num_layer // 2)]
            top_k_experts_str = ','.join(
                ','.join(str(expert) for expert in row) for row in default_experts)
            # Update top_k_experts to use the generated default configuration
            top_k_experts = default_experts
        else:
            top_k_experts_str = ""
            top_k_experts = []

    # experts_list = [int(x) for x in top_k_experts_str.split(',') if x.strip()]

    cpp_config["default"] = {
        "arena_size": f"{arena_size}",
        "encoder_fetcher_mode": encoder_fetcher_mode,
        "decoder_fetcher_mode": decoder_fetcher_mode,
        "profiling": "1",
        "detailed_timing": "0",
        "offload_path": f"/data/ft/{model}/",
        "load_from_cpp": "1",
        "quant_mode": "0",
        "vocab_size": "32128",
        "cache_size": cache_size,
        "use_moe_cache": use_moecache,
        "top_k_experts": top_k_experts,
        "fix_cache_size": fix_cache_size,
        "max_val": max_val,
        "threshold": threshold,
        "dec_in_cache": dec_in_cache,
        "dec_out_cache": dec_out_cache,
    }

    with open("/workspace/FasterTransformer/cpp_config.ini", "w") as fp:
        cpp_config.write(fp)
    # Build command with conditional adapter_path
    adapter_arg = "" if adapter_path == "" else f"--adapter_path {adapter_path} "
    command = (
        f"python /workspace/FasterTransformer/examples/pytorch/t5/perf_benchmark.py "
        f"--batch_size {batch_size} "
        f"--beam_width 4 "
        f"--seq_len 256 "
        f"--data_type fp32 "
        f"--test_time 3 "
        f"--sampling_topk 1 "
        f"--model_type Megatron-DeepSpeed "
        f"--ckpt_path /data/ft/{model}/ "
        f"--model {base_model} "
        f"--duration 0 "
        f"--iterations {iterations} "
        f"--dataset_name {dataset_name} "
        f"--use_moe_cache {use_moecache} "
        f"{adapter_arg}"
        f"--cache_size {cache_size} "
        f"--fix_cache_size {fix_cache_size} "
        f"--top_k_experts {top_k_experts_str} "
        f"--layer_num {num_layer} "
    )

    # f"--test_time 3 "
    print(command)
    if print_log:
        result = subprocess.run(
            command,
            shell=True,
            stdout=None,
            stderr=None,
            text=True,
            cwd="/workspace/FasterTransformer/build"
        )
    else:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd="/workspace/FasterTransformer/build"
        )

    with open(f"/workspace/FasterTransformer/logs/{exp_name}.log", "w") as fp:
        fp.write(result.stdout)

    block_lat, throughput, peak_mem_encoder, peak_mem_decoder, max_active_experts, cache_hit_rate = parse_output(
        result.stdout)

    if method == "Pre-gated":
        used_buffer = 2 * max_active_experts
    elif method == "DeepSpeed":
        used_buffer = max_active_experts
    elif method == "GPU-only":
        used_buffer = num_layer * total_experts
    # elif method == "SE-MoE":
    #     used_buffer = 2 * total_experts

    peak_mem = peak_mem_decoder - arena_size - \
        size_per_expert * (2 * total_experts - used_buffer)
    if cache_ratio != 0:
        peak_mem = peak_mem + arena_size
    print(
        f"BLK AVG: {block_lat} ms, "
        f"throughput: {throughput} tokens/sec, "
        f"peak_mem_encoder: {peak_mem_encoder}, "
        f"peak_mem_decoder: {peak_mem_decoder}, "
        f"max_active_experts: {max_active_experts}, "
        f"peak_mem: {peak_mem}, "
        f"cache_hit_rate: {cache_hit_rate}"
    )

    return {
        "block_lat": block_lat,
        "throughput": throughput,
        "peak_mem": peak_mem,
        "max_active_expert": max_active_experts,
        "cache_hit_rate": cache_hit_rate,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print_log", "-p", type=bool, default=False)

    return parser.parse_args()


def main():
    args = parse_args()
    
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} CUDA device(s)")

    model_dataset_pairs = [
        # for finetuned model
        # ("switch-base-128-xsum", "EdinburghNLP/xsum"),
        # ("switch-base-128-squad", "squad"),
        # ("switch-base-128-coqa", "stanfordnlp/coqa"),
        # ("switch-large-128-xsum", "EdinburghNLP/xsum"),
        # ("switch-large-128-squad", "squad"),
        # ("switch-large-128-coqa", "stanfordnlp/coqa"),
        
        # # for base model
        ("switch-base-128", "EdinburghNLP/xsum"),
        # ("switch-base-128", "squad"),
        # ("switch-base-128", "stanfordnlp/coqa"),
        # ("switch-large-128", "EdinburghNLP/xsum"),
        # ("switch-large-128", "squad"),
        # ("switch-large-128", "stanfordnlp/coqa"),
        
    ]
    batch_sizes = [
        128,
        64,
        32,
        16,
        8,
        4,
        2,
        1,
    ]
    methods = [
        "Diff-MoE",
        # "Pre-gated",
        # "GPU-only",
        # "DeepSpeed",
        # "SE-MoE",
    ]

    cache_size_fix_cache_size_pairs = [ # (cache_size, fix_cache_size) i.e., (HPC+MPC, HPC)
        (3, 1),
        (6, 2),
        (13, 4),
        # (3, 0),
        # (6, 0),
        # (13, 0),
    ]

    thresholds = [ # (threshold, dec_out_cache)
        (2.0, 1.0),
        # (3.0, 1.5),
        # (3.0, 1.0),
    ]

    decs = [ # (dec_in_cache, dec_out_cache)
        (0.4, 0.2), #best
        # (0.4, 0.4),
        # (0.5, 0.5),
        # (0.6, 0.4), 
        # (0.4, 0.6),
        # (0.8, 0.8),
        # (0.2, 0.2),
        # (0.4, 0.8),
        # (0.8, 0.4),
        # (0.2, 0.4),
    ]
    cpp_config = configparser.ConfigParser()
    cpp_config.read("/workspace/FasterTransformer/cpp_config.ini")

    # Run performance tests
    for method in methods:
        for (model, dataset_name) in model_dataset_pairs:
            for batch_size in batch_sizes:
                use_moecache = True if method == "Diff-MoE" else False
                cache_size_fix_cache_size_pairs_list, thresholds_list, decs_list = (cache_size_fix_cache_size_pairs, thresholds, decs) if use_moecache else ([(0, 0)], [(0, 0)], [(0, 0)])
                for cache_size, fix_cache_size in cache_size_fix_cache_size_pairs_list:
                    for (max_val, threshold) in thresholds_list:
                        for (dec_in_cache, dec_out_cache) in decs_list:
                            result = profile_config(
                                cpp_config,
                                model,
                                method,
                                batch_size,
                                cache_ratio=0,
                                cache_policy="LFU",
                                dataset_name=dataset_name,
                                use_moecache=use_moecache,
                                cache_size=cache_size,
                                fix_cache_size=fix_cache_size,
                                max_val=max_val,
                                threshold=threshold,
                                dec_in_cache=dec_in_cache,
                                dec_out_cache=dec_out_cache,
                                print_log=args.print_log,
                            )
                            
                            # Print detailed performance metrics
                            print(f"\n=== Performance Results ===")
                            print(f"Configuration:")
                            print(f"  - Model: {model}")
                            print(f"  - Method: {method}")
                            print(f"  - Batch Size: {batch_size}")
                            print(f"  - Cache Size: {cache_size}")
                            print(f"  - Fix Cache Size: {fix_cache_size}")
                            print(f"  - Dataset: {dataset_name}")
                            print(f"  - Use MoE Cache: {use_moecache}")
                            print(f"  - Threshold: {threshold}")
                            print(f"  - Dec In/Out Cache: {dec_in_cache}/{dec_out_cache}")
                            print(f"")
                            print(f"Metrics:")
                            print(f"  - Block Latency: {result.get('block_lat', 'N/A')} ms")
                            print(f"  - Throughput: {result.get('throughput', 'N/A')} tokens/sec")
                            print(f"  - Peak Memory: {result.get('peak_mem', 'N/A')} MB")
                            print(f"  - Max Active Experts: {result.get('max_active_expert', 'N/A')}")
                            print(f"  - Cache Hit Rate: {result.get('cache_hit_rate', 'N/A')}")
                            print("=" * 50)


if __name__ == "__main__":
    main()
