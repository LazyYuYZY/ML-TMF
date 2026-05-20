import os
import time
import numpy as np
from sketchs import *   # 确保 SeqSketch, EmbedSketch, rw_files 可用

# ========== 辅助函数 ==========
def compute_error_metrics(real_sizes, est_sizes):
    real = np.array(real_sizes, dtype=np.float64)
    est = np.array(est_sizes, dtype=np.float64)
    abs_err = np.abs(real - est)
    rel_err = abs_err / (real + 1e-12)
    mean_rel_err = np.mean(rel_err)
    ratio_lt_0_1pct = np.sum(rel_err < 0.001) / len(rel_err)
    return mean_rel_err, ratio_lt_0_1pct

# ========== 测试 SeqSketch（使用压缩感知恢复）==========
def test_seqsketch_cs(memory_factors, base_config):
    flow_index_path = "./testdata_set_5s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flow_index_path)
    all_errors, all_ratios = [], []
    all_times = []

    for factor in memory_factors:
        fs_cols = int(base_config['base_fs_cols'] * factor)
        heavy_buckets = int(base_config['base_heavy_buckets'] * factor)
        bf_size = int(base_config['base_bf_size'] * factor)

        seq = SeqSketch(
            num_hash_buckets=heavy_buckets,
            bf_size=bf_size,
            bf_hash_num=base_config['bf_hash_num'],
            fs_rows=base_config['fs_rows'],
            fs_cols=fs_cols,
            flag=0
        )

        file_errors, file_ratios = [], []
        file_times = []

        for file_idx in range(10):
            json_path = f"./testdata_set_5s/testdata_flows_5s/{file_idx:05d}.txt"
            flows_data = rw_files.get_dict(json_path)
            flows_dict = {flows_alltime_dict[name]: int(cnt) for name, cnt in flows_data.items()}
            flow_ids = list(flows_dict.keys())
            real_vals = list(flows_dict.values())

            seq.insert_dict(flows_dict)

            start = time.time()
            recovered = seq.recover(candidate_flows=flow_ids)
            elapsed = time.time() - start
            file_times.append(elapsed)

            est_vals = [seq.query_recovered(fid) for fid in flow_ids]

            mean_rel_err, ratio = compute_error_metrics(real_vals, est_vals)
            file_errors.append(mean_rel_err)
            file_ratios.append(ratio)

            print(f"[SeqSketch-CS] factor={factor}, file={file_idx}: "
                  f"mean_rel_err={mean_rel_err:.6e}, ratio={ratio:.4f}, time={elapsed:.2f}s")

            seq.clear()

        all_errors.append(file_errors)
        all_ratios.append(file_ratios)
        all_times.append(file_times)

        result_dir = "./results/wide/seqsketch_cs/"
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "seq_memory_errors.txt"), 'w') as f:
            for errs in all_errors:
                f.write(" ".join(f"{e:.18e}" for e in errs) + "\n")
        with open(os.path.join(result_dir, "seq_memory_ratios.txt"), 'w') as f:
            for rats in all_ratios:
                f.write(" ".join(f"{r:.6f}" for r in rats) + "\n")
        with open(os.path.join(result_dir, "seq_recovery_times.txt"), 'w') as f:
            for times in all_times:
                f.write(" ".join(f"{t:.2f}" for t in times) + "\n")
        print(f"SeqSketch-CS results saved to {result_dir}")

    return all_errors, all_ratios, all_times

# ========== 测试 EmbedSketch（使用压缩感知恢复）==========
def test_embedsketch_cs(memory_factors, base_config):
    flow_index_path = "./testdata_set_5s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flow_index_path)
    all_errors, all_ratios = [], []
    all_times = []

    for factor in memory_factors:
        fs_cols = int(base_config['base_fs_cols'] * factor)
        bf_bits_per_bucket = int(base_config['base_bf_bits_per_bucket'] * factor)

        embed = EmbedSketch(
            fs_rows=base_config['fs_rows'],
            fs_cols=fs_cols,
            bf_bits_per_bucket=bf_bits_per_bucket,
            flag=0
        )

        file_errors, file_ratios = [], []
        file_times = []

        for file_idx in range(10):
            json_path = f"./testdata_set_5s/testdata_flows_5s/{file_idx:05d}.txt"
            flows_data = rw_files.get_dict(json_path)
            flows_dict = {flows_alltime_dict[name]: int(cnt) for name, cnt in flows_data.items()}
            flow_ids = list(flows_dict.keys())
            real_vals = list(flows_dict.values())

            embed.insert_dict(flows_dict)

            start = time.time()
            recovered = embed.recover(candidate_flows=flow_ids)
            elapsed = time.time() - start
            file_times.append(elapsed)

            est_vals = [embed.query_recovered(fid) for fid in flow_ids]

            mean_rel_err, ratio = compute_error_metrics(real_vals, est_vals)
            file_errors.append(mean_rel_err)
            file_ratios.append(ratio)

            print(f"[EmbedSketch-CS] factor={factor}, file={file_idx}: "
                  f"mean_rel_err={mean_rel_err:.6e}, ratio={ratio:.4f}, time={elapsed:.2f}s")

            embed.clear()

        all_errors.append(file_errors)
        all_ratios.append(file_ratios)
        all_times.append(file_times)

        result_dir = "./results/wide/embedsketch_cs/"
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "embed_memory_errors.txt"), 'w') as f:
            for errs in all_errors:
                f.write(" ".join(f"{e:.18e}" for e in errs) + "\n")
        with open(os.path.join(result_dir, "embed_memory_ratios.txt"), 'w') as f:
            for rats in all_ratios:
                f.write(" ".join(f"{r:.6f}" for r in rats) + "\n")
        with open(os.path.join(result_dir, "embed_recovery_times.txt"), 'w') as f:
            for times in all_times:
                f.write(" ".join(f"{t:.2f}" for t in times) + "\n")
        print(f"EmbedSketch-CS results saved to {result_dir}")

    return all_errors, all_ratios, all_times

# ========== 测试 SeqSketch 传统查询（无恢复）==========
def test_seqsketch_traditional(memory_factors, base_config):
    flow_index_path = "./testdata_set_5s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flow_index_path)
    all_errors, all_ratios = [], []

    for factor in memory_factors:
        fs_cols = int(base_config['base_fs_cols'] * factor)
        heavy_buckets = int(base_config['base_heavy_buckets'] * factor)
        bf_size = int(base_config['base_bf_size'] * factor)

        seq = SeqSketch(
            num_hash_buckets=heavy_buckets,
            bf_size=bf_size,
            bf_hash_num=base_config['bf_hash_num'],
            fs_rows=base_config['fs_rows'],
            fs_cols=fs_cols,
            flag=0
        )

        file_errors, file_ratios = [], []
        for file_idx in range(10):
            json_path = f"./testdata_set_5s/testdata_flows_5s/{file_idx:05d}.txt"
            flows_data = rw_files.get_dict(json_path)
            flows_dict = {flows_alltime_dict[name]: int(cnt) for name, cnt in flows_data.items()}
            flow_ids = list(flows_dict.keys())
            real_vals = list(flows_dict.values())

            seq.insert_dict(flows_dict)
            est_vals = [seq.query_one(fid) for fid in flow_ids]

            mean_rel_err, ratio = compute_error_metrics(real_vals, est_vals)
            file_errors.append(mean_rel_err)
            file_ratios.append(ratio)
            print(f"[SeqSketch-Trad] factor={factor}, file={file_idx}: "
                  f"mean_rel_err={mean_rel_err:.6e}, ratio={ratio:.4f}")
            seq.clear()

        all_errors.append(file_errors)
        all_ratios.append(file_ratios)

        result_dir = "./results/wide/seqsketch_trad/"
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "seq_memory_errors.txt"), 'w') as f:
            for errs in all_errors:
                f.write(" ".join(f"{e:.18e}" for e in errs) + "\n")
        with open(os.path.join(result_dir, "seq_memory_ratios.txt"), 'w') as f:
            for rats in all_ratios:
                f.write(" ".join(f"{r:.6f}" for r in rats) + "\n")
        print(f"SeqSketch-Trad results saved to {result_dir}")

    return all_errors, all_ratios

# ========== 测试 EmbedSketch 传统查询（无恢复）==========
def test_embedsketch_traditional(memory_factors, base_config):
    flow_index_path = "./testdata_set_5s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flow_index_path)
    all_errors, all_ratios = [], []

    for factor in memory_factors:
        fs_cols = int(base_config['base_fs_cols'] * factor)
        bf_bits_per_bucket = int(base_config['base_bf_bits_per_bucket'] * factor)

        embed = EmbedSketch(
            fs_rows=base_config['fs_rows'],
            fs_cols=fs_cols,
            bf_bits_per_bucket=bf_bits_per_bucket,
            flag=0
        )

        file_errors, file_ratios = [], []
        for file_idx in range(10):
            json_path = f"./testdata_set_5s/testdata_flows_5s/{file_idx:05d}.txt"
            flows_data = rw_files.get_dict(json_path)
            flows_dict = {flows_alltime_dict[name]: int(cnt) for name, cnt in flows_data.items()}
            flow_ids = list(flows_dict.keys())
            real_vals = list(flows_dict.values())

            embed.insert_dict(flows_dict)
            est_vals = [embed.query_one(fid) for fid in flow_ids]

            mean_rel_err, ratio = compute_error_metrics(real_vals, est_vals)
            file_errors.append(mean_rel_err)
            file_ratios.append(ratio)
            print(f"[EmbedSketch-Trad] factor={factor}, file={file_idx}: "
                  f"mean_rel_err={mean_rel_err:.6e}, ratio={ratio:.4f}")
            embed.clear()

        all_errors.append(file_errors)
        all_ratios.append(file_ratios)

        result_dir = "./results/wide/embedsketch_trad/"
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "embed_memory_errors.txt"), 'w') as f:
            for errs in all_errors:
                f.write(" ".join(f"{e:.18e}" for e in errs) + "\n")
        with open(os.path.join(result_dir, "embed_memory_ratios.txt"), 'w') as f:
            for rats in all_ratios:
                f.write(" ".join(f"{r:.6f}" for r in rats) + "\n")
        print(f"EmbedSketch-Trad results saved to {result_dir}")

    return all_errors, all_ratios

# ========== 主程序 ==========
if __name__ == '__main__':
    memory_factors = [8, 12, 16, 24, 32]   # 内存倍数

    # SeqSketch 基础配置（factor=1 时总内存约 0.124 MB）
    base_seq_config = {
        'fs_rows': 2,
        'base_fs_cols': 5000,
        'base_heavy_buckets': 2500,
        'base_bf_size': 80000,      # bits
        'bf_hash_num': 7
    }

    # EmbedSketch 基础配置（factor=1 时总内存约 0.119 MB）
    base_embed_config = {
        'fs_rows': 2,
        'base_fs_cols': 2500,
        'base_bf_bits_per_bucket': 8
    }

    # print("=== Testing SeqSketch with Compressive Sensing Recovery ===")
    # test_seqsketch_cs(memory_factors, base_seq_config)
    #
    # print("\n=== Testing EmbedSketch with Compressive Sensing Recovery ===")
    # test_embedsketch_cs(memory_factors, base_embed_config)

    # 可选：运行传统查询以作对比（取消注释）
    print("\n=== Testing SeqSketch Traditional Query (No Recovery) ===")
    test_seqsketch_traditional(memory_factors, base_seq_config)

    print("\n=== Testing EmbedSketch Traditional Query (No Recovery) ===")
    test_embedsketch_traditional(memory_factors, base_embed_config)

    print("All tests completed.")