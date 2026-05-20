import os
import numpy as np
from sketchs import *   # 包含 fcm_sketch, rw_files

# ---------------------- 通用内存计算函数 ----------------------
def fcm_memory_usage(num_trees, num_stages, branch_factor, bits_per_stage, base_w):
    stage_widths = []
    w = base_w
    for s in range(num_stages):
        stage_widths.append(w)
        w = w // branch_factor
        if w < 1:
            w = 1

    stage_bytes_per_tree = []
    for s in range(num_stages):
        bits = bits_per_stage[s] * stage_widths[s]
        bytes_ = bits / 8.0
        stage_bytes_per_tree.append(bytes_)

    tree_bytes = sum(stage_bytes_per_tree)
    total_bytes = tree_bytes * num_trees
    return total_bytes, stage_widths, stage_bytes_per_tree

# ---------------------- 主测试 ----------------------
if __name__ == '__main__':
    # ---------- FCM 固定参数 ----------
    NUM_TREES = 3
    NUM_STAGES = 3
    BRANCH_FACTOR = 8
    BITS_PER_STAGE = (8, 16, 32)

    # 基础宽度因子
    w = 10 ** 4

    # 目标：mem_factor = 8 时总内存约为 0.96 MB
    TARGET_MEM_AT_8 = 0.96 * 1024 * 1024   # 字节

    # 计算缩放系数
    tmp_base_w = w * 8
    tmp_total, _, _ = fcm_memory_usage(NUM_TREES, NUM_STAGES, BRANCH_FACTOR, BITS_PER_STAGE, tmp_base_w)
    SCALE = TARGET_MEM_AT_8 / tmp_total
    print(f"计算得到的缩放系数 SCALE = {SCALE:.6f}")

    # 内存因子列表
    memory_list = [8, 12, 16, 24, 32]

    # 读取流ID映射
    flow_index_path = "./testdata_set_5s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flow_index_path)

    all_results = []

    for mem_factor in memory_list:
        # 计算 base_w
        base_w = int(w * mem_factor * SCALE)

        # 计算并打印内存占用详情
        total_bytes, widths, bytes_per_tree = fcm_memory_usage(
            NUM_TREES, NUM_STAGES, BRANCH_FACTOR, BITS_PER_STAGE, base_w
        )
        print("\n" + "=" * 60)
        print(f"内存因子 = {mem_factor},  base_w = {base_w}")
        print(f"各层宽度: {widths}")
        print(f"单棵树各层内存 (KB): {[b/1024 for b in bytes_per_tree]}")
        print(f"单棵树总内存: {sum(bytes_per_tree)/1024:.3f} KB")
        print(f"总内存 (MB): {total_bytes / (1024**2):.4f} MB")
        print("=" * 60)

        # 创建 FCM 实例
        fcm = fcm_sketch(
            num_trees=NUM_TREES,
            num_stages=NUM_STAGES,
            branch_factor=BRANCH_FACTOR,
            bits_per_stage=BITS_PER_STAGE,
            base_w=base_w,
            flag=0
        )

        file_errors = []
        for file_idx in range(10):
            json_file_path = f"./testdata_set_5s/testdata_flows_5s/{file_idx:05d}.txt"
            flows_data = rw_files.get_dict(json_file_path)

            flows_data_onetime = {}
            for flow_name, size in flows_data.items():
                flow_id = flows_alltime_dict[flow_name]
                flows_data_onetime[flow_id] = int(size)

            fcm.insert_dict(flows_data_onetime)

            flow_ids = list(flows_data_onetime.keys())
            real_sizes = np.array(list(flows_data_onetime.values()))
            est_sizes = np.array([fcm.query_one(fid) for fid in flow_ids])

            absolute_error = np.abs(real_sizes - est_sizes)
            relative_error = absolute_error / (real_sizes + 1e-12)
            mean_rel_err = np.mean(relative_error)
            file_errors.append(mean_rel_err)

            print(f"  文件 {file_idx}: 平均相对误差 = {mean_rel_err:.6f}")

            fcm.clear()

        all_results.append(file_errors)

    # 保存结果
    result_dir = "./results/wide/fcm/"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, "fcm_memory.txt")
    with open(result_file, 'w') as f:
        for errors in all_results:
            line = " ".join(f"{err:.18e}" for err in errors)
            f.write(line + "\n")
    print(f"\n所有结果已保存至 {result_file}")