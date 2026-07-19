/* -*- P4_16 -*- */
// =========================================================================
// 模块 3：高精度训练数据直接收集表 (V2.0: 32位完整哈希Key, 复用模块1.1的h1_raw)
// 1024行精确哈希表，跨Stage状态机切分破局死锁
// =========================================================================
control Module_3_TrainData(
    inout headers_t hdr,
    inout my_ingress_metadata_t meta,
    in    ingress_intrinsic_metadata_t ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    // =========================================================================
    // V2.0 修改点：复用模块1.1的首个哈希结果 h1_raw 作为 flow_key (32位)
    // =========================================================================
    action do_fetch_flow_key() {
        meta.flow_key = meta.h1_raw;
        // 截取低 10 位作为 1024 行表的物理寻址索引
        meta.train_idx = meta.h1_raw[9:0];
    }
    table t_fetch_flow_key {
        actions = { do_fetch_flow_key; }
        default_action = do_fetch_flow_key();
        size = 1;
    }

    // =========================================================================
    // TRAINING DATA COLLECTION (核心拆解：解决级联死锁)
    // =========================================================================

    // --- 拆分 1：Key 表项处理 (V2.0: 32位完整哈希) ---
    Register<bit<32>, bit<10>>(1024, 0) reg_key;
    RegisterAction<bit<32>, bit<10>, bit<32>>(reg_key) process_key = {
        void apply(inout bit<32> reg_val, out bit<32> out_key) {
            out_key = reg_val; // 读出完整的 32 位旧值
            // 如果是首包（位置为空），则抢占坑位写入新 Key
            if (meta.is_probe == 0 && reg_val == 0) {
                reg_val = meta.flow_key; // 写入完整的 32 位 Key
            }
        }
    };
    action do_process_key() { meta.train_key_out = process_key.execute(meta.train_idx); }
    table t_update_train_key { actions = { do_process_key; } default_action = do_process_key(); size = 1; }

    // --- 拆分 2：Size 表项处理 ---
    Register<bit<32>, bit<10>>(1024, 0) reg_size;
    RegisterAction<bit<32>, bit<10>, bit<32>>(reg_size) process_size = {
        void apply(inout bit<32> reg_val, out bit<32> out_size) {
            out_size = reg_val;
            // 只有当外部判断匹配时，才允许计数累加
            if (meta.is_probe == 0 && meta.train_key_match == 1) { reg_val = reg_val + 1; }
        }
    };
    action do_process_size() { meta.train_size_out = process_size.execute(meta.train_idx); }
    table t_update_train_size { actions = { do_process_size; } default_action = do_process_size(); size = 1; }

    apply {
        if (hdr.ipv4.isValid()) {

            // --- 阶段 1：解析参数 ---
            if (meta.is_probe == 1) {
                // 截取探针 seq_id 的低 10 位作为拉取索引
                meta.train_idx = hdr.probe.seq_id[9:0];
            } else {
                // V2.0: 复用模块1.1的哈希结果
                t_fetch_flow_key.apply();
            }

            // --- 阶段 2：强制表级联隔离（Tofino 核心命脉） ---

            // 动作 A：读 Key
            t_update_train_key.apply();

            // 全局控制流：隔离判断 (V2.0: 32位完整哈希比对)
            if (meta.is_probe == 0) {
                if (meta.train_key_out == meta.flow_key || meta.train_key_out == 0) {
                    meta.train_key_match = 1;
                } else {
                    meta.train_key_match = 0; // 老鼠流冲突时，直接放弃计数
                }
            }

            // 动作 B：基于前置状态，独立执行加法
            t_update_train_size.apply();

            // --- 阶段 3：探针回填 ---
            if (meta.is_probe == 1) {
                hdr.probe.train_key = meta.train_key_out;
                hdr.probe.train_size = meta.train_size_out;
            }
        }
    }
}
