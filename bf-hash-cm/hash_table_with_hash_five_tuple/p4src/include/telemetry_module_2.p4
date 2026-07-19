/* -*- P4_16 -*- */
// =========================================================================
// 模块 2：全量流频次统计 (V2.0: 哈希前置，复用模块1.1的 h1/h2/h3_raw)
// 3行CM Sketch，每行16384列，防溢出饱和机制，主动探针拦截提取
// =========================================================================
control Module_2_Sketch(
    inout headers_t hdr,
    inout my_ingress_metadata_t meta,
    in    ingress_intrinsic_metadata_t ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    // =========================================================================
    // V2.0 修改点：哈希前置，直接复用模块1.1计算的三个哈希结果
    // 截取低 14 位作为物理索引
    // =========================================================================
    action do_fetch_hash_idx() {
        meta.s1_idx = meta.h1_raw[13:0];
        meta.s2_idx = meta.h2_raw[13:0];
        meta.s3_idx = meta.h3_raw[13:0];
    }
    table t_fetch_hash_idx {
        actions = { do_fetch_hash_idx; }
        default_action = do_fetch_hash_idx();
        size = 1;
    }

    // =========================================================================
    // CM SKETCH REGISTERS & ACTIONS (保持原状，防溢出饱和机制)
    // =========================================================================
    Register<bit<16>, bit<14>>(16384, 0) sketch_1;
    RegisterAction<bit<16>, bit<14>, bit<16>>(sketch_1) update_s1 = {
        void apply(inout bit<16> reg_val, out bit<16> read_val) {
            read_val = reg_val;
            if (meta.is_probe == 0 && reg_val < 65535) { reg_val = reg_val + 1; }
        }
    };
    action do_update_s1() { meta.s1_out = update_s1.execute(meta.s1_idx); }
    table t_update_s1 { actions = { do_update_s1; } default_action = do_update_s1(); size = 1; }

    Register<bit<16>, bit<14>>(16384, 0) sketch_2;
    RegisterAction<bit<16>, bit<14>, bit<16>>(sketch_2) update_s2 = {
        void apply(inout bit<16> reg_val, out bit<16> read_val) {
            read_val = reg_val;
            if (meta.is_probe == 0 && reg_val < 65535) { reg_val = reg_val + 1; }
        }
    };
    action do_update_s2() { meta.s2_out = update_s2.execute(meta.s2_idx); }
    table t_update_s2 { actions = { do_update_s2; } default_action = do_update_s2(); size = 1; }

    Register<bit<16>, bit<14>>(16384, 0) sketch_3;
    RegisterAction<bit<16>, bit<14>, bit<16>>(sketch_3) update_s3 = {
        void apply(inout bit<16> reg_val, out bit<16> read_val) {
            read_val = reg_val;
            if (meta.is_probe == 0 && reg_val < 65535) { reg_val = reg_val + 1; }
        }
    };
    action do_update_s3() { meta.s3_out = update_s3.execute(meta.s3_idx); }
    table t_update_s3 { actions = { do_update_s3; } default_action = do_update_s3(); size = 1; }

    apply {
        if (hdr.ipv4.isValid()) {
            // --- 阶段 1：探针检查与参数覆盖 ---
            if (hdr.probe.isValid() && hdr.udp.dst_port == PROBE_PORT) {
                meta.is_probe = 1;
                // 覆盖哈希索引为探针提供的目标行号
                meta.s1_idx = hdr.probe.seq_id[13:0];
                meta.s2_idx = hdr.probe.seq_id[13:0];
                meta.s3_idx = hdr.probe.seq_id[13:0];
                hdr.udp.checksum = 0;
            } else {
                meta.is_probe = 0;
                // V2.0: 复用模块1.1的哈希结果，截取低14位
                t_fetch_hash_idx.apply();
            }

            // --- 阶段 2：强制表隔离调度流（防死锁） ---
            t_update_s1.apply();
            t_update_s2.apply();
            t_update_s3.apply();

            // --- 阶段 3：探针数据装载回填 ---
            if (meta.is_probe == 1) {
                hdr.probe.sketch_1_val = meta.s1_out;
                hdr.probe.sketch_2_val = meta.s2_out;
                hdr.probe.sketch_3_val = meta.s3_out;
            }
        }
    }
}
