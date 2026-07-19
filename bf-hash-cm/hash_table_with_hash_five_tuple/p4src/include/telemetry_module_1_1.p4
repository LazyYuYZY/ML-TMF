/* -*- P4_16 -*- */
// =========================================================================
// 模块 1.1：新流检测 (V2.0: 3个哈希 + 3个布隆过滤器寄存器 + 查表判新流)
// 计算出的 h1_raw / h2_raw / h3_raw 通过总线传递给模块2和模块3复用
// =========================================================================
control Module_1_1_BloomFilter(
    inout headers_t hdr,
    inout my_ingress_metadata_t meta,
    in    ingress_intrinsic_metadata_t ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    // --- 三个独立的 CRC32 哈希算法 ---
    CRCPolynomial<bit<32>>(coeff=0x04C11DB7, reversed=true, msb=false, extended=false, init=0xFFFFFFFF, xor=0xFFFFFFFF) poly1;
    CRCPolynomial<bit<32>>(coeff=0x741B8CD7, reversed=true, msb=false, extended=false, init=0xFFFFFFFF, xor=0xFFFFFFFF) poly2;
    CRCPolynomial<bit<32>>(coeff=0xDB710641, reversed=true, msb=false, extended=false, init=0xFFFFFFFF, xor=0xFFFFFFFF) poly3;

    Hash<bit<32>>(HashAlgorithm_t.CUSTOM, poly1) hash_algo1;
    Hash<bit<32>>(HashAlgorithm_t.CUSTOM, poly2) hash_algo2;
    Hash<bit<32>>(HashAlgorithm_t.CUSTOM, poly3) hash_algo3;

    // --- 哈希 1 ---
    action do_compute_hash_1() {
        meta.h1_raw = hash_algo1.get({hdr.ipv4.src_addr, hdr.ipv4.dst_addr, hdr.ipv4.protocol, hdr.udp.src_port, hdr.udp.dst_port});
        meta.bf_idx_1 = (bit<16>)meta.h1_raw;
    }
    table t_compute_hash_1 { actions = { do_compute_hash_1; } default_action = do_compute_hash_1(); size = 1; }

    // --- 哈希 2 ---
    action do_compute_hash_2() {
        meta.h2_raw = hash_algo2.get({hdr.ipv4.src_addr, hdr.ipv4.dst_addr, hdr.ipv4.protocol, hdr.udp.src_port, hdr.udp.dst_port});
        meta.bf_idx_2 = (bit<16>)meta.h2_raw;
    }
    table t_compute_hash_2 { actions = { do_compute_hash_2; } default_action = do_compute_hash_2(); size = 1; }

    // --- 哈希 3 ---
    action do_compute_hash_3() {
        meta.h3_raw = hash_algo3.get({hdr.ipv4.src_addr, hdr.ipv4.dst_addr, hdr.ipv4.protocol, hdr.udp.src_port, hdr.udp.dst_port});
        meta.bf_idx_3 = (bit<16>)meta.h3_raw;
    }
    table t_compute_hash_3 { actions = { do_compute_hash_3; } default_action = do_compute_hash_3(); size = 1; }

    // --- 三轨并行的布隆过滤器寄存器 ---
    Register<bit<1>, bit<16>>(65536, 0) bloom_filter_1;
    RegisterAction<bit<1>, bit<16>, bit<1>>(bloom_filter_1) check_bf_1 = {
        void apply(inout bit<1> reg_val, out bit<1> read_val) {
            read_val = reg_val;
            reg_val = 1;
        }
    };

    Register<bit<1>, bit<16>>(65536, 0) bloom_filter_2;
    RegisterAction<bit<1>, bit<16>, bit<1>>(bloom_filter_2) check_bf_2 = {
        void apply(inout bit<1> reg_val, out bit<1> read_val) {
            read_val = reg_val;
            reg_val = 1;
        }
    };

    Register<bit<1>, bit<16>>(65536, 0) bloom_filter_3;
    RegisterAction<bit<1>, bit<16>, bit<1>>(bloom_filter_3) check_bf_3 = {
        void apply(inout bit<1> reg_val, out bit<1> read_val) {
            read_val = reg_val;
            reg_val = 1;
        }
    };

    // --- 独立动作拆分，解除 Stateful ALU 绑定限制 ---
    action do_check_bf_1() { meta.bf_out_1 = check_bf_1.execute(meta.bf_idx_1); }
    table t_check_bf_1 { actions = { do_check_bf_1; } default_action = do_check_bf_1(); size = 1; }

    action do_check_bf_2() { meta.bf_out_2 = check_bf_2.execute(meta.bf_idx_2); }
    table t_check_bf_2 { actions = { do_check_bf_2; } default_action = do_check_bf_2(); size = 1; }

    action do_check_bf_3() { meta.bf_out_3 = check_bf_3.execute(meta.bf_idx_3); }
    table t_check_bf_3 { actions = { do_check_bf_3; } default_action = do_check_bf_3(); size = 1; }

    // --- 终极降维逻辑 (查表替代 ALU 链式位运算) ---
    action do_set_new_flow(bit<1> flag) {
        meta.is_new_flow = flag;
    }
    table t_eval_new_flow {
        key = {
            meta.bf_out_1 : exact;
            meta.bf_out_2 : exact;
            meta.bf_out_3 : exact;
        }
        actions = { do_set_new_flow; }
        // 默认行为：只要不是全 1，就判定为新流 (置 1)
        default_action = do_set_new_flow(1);
        // 常量表项：只有当三个布隆槽位全为 1 时，才判定为旧流 (置 0)
        const entries = {
            (1, 1, 1) : do_set_new_flow(0);
        }
        size = 16;
    }

    apply {
        if (hdr.ipv4.isValid() && hdr.udp.isValid()) {
            // 阶段 1：独立计算三个哈希
            t_compute_hash_1.apply();
            t_compute_hash_2.apply();
            t_compute_hash_3.apply();

            // 阶段 2：并行过筛（编译器会自动将这三个表压入同一个物理 Stage）
            t_check_bf_1.apply();
            t_check_bf_2.apply();
            t_check_bf_3.apply();

            // 阶段 3：状态结算 (查表法)
            t_eval_new_flow.apply();
        }
    }
}
