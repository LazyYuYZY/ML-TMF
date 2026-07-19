/* -*- P4_16 -*- */
// =========================================================================
// 模块 1.1：新流检测 (V3.0: 哈希输入改为源目IP)
// =========================================================================
control Module_1_1_BloomFilter(
    inout headers_t hdr,
    inout my_ingress_metadata_t meta,
    in    ingress_intrinsic_metadata_t ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    CRCPolynomial<bit<32>>(coeff=0x04C11DB7, reversed=true, msb=false, extended=false, init=0xFFFFFFFF, xor=0xFFFFFFFF) poly1;
    CRCPolynomial<bit<32>>(coeff=0x741B8CD7, reversed=true, msb=false, extended=false, init=0xFFFFFFFF, xor=0xFFFFFFFF) poly2;
    CRCPolynomial<bit<32>>(coeff=0xDB710641, reversed=true, msb=false, extended=false, init=0xFFFFFFFF, xor=0xFFFFFFFF) poly3;

    Hash<bit<32>>(HashAlgorithm_t.CUSTOM, poly1) hash_algo1;
    Hash<bit<32>>(HashAlgorithm_t.CUSTOM, poly2) hash_algo2;
    Hash<bit<32>>(HashAlgorithm_t.CUSTOM, poly3) hash_algo3;

    action do_compute_hash_1() {
        meta.h1_raw = hash_algo1.get({hdr.ipv4.src_addr, hdr.ipv4.dst_addr});
        meta.bf_idx_1 = (bit<16>)meta.h1_raw;
    }
    table t_compute_hash_1 { actions = { do_compute_hash_1; } default_action = do_compute_hash_1(); size = 1; }

    action do_compute_hash_2() {
        meta.h2_raw = hash_algo2.get({hdr.ipv4.src_addr, hdr.ipv4.dst_addr});
        meta.bf_idx_2 = (bit<16>)meta.h2_raw;
    }
    table t_compute_hash_2 { actions = { do_compute_hash_2; } default_action = do_compute_hash_2(); size = 1; }

    action do_compute_hash_3() {
        meta.h3_raw = hash_algo3.get({hdr.ipv4.src_addr, hdr.ipv4.dst_addr});
        meta.bf_idx_3 = (bit<16>)meta.h3_raw;
    }
    table t_compute_hash_3 { actions = { do_compute_hash_3; } default_action = do_compute_hash_3(); size = 1; }

    Register<bit<1>, bit<16>>(65536, 0) bloom_filter_1;
    RegisterAction<bit<1>, bit<16>, bit<1>>(bloom_filter_1) check_bf_1 = {
        void apply(inout bit<1> reg_val, out bit<1> read_val) { read_val = reg_val; reg_val = 1; }
    };
    Register<bit<1>, bit<16>>(65536, 0) bloom_filter_2;
    RegisterAction<bit<1>, bit<16>, bit<1>>(bloom_filter_2) check_bf_2 = {
        void apply(inout bit<1> reg_val, out bit<1> read_val) { read_val = reg_val; reg_val = 1; }
    };
    Register<bit<1>, bit<16>>(65536, 0) bloom_filter_3;
    RegisterAction<bit<1>, bit<16>, bit<1>>(bloom_filter_3) check_bf_3 = {
        void apply(inout bit<1> reg_val, out bit<1> read_val) { read_val = reg_val; reg_val = 1; }
    };

    action do_check_bf_1() { meta.bf_out_1 = check_bf_1.execute(meta.bf_idx_1); }
    table t_check_bf_1 { actions = { do_check_bf_1; } default_action = do_check_bf_1(); size = 1; }
    action do_check_bf_2() { meta.bf_out_2 = check_bf_2.execute(meta.bf_idx_2); }
    table t_check_bf_2 { actions = { do_check_bf_2; } default_action = do_check_bf_2(); size = 1; }
    action do_check_bf_3() { meta.bf_out_3 = check_bf_3.execute(meta.bf_idx_3); }
    table t_check_bf_3 { actions = { do_check_bf_3; } default_action = do_check_bf_3(); size = 1; }

    action do_set_new_flow(bit<1> flag) { meta.is_new_flow = flag; }
    table t_eval_new_flow {
        key = { meta.bf_out_1 : exact; meta.bf_out_2 : exact; meta.bf_out_3 : exact; }
        actions = { do_set_new_flow; }
        default_action = do_set_new_flow(1);
        const entries = { (1, 1, 1) : do_set_new_flow(0); }
        size = 16;
    }

    apply {
        if (hdr.ipv4.isValid() && hdr.udp.isValid()) {
            t_compute_hash_1.apply();
            t_compute_hash_2.apply();
            t_compute_hash_3.apply();
            t_check_bf_1.apply();
            t_check_bf_2.apply();
            t_check_bf_3.apply();
            t_eval_new_flow.apply();
        }
    }
}
