/* -*- P4_16 -*- */
// =========================================================================
// 模块 1.2：新流转发展开式缓存 (V3.0: 4行，仅存源目IP)
// =========================================================================
control Module_1_2_IPCache(
    inout headers_t hdr,
    inout my_ingress_metadata_t meta,
    in    ingress_intrinsic_metadata_t ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    Register<bit<8>, bit<1>>(1, 0) circular_counter;
    RegisterAction<bit<8>, bit<1>, bit<8>>(circular_counter) update_cnt = {
        void apply(inout bit<8> reg_val, out bit<8> cnt) {
            cnt = reg_val;
            if (reg_val == 3) { reg_val = 0; } else { reg_val = reg_val + 1; }
        }
    };
    action do_update_counter() { meta.current_cnt = update_cnt.execute(0); }
    table t_update_counter { actions = { do_update_counter; } default_action = do_update_counter(); size = 1; }

    action do_prepare_ip() {
        meta.prep_src = hdr.ipv4.src_addr;
        meta.prep_dst = hdr.ipv4.dst_addr;
    }
    table t_prepare_ip { actions = { do_prepare_ip; } default_action = do_prepare_ip(); size = 1; }

    // Slot 0
    Register<bit<32>, bit<1>>(1, 0) reg_src_0;
    Register<bit<32>, bit<1>>(1, 0) reg_dst_0;
    RegisterAction<bit<32>, bit<1>, bit<32>>(reg_src_0) rw_s0 = { void apply(inout bit<32> reg_val, out bit<32> ret_val) { ret_val = reg_val; if (meta.cache_ctrl == 0x4) { reg_val = meta.prep_src; } } };
    RegisterAction<bit<32>, bit<1>, bit<32>>(reg_dst_0) rw_d0 = { void apply(inout bit<32> reg_val, out bit<32> ret_val) { ret_val = reg_val; if (meta.cache_ctrl == 0x4) { reg_val = meta.prep_dst; } } };
    action do_rw_s0() { hdr.report.src_0 = rw_s0.execute(0); }
    table t_rw_s0 { actions = { do_rw_s0; } default_action = do_rw_s0(); size = 1; }
    action do_rw_d0() { hdr.report.dst_0 = rw_d0.execute(0); }
    table t_rw_d0 { actions = { do_rw_d0; } default_action = do_rw_d0(); size = 1; }

    // Slot 1
    Register<bit<32>, bit<1>>(1, 0) reg_src_1;
    Register<bit<32>, bit<1>>(1, 0) reg_dst_1;
    RegisterAction<bit<32>, bit<1>, bit<32>>(reg_src_1) rw_s1 = { void apply(inout bit<32> reg_val, out bit<32> ret_val) { ret_val = reg_val; if (meta.cache_ctrl == 0x5) { reg_val = meta.prep_src; } } };
    RegisterAction<bit<32>, bit<1>, bit<32>>(reg_dst_1) rw_d1 = { void apply(inout bit<32> reg_val, out bit<32> ret_val) { ret_val = reg_val; if (meta.cache_ctrl == 0x5) { reg_val = meta.prep_dst; } } };
    action do_rw_s1() { hdr.report.src_1 = rw_s1.execute(0); }
    table t_rw_s1 { actions = { do_rw_s1; } default_action = do_rw_s1(); size = 1; }
    action do_rw_d1() { hdr.report.dst_1 = rw_d1.execute(0); }
    table t_rw_d1 { actions = { do_rw_d1; } default_action = do_rw_d1(); size = 1; }

    // Slot 2
    Register<bit<32>, bit<1>>(1, 0) reg_src_2;
    Register<bit<32>, bit<1>>(1, 0) reg_dst_2;
    RegisterAction<bit<32>, bit<1>, bit<32>>(reg_src_2) rw_s2 = { void apply(inout bit<32> reg_val, out bit<32> ret_val) { ret_val = reg_val; if (meta.cache_ctrl == 0x6) { reg_val = meta.prep_src; } } };
    RegisterAction<bit<32>, bit<1>, bit<32>>(reg_dst_2) rw_d2 = { void apply(inout bit<32> reg_val, out bit<32> ret_val) { ret_val = reg_val; if (meta.cache_ctrl == 0x6) { reg_val = meta.prep_dst; } } };
    action do_rw_s2() { hdr.report.src_2 = rw_s2.execute(0); }
    table t_rw_s2 { actions = { do_rw_s2; } default_action = do_rw_s2(); size = 1; }
    action do_rw_d2() { hdr.report.dst_2 = rw_d2.execute(0); }
    table t_rw_d2 { actions = { do_rw_d2; } default_action = do_rw_d2(); size = 1; }

    // Slot 3
    Register<bit<32>, bit<1>>(1, 0) reg_src_3;
    Register<bit<32>, bit<1>>(1, 0) reg_dst_3;
    RegisterAction<bit<32>, bit<1>, bit<32>>(reg_src_3) rw_s3 = { void apply(inout bit<32> reg_val, out bit<32> ret_val) { ret_val = reg_val; if (meta.cache_ctrl == 0x7) { reg_val = meta.prep_src; } } };
    RegisterAction<bit<32>, bit<1>, bit<32>>(reg_dst_3) rw_d3 = { void apply(inout bit<32> reg_val, out bit<32> ret_val) { ret_val = reg_val; if (meta.cache_ctrl == 0x7) { reg_val = meta.prep_dst; } } };
    action do_rw_s3() { hdr.report.src_3 = rw_s3.execute(0); }
    table t_rw_s3 { actions = { do_rw_s3; } default_action = do_rw_s3(); size = 1; }
    action do_rw_d3() { hdr.report.dst_3 = rw_d3.execute(0); }
    table t_rw_d3 { actions = { do_rw_d3; } default_action = do_rw_d3(); size = 1; }

    apply {
        if (hdr.ipv4.isValid()) {
            if (meta.is_new_flow == 1) { t_update_counter.apply(); }
            meta.cache_ctrl = meta.is_new_flow ++ meta.current_cnt[1:0];
            hdr.report.setValid();
            t_prepare_ip.apply();
            t_rw_s0.apply(); t_rw_d0.apply();
            t_rw_s1.apply(); t_rw_d1.apply();
            t_rw_s2.apply(); t_rw_d2.apply();
            t_rw_s3.apply(); t_rw_d3.apply();
            if (meta.cache_ctrl == 0x7) { meta.do_clone = 1; }
        }
    }
}
