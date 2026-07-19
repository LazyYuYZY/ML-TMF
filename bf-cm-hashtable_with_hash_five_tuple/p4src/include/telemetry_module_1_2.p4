/* -*- P4_16 -*- */
// =========================================================================
// 模块 1.2：新流转发展开式缓存 (V2.0: 环形0-1, 2行×4块=8寄存器, 完整五元组)
// 从模块1.1接收 is_new_flow 信号，满载时向模块1.3发出 do_clone 信号
// =========================================================================
control Module_1_2_IPCache(
    inout headers_t hdr,
    inout my_ingress_metadata_t meta,
    in    ingress_intrinsic_metadata_t ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    // =========================================================================
    // 1. 全局环形计数器单元 (V2.0: 缩减为 0~1 循环)
    // =========================================================================
    Register<bit<8>, bit<1>>(1, 0) circular_counter;
    RegisterAction<bit<8>, bit<1>, bit<8>>(circular_counter) update_cnt = {
        void apply(inout bit<8> reg_val, out bit<8> cnt) {
            cnt = reg_val;
            if (reg_val == 1) {  // 逢 1 归零
                reg_val = 0;
            } else {
                reg_val = reg_val + 1;
            }
        }
    };

    action do_update_counter() {
        meta.current_cnt = update_cnt.execute(0);
    }
    table t_update_counter {
        actions = { do_update_counter; }
        default_action = do_update_counter();
        size = 1;
    }

    // =========================================================================
    // 2. 五元组预处理提取 (将报文头部字段提前搬移到 meta，减轻 ALU IXBar 压力)
    // =========================================================================
    action do_prepare_5tuple() {
        meta.prep_src = hdr.ipv4.src_addr;
        meta.prep_dst = hdr.ipv4.dst_addr;
        meta.prep_ports = hdr.udp.src_port ++ hdr.udp.dst_port;
        meta.prep_proto = (bit<32>)hdr.ipv4.protocol;
    }
    table t_prepare_5tuple {
        actions = { do_prepare_5tuple; }
        default_action = do_prepare_5tuple();
        size = 1;
    }

    // =========================================================================
    // 3. 横向展开式五元组缓存矩阵 (V2.0: 2行×4块，引入令牌传递强制串行)
    // =========================================================================

    // --- Slot 0 (暗号: 0x2) --- 4个独立的32-bit物理块
    Register<bit<32>, bit<1>>(1, 0) reg_s0;
    Register<bit<32>, bit<1>>(1, 0) reg_d0;
    Register<bit<32>, bit<1>>(1, 0) reg_p0;
    Register<bit<32>, bit<1>>(1, 0) reg_pr0;

    RegisterAction<bit<32>, bit<1>, bit<32>>(reg_s0) rw_s0 = {
        void apply(inout bit<32> reg_val, out bit<32> ret_val) { ret_val = reg_val; if (meta.cache_ctrl == 0x2) { reg_val = meta.prep_src; } }
    };
    RegisterAction<bit<32>, bit<1>, bit<32>>(reg_d0) rw_d0 = {
        void apply(inout bit<32> reg_val, out bit<32> ret_val) { ret_val = reg_val; if (meta.cache_ctrl == 0x2) { reg_val = meta.prep_dst; } }
    };
    RegisterAction<bit<32>, bit<1>, bit<32>>(reg_p0) rw_p0 = {
        void apply(inout bit<32> reg_val, out bit<32> ret_val) { ret_val = reg_val; if (meta.cache_ctrl == 0x2) { reg_val = meta.prep_ports; } }
    };
    RegisterAction<bit<32>, bit<1>, bit<32>>(reg_pr0) rw_pr0 = {
        void apply(inout bit<32> reg_val, out bit<32> ret_val) { ret_val = reg_val; if (meta.cache_ctrl == 0x2) { reg_val = meta.prep_proto; } }
    };

    action do_rw_s0() { hdr.report.src_addr_0 = rw_s0.execute(0); meta.dep_token = meta.dep_token + 1; }
    table t_rw_s0 { actions = { do_rw_s0; } default_action = do_rw_s0(); size = 1; }
    action do_rw_d0() { hdr.report.dst_addr_0 = rw_d0.execute(0); meta.dep_token = meta.dep_token + 1; }
    table t_rw_d0 { actions = { do_rw_d0; } default_action = do_rw_d0(); size = 1; }
    action do_rw_p0() { hdr.report.ports_0 = rw_p0.execute(0); meta.dep_token = meta.dep_token + 1; }
    table t_rw_p0 { actions = { do_rw_p0; } default_action = do_rw_p0(); size = 1; }
    action do_rw_pr0() { hdr.report.proto_0 = rw_pr0.execute(0); meta.dep_token = meta.dep_token + 1; }
    table t_rw_pr0 { actions = { do_rw_pr0; } default_action = do_rw_pr0(); size = 1; }

    // --- Slot 1 (暗号: 0x3) --- 4个独立的32-bit物理块
    Register<bit<32>, bit<1>>(1, 0) reg_s1;
    Register<bit<32>, bit<1>>(1, 0) reg_d1;
    Register<bit<32>, bit<1>>(1, 0) reg_p1;
    Register<bit<32>, bit<1>>(1, 0) reg_pr1;

    RegisterAction<bit<32>, bit<1>, bit<32>>(reg_s1) rw_s1 = {
        void apply(inout bit<32> reg_val, out bit<32> ret_val) { ret_val = reg_val; if (meta.cache_ctrl == 0x3) { reg_val = meta.prep_src; } }
    };
    RegisterAction<bit<32>, bit<1>, bit<32>>(reg_d1) rw_d1 = {
        void apply(inout bit<32> reg_val, out bit<32> ret_val) { ret_val = reg_val; if (meta.cache_ctrl == 0x3) { reg_val = meta.prep_dst; } }
    };
    RegisterAction<bit<32>, bit<1>, bit<32>>(reg_p1) rw_p1 = {
        void apply(inout bit<32> reg_val, out bit<32> ret_val) { ret_val = reg_val; if (meta.cache_ctrl == 0x3) { reg_val = meta.prep_ports; } }
    };
    RegisterAction<bit<32>, bit<1>, bit<32>>(reg_pr1) rw_pr1 = {
        void apply(inout bit<32> reg_val, out bit<32> ret_val) { ret_val = reg_val; if (meta.cache_ctrl == 0x3) { reg_val = meta.prep_proto; } }
    };

    action do_rw_s1() { hdr.report.src_addr_1 = rw_s1.execute(0); meta.dep_token = meta.dep_token + 1; }
    table t_rw_s1 { actions = { do_rw_s1; } default_action = do_rw_s1(); size = 1; }
    action do_rw_d1() { hdr.report.dst_addr_1 = rw_d1.execute(0); meta.dep_token = meta.dep_token + 1; }
    table t_rw_d1 { actions = { do_rw_d1; } default_action = do_rw_d1(); size = 1; }
    action do_rw_p1() { hdr.report.ports_1 = rw_p1.execute(0); meta.dep_token = meta.dep_token + 1; }
    table t_rw_p1 { actions = { do_rw_p1; } default_action = do_rw_p1(); size = 1; }
    action do_rw_pr1() { hdr.report.proto_1 = rw_pr1.execute(0); meta.dep_token = meta.dep_token + 1; }
    table t_rw_pr1 { actions = { do_rw_pr1; } default_action = do_rw_pr1(); size = 1; }

    apply {
        if (hdr.ipv4.isValid()) {

            // 1. 如果是新流，触发计数器翻转
            if (meta.is_new_flow == 1) {
                t_update_counter.apply();
            }

            // 2. 融合单变量控制字: 1 bit (is_new_flow) + 1 bit (cnt[0:0]) = 2 bits
            meta.cache_ctrl = meta.is_new_flow ++ meta.current_cnt[0:0];

            // 3. 传送带流动：激活报文头部，通过显式 apply 触发相邻 Stage 的物理并行过筛
            hdr.report.setValid();

            // 预处理五元组提取
            t_prepare_5tuple.apply();

            // 串行化执行，强制分 Stage 规避 IXBar 溢出
            t_rw_s0.apply();
            t_rw_d0.apply();
            t_rw_p0.apply();
            t_rw_pr0.apply();

            t_rw_s1.apply();
            t_rw_d1.apply();
            t_rw_p1.apply();
            t_rw_pr1.apply();

            // 4. 门控判定：当第 2 个槽位装满且为新流时 (0x3 = 11)，发出上报信号
            if (meta.cache_ctrl == 0x3) {
                meta.do_clone = 1;
            }
        }
    }
}
