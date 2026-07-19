/* -*- P4_16 -*- */
#include <core.p4>
#include <tna.p4>

const bit<16> ETHERTYPE_IPV4 = 0x0800;
const bit<8>  IP_PROTO_UDP   = 17;

/*************************************************************************
 *********************** H E A D E R S  *********************************
 *************************************************************************/
header ethernet_h {
    bit<48> dst_addr;
    bit<48> src_addr;
    bit<16> ether_type;
}

header ipv4_h {
    bit<4>  version;
    bit<4>  ihl;
    bit<8>  diffserv;
    bit<16> total_len;
    bit<16> identification;
    bit<3>  flags;
    bit<13> frag_offset;
    bit<8>  ttl;
    bit<8>  protocol;
    bit<16> hdr_checksum;
    bit<32> src_addr;
    bit<32> dst_addr;
}

header udp_h {
    bit<16> src_port;
    bit<16> dst_port;
    bit<16> hdr_length;
    bit<16> checksum;
}

// 新流批量上报头部 (缩减为 2 个槽位，每槽位存五元组的 4 个 32-bit 拆分块)
header report_t {
    // --- Slot 0: 五元组拆分块 ---
    bit<32> src_addr_0;     // Block 1
    bit<32> dst_addr_0;     // Block 2
    bit<32> ports_0;        // Block 3 (src_port ++ dst_port)
    bit<32> proto_0;        // Block 4 (protocol 扩展至 32 bit)
    // --- Slot 1: 五元组拆分块 ---
    bit<32> src_addr_1;
    bit<32> dst_addr_1;
    bit<32> ports_1;
    bit<32> proto_1;
}

struct headers_t {
    ethernet_h ethernet;
    ipv4_h     ipv4;
    udp_h      udp;
    report_t   report;
}

// 全局通信元数据总线接口 (留好前后通信接口)
struct my_ingress_metadata_t {
    // 【前后总线通信接口】
    bit<1>  is_new_flow;          // [输入接口] 来自模块 1.1 的布隆过滤器判定结果
    bit<1>  do_clone;             // [输出接口] 传递给模块 1.3 的满载克隆触发信号
    
    // 【模块 1.2 内部状态】
    bit<8>  current_cnt;          // 环形计数器当前指针 (0~1)
    bit<2>  cache_ctrl;           // 位宽折叠控制字 = {is_new_flow, current_cnt[0:0]}
    
    // 【预处理暂存区】(减轻 ALU 输入交叉开关压力)
    bit<32> prep_src;
    bit<32> prep_dst;
    bit<32> prep_ports;
    bit<32> prep_proto;
    
    // 【强制串行依赖令牌】(打破编译器并行优化，强制分 Stage)
    bit<32> dep_token;
}

struct my_egress_metadata_t {
    bit<8> dummy;
}

/*************************************************************************
 ************** I N G R E S S   P A R S E R S   **************************
 *************************************************************************/
parser IngressParser(
    packet_in pkt,
    out headers_t hdr,
    out my_ingress_metadata_t meta,
    out ingress_intrinsic_metadata_t ig_intr_md) {
    
    state start {
        // [防坑指南] 必须在这里把元数据全部清零，防止 Tofino 硬件分配脏内存
        meta.do_clone = 0;
        meta.current_cnt = 0;
        meta.cache_ctrl = 0;
        meta.prep_src = 0;
        meta.prep_dst = 0;
        meta.prep_ports = 0;
        meta.prep_proto = 0;
        meta.dep_token = 0;
        
        // 【硬编码模拟】模拟前序模块 1.1 始终发来新流信号
        meta.is_new_flow = 1; 
        
        pkt.extract(ig_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        transition parse_ethernet;
    }
    
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4: parse_ipv4;
            default: accept;
        }
    }
    
    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            IP_PROTO_UDP: parse_udp;
            default: accept;
        }
    }
    
    state parse_udp {
        pkt.extract(hdr.udp);
        transition accept;
    }
}

control IngressDeparser(
    packet_out pkt,
    inout headers_t hdr,
    in my_ingress_metadata_t meta,
    in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {
    apply {
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.udp);
        pkt.emit(hdr.report);
    }
}

/*************************************************************************
 ********************* I N G R E S S   C O N T R O L *********************
 *************************************************************************/
control Ingress(
    inout headers_t hdr,
    inout my_ingress_metadata_t meta,
    in    ingress_intrinsic_metadata_t ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    // =========================================================================
    // 1. 全局环形计数器单元 (缩减为 0~1 循环)
    // =========================================================================
    Register<bit<8>, bit<1>>(1, 0) circular_counter;
    RegisterAction<bit<8>, bit<1>, bit<8>>(circular_counter) update_cnt = {
        void apply(inout bit<8> reg_val, out bit<8> cnt) {
            cnt = reg_val; 
            if (reg_val == 1) {
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
    // 3. 横向展开式五元组缓存矩阵 (引入令牌传递强制串行，彻底解决 IXBar 溢出)
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

    // --- 基础转发端口匹配表 ---
    action send(bit<9> port) { 
        ig_tm_md.ucast_egress_port = port; 
    }
    
    table port_match {
        key = { ig_intr_md.ingress_port: exact; }
        actions = { send; }
        size = 64;
    }

    apply {
        if (hdr.ipv4.isValid()) {
            
            // 1. 如果是新流，触发计数器翻转
            if (meta.is_new_flow == 1) {
                t_update_counter.apply();
            }
            
            // 2. 融合单变量控制字: 1 bit (is_new_flow) + 1 bit (cnt) = 2 bits
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
        port_match.apply();
    }
}

/*************************************************************************
 ********************** E G R E S S   C O N T R O L **********************
 *************************************************************************/
parser EgressParser(
    packet_in pkt, 
    out headers_t hdr, 
    out my_egress_metadata_t meta, 
    out egress_intrinsic_metadata_t eg_intr_md) {
    
    state start { 
        meta.dummy = 0; 
        pkt.extract(eg_intr_md); 
        transition accept; 
    }
}

control EgressDeparser(
    packet_out pkt, 
    inout headers_t hdr, 
    in my_egress_metadata_t meta, 
    in egress_intrinsic_metadata_for_deparser_t eg_dprsr_md) {
    
    apply { 
        pkt.emit(hdr.ethernet); 
        pkt.emit(hdr.ipv4); 
        pkt.emit(hdr.udp); 
        pkt.emit(hdr.report); 
    }
}

control Egress(
    inout headers_t hdr, 
    inout my_egress_metadata_t meta, 
    in egress_intrinsic_metadata_t eg_intr_md, 
    in egress_intrinsic_metadata_from_parser_t eg_prsr_md, 
    inout egress_intrinsic_metadata_for_deparser_t eg_dprsr_md, 
    inout egress_intrinsic_metadata_for_output_port_t eg_port_md) {
    
    apply {}
}

Pipeline(IngressParser(), Ingress(), IngressDeparser(), EgressParser(), Egress(), EgressDeparser()) pipe;
Switch(pipe) main;

