/* -*- P4_16 -*- */
#include <core.p4>
#include <tna.p4>

// =========================================================================
// 1. 独立编译所需的最小化定义 (Headers & Metadata)
// =========================================================================
const bit<16> ETHERTYPE_IPV4 = 0x0800;
const bit<8>  IP_PROTO_UDP   = 17;

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

struct headers_t {
    ethernet_h ethernet;
    ipv4_h     ipv4;
    udp_h      udp;
}

// 包含了供 V2.0 布隆过滤器使用的全套变量
struct my_ingress_metadata_t {
    bit<32> h1_raw;
    bit<32> h2_raw;
    bit<32> h3_raw;
    bit<16> bf_idx_1;
    bit<16> bf_idx_2;
    bit<16> bf_idx_3;
    bit<1>  bf_out_1;
    bit<1>  bf_out_2;
    bit<1>  bf_out_3;
    bit<1>  is_new_flow; 
}

struct my_egress_metadata_t {
    bit<8> dummy;
}

// =========================================================================
// 2. 解析器 (Parser)
// =========================================================================
parser IngressParser(
    packet_in pkt,
    out headers_t hdr,
    out my_ingress_metadata_t meta,
    out ingress_intrinsic_metadata_t ig_intr_md) {
    
    state start {
        // [防坑指南] 必须在这里把元数据全部清零，防止 Tofino 硬件分配脏内存
        meta.h1_raw = 0; meta.h2_raw = 0; meta.h3_raw = 0;
        meta.bf_idx_1 = 0; meta.bf_idx_2 = 0; meta.bf_idx_3 = 0;
        meta.bf_out_1 = 0; meta.bf_out_2 = 0; meta.bf_out_3 = 0;
        meta.is_new_flow = 0;

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

// =========================================================================
// 3. 入口控制逻辑 (V2.0 核心：三哈希拆分 + 物理拆分布隆 + 查表降维)
// =========================================================================
control Ingress(
    inout headers_t hdr,
    inout my_ingress_metadata_t meta,
    in    ingress_intrinsic_metadata_t ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    // --- 哈希计算单元 (拆分为三个独立表，规避 32-bit 总线限制) ---
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

    // --- 独立测试的基础转发 ---
    action send(bit<9> port) { ig_tm_md.ucast_egress_port = port; }
    table port_match {
        key = { ig_intr_md.ingress_port: exact; }
        actions = { send; }
        size = 64;
    }

    // =========================================================================
    // 4. 流水线装配
    // =========================================================================
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
        port_match.apply(); // 打通数据面转发
    }
}

// =========================================================================
// 5. 收尾占位符 (Deparser & Egress) 
// =========================================================================
control IngressDeparser(
    packet_out pkt,
    inout headers_t hdr,
    in my_ingress_metadata_t meta,
    in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {
    apply {
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.udp);
    }
}

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

control Egress(
    inout headers_t hdr, 
    inout my_egress_metadata_t meta, 
    in egress_intrinsic_metadata_t eg_intr_md, 
    in egress_intrinsic_metadata_from_parser_t eg_prsr_md, 
    inout egress_intrinsic_metadata_for_deparser_t eg_dprsr_md, 
    inout egress_intrinsic_metadata_for_output_port_t eg_port_md) {
    
    apply {}
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
    }
}

// =========================================================================
// 6. 实例化主流水线
// =========================================================================
Pipeline(IngressParser(), Ingress(), IngressDeparser(), EgressParser(), Egress(), EgressDeparser()) pipe;
Switch(pipe) main;