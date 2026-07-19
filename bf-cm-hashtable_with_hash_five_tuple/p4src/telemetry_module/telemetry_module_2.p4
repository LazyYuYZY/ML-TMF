/* -*- P4_16 -*- */
#include <core.p4>
#include <tna.p4>

const bit<16> ETHERTYPE_IPV4 = 0x0800;
const bit<8>  IP_PROTO_UDP   = 17;
const bit<16> PROBE_PORT     = 9999;

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

// 专用探针头部，用于拉取 Sketch 数据
header probe_t {
    bit<32> seq_id;
    bit<16> sketch_1_val;
    bit<16> sketch_2_val;
    bit<16> sketch_3_val;
    bit<32> dummy_pad; // 对齐位宽
}

struct headers_t {
    ethernet_h ethernet;
    ipv4_h     ipv4;
    udp_h      udp;
    probe_t    probe;
}

struct my_ingress_metadata_t {
    bit<1>  is_probe;        
    // 保留哈希总线接口，接收前序或模拟注入的哈希结果
    bit<32> h1_raw;
    bit<32> h2_raw;
    bit<32> h3_raw;
    
    // 最终截取的 14 位物理寻址索引
    bit<14> s1_idx;
    bit<14> s2_idx;
    bit<14> s3_idx;
    
    // 寄存器读取出参暂存
    bit<16> s1_out;
    bit<16> s2_out;
    bit<16> s3_out;
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
        // 关键避坑：初始化所有输出元数据，防止 Parser 悬空黑洞
        meta.is_probe = 0;
        meta.h1_raw = 0;
        meta.h2_raw = 0;
        meta.h3_raw = 0;
        meta.s1_idx = 0;
        meta.s2_idx = 0;
        meta.s3_idx = 0;
        meta.s1_out = 0;
        meta.s2_out = 0;
        meta.s3_out = 0;
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
        transition select(hdr.udp.dst_port) {
            PROBE_PORT: parse_probe;
            default: accept;
        }
    }
    
    state parse_probe {
        pkt.extract(hdr.probe);
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
        pkt.emit(hdr.probe);
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
    // MOCK HASH (模拟前序模块传入的哈希结果)
    // =========================================================================
    action do_mock_hash() {
        // 硬编码模拟前序模块计算出的三个哈希结果
        // 利用报文头部字段简单拼凑以产生数据依赖模拟
        meta.h1_raw = (bit<32>)hdr.ipv4.src_addr;
        meta.h2_raw = (bit<32>)hdr.ipv4.dst_addr;
        meta.h3_raw = (bit<32>)hdr.ipv4.src_addr ^ (bit<32>)hdr.ipv4.dst_addr ^ (bit<32>)hdr.udp.src_port;
        
        // 截取低 14 位作为物理索引
        meta.s1_idx = meta.h1_raw[13:0];
        meta.s2_idx = meta.h2_raw[13:0];
        meta.s3_idx = meta.h3_raw[13:0];
    }
    table t_mock_hash {
        actions = { do_mock_hash; }
        default_action = do_mock_hash();
        size = 1;
    }

    // =========================================================================
    // CM SKETCH REGISTERS & ACTIONS (保持原状)
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

    // =========================================================================
    // FORWARDING & ROUTING
    // =========================================================================
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
                // 调用模拟前序注入的哈希结果
                t_mock_hash.apply();
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
        
        // 匹配打通 U-Turn 链路的物理口表项
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
        pkt.emit(hdr.probe);
    }
}

control Egress(
    inout headers_t hdr, 
    inout my_egress_metadata_t meta, 
    in    egress_intrinsic_metadata_t eg_intr_md,
    in    egress_intrinsic_metadata_from_parser_t eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t eg_port_md) {
    
    apply {
        // Module 2 暂无特殊的 Egress 逻辑，保持默认通过即可
    }
}

Pipeline(
    IngressParser(), Ingress(), IngressDeparser(),
    EgressParser(), Egress(), EgressDeparser()
) pipe;

Switch(pipe) main;
