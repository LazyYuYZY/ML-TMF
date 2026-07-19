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

// 包含 Module 3 回填字段的探针头 (train_key 扩展为 32 位)
header probe_t {
    bit<32> seq_id;
    bit<16> sketch_1_val;
    bit<16> sketch_2_val;
    bit<16> sketch_3_val;
    bit<32> train_size;
    bit<32> train_key;  // 修改点：扩展为 32 位，移除原 2 位 padding
}

struct headers_t {
    ethernet_h ethernet;
    ipv4_h     ipv4;
    udp_h      udp;
    probe_t    probe;
}

struct my_ingress_metadata_t {
    bit<1>  is_probe;        
    bit<32> hash_raw;
    
    // 修改点：流的全局唯一标志扩展为 32 位完整哈希
    bit<32> flow_key;        
    
    // 寻址索引与中间状态 (严格遵循“逢加必初”防黑洞规范)
    bit<10> train_idx;
    bit<1>  train_key_match;    
    
    // 修改点：寄存器读取回填扩展为 32 位
    bit<32> train_key_out;
    bit<32> train_size_out;
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
        // 关键：初始化所有输出元数据
        meta.is_probe = 0;
        meta.hash_raw = 0;
        meta.flow_key = 0;
        meta.train_idx = 0;
        meta.train_key_match = 0;
        meta.train_key_out = 0;
        meta.train_size_out = 0;
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
    // HASH ALGORITHM (模拟 Module 1.1 前置计算的首个哈希结果)
    // =========================================================================
    CRCPolynomial<bit<32>>(coeff=0x04C11DB7, reversed=true, msb=false, extended=false, init=0xFFFFFFFF, xor=0xFFFFFFFF) poly1;
    Hash<bit<32>>(HashAlgorithm_t.CUSTOM, poly1) hash_algo1;
    
    action do_mock_flow_key() {
        meta.hash_raw = hash_algo1.get({hdr.ipv4.src_addr, hdr.ipv4.dst_addr, hdr.ipv4.protocol, hdr.udp.src_port, hdr.udp.dst_port});
        
        // 修改点：完整保留 32 位哈希结果作为 flow_key
        meta.flow_key = meta.hash_raw; 
        // 截取低 10 位作为 1024 行表的物理索引
        meta.train_idx = meta.flow_key[9:0];     
    }
    table t_mock_flow_key { 
        actions = { do_mock_flow_key; } 
        default_action = do_mock_flow_key(); 
        size = 1; 
    }

    // =========================================================================
    // TRAINING DATA COLLECTION (核心拆解：解决级联死锁)
    // =========================================================================
    
    // --- 拆分 1：Key 表项处理 ---
    // 修改点：物理寄存器升级为标准 32-bit，存储完整哈希
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

    // =========================================================================
    // ROUTING
    // =========================================================================
    action send(bit<9> port) { ig_tm_md.ucast_egress_port = port; }
    table port_match {
        key = { ig_intr_md.ingress_port: exact; }
        actions = { send; }
        size = 64;
    }

    apply {
        if (hdr.ipv4.isValid()) {
            
            // --- 阶段 1：解析参数 ---
            if (hdr.probe.isValid() && hdr.udp.dst_port == PROBE_PORT) {
                meta.is_probe = 1;
                // 截取探针 seq_id 的低 10 位作为拉取索引
                meta.train_idx = hdr.probe.seq_id[9:0];
                hdr.udp.checksum = 0; 
            } else {
                meta.is_probe = 0;
                t_mock_flow_key.apply();
            }
            
            // --- 阶段 2：强制表级联隔离（Tofino 核心命脉） ---
            
            // 动作 A：读 Key
            t_update_train_key.apply();
            
            // 全局控制流：隔离判断
            if (meta.is_probe == 0) {
                // 修改点：32位完整哈希比对，若相同则认为是同一个流（允许精度损失）
                if (meta.train_key_out == meta.flow_key || meta.train_key_out == 0) {
                    meta.train_key_match = 1;
                } else {
                    meta.train_key_match = 0; // 老鼠流冲突时，直接放弃计数（用于支撑漏报率论点）
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
    
    apply {}
}

Pipeline(
    IngressParser(), Ingress(), IngressDeparser(),
    EgressParser(), Egress(), EgressDeparser()
) pipe;

Switch(pipe) main;
