/* -*- P4_16 -*- */
#include <core.p4>
#include <tna.p4>

// =========================================================================
// 导入阶段一：全局契约 (定义、报文头、元数据总线)
// =========================================================================
#include "include/telemetry_defines.p4"

// =========================================================================
// 导入阶段二：解耦后的功能子模块
// =========================================================================
#include "include/telemetry_module_1_1.p4"
#include "include/telemetry_module_1_2.p4"
#include "include/telemetry_module_1_3.p4"
#include "include/telemetry_module_2.p4"
#include "include/telemetry_module_3.p4"

// =========================================================================
// 全局解析器
// =========================================================================
parser IngressParser(
    packet_in pkt,
    out headers_t hdr,
    out my_ingress_metadata_t meta,
    out ingress_intrinsic_metadata_t ig_intr_md) {

    state start {
        // [防坑指南] 必须在这里把元数据总线全部清零，防止 Tofino 硬件分配脏内存
        // --- 总线控制信号 ---
        meta.is_new_flow = 0;
        meta.do_clone = 0;
        meta.is_probe = 0;
        meta.is_test_pkt = 0;

        // --- 模块 1.1：布隆过滤器 ---
        meta.h1_raw = 0;       meta.h2_raw = 0;       meta.h3_raw = 0;
        meta.bf_idx_1 = 0;     meta.bf_idx_2 = 0;     meta.bf_idx_3 = 0;
        meta.bf_out_1 = 0;     meta.bf_out_2 = 0;     meta.bf_out_3 = 0;

        // --- 模块 1.2：IP缓存 ---
        meta.current_cnt = 0;  meta.cache_ctrl = 0;
        meta.prep_src = 0;     meta.prep_dst = 0;
        meta.prep_ports = 0;   meta.prep_proto = 0;
        meta.dep_token = 0;

        // --- 模块 2：CM Sketch ---
        meta.s1_idx = 0;       meta.s2_idx = 0;       meta.s3_idx = 0;
        meta.s1_out = 0;       meta.s2_out = 0;       meta.s3_out = 0;

        // --- 模块 3：训练数据收集 ---
        meta.flow_key = 0;     meta.train_idx = 0;
        meta.train_key_match = 0;
        meta.train_key_out = 0; meta.train_size_out = 0;

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
            PROBE_PORT: parse_probe; // 拦截 9999 端口的探针包
            default: accept;
        }
    }

    state parse_probe {
        pkt.extract(hdr.probe);
        transition accept;
    }
}

// =========================================================================
// 全局入口流水线
// =========================================================================
control Ingress(
    inout headers_t hdr,
    inout my_ingress_metadata_t meta,
    in    ingress_intrinsic_metadata_t ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    // 实例化子模块
    Module_1_1_BloomFilter() mod_1_1;
    Module_1_2_IPCache()     mod_1_2;
    Module_2_Sketch()        mod_2;
    Module_3_TrainData()     mod_3;

    // 全局基础转发单元 (U-Turn 打流)
    action send(bit<9> port) { ig_tm_md.ucast_egress_port = port; }
    table port_match {
        key = { ig_intr_md.ingress_port: exact; }
        actions = { send; }
        size = 64;
    }

    apply {
        // --- 遥测核心逻辑流水线 ---
        // 严格按照功能依赖顺序，通过 meta 总线线速握手
        mod_1_1.apply(hdr, meta, ig_intr_md, ig_prsr_md, ig_dprsr_md, ig_tm_md);
        mod_1_2.apply(hdr, meta, ig_intr_md, ig_prsr_md, ig_dprsr_md, ig_tm_md);
        mod_2.apply(hdr, meta, ig_intr_md, ig_prsr_md, ig_dprsr_md, ig_tm_md);
        mod_3.apply(hdr, meta, ig_intr_md, ig_prsr_md, ig_dprsr_md, ig_tm_md);

        // --- 物理层路由 ---
        port_match.apply();
    }
}

// =========================================================================
// 全局入口逆解析器 - 核心克隆触发地
// =========================================================================
control IngressDeparser(
    packet_out pkt,
    inout headers_t hdr,
    in my_ingress_metadata_t meta,
    in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {

    Mirror() mirror;

    apply {
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.udp);
        pkt.emit(hdr.report); // 满载时这里面已经被填满了五元组特征
        pkt.emit(hdr.probe);  // 探针包原路反射时携带的数据

        // 监听模块 1.2 发出的信号，一旦满载，瞬间克隆发往收集服务器
        if (meta.do_clone == 1) {
            mirror.emit(MIRROR_SESSION_ID);
        }
    }
}

// =========================================================================
// 出口流水线
// =========================================================================
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
    in    egress_intrinsic_metadata_t eg_intr_md,
    in    egress_intrinsic_metadata_from_parser_t eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t eg_port_md) {

    Module_1_3_Egress() mod_1_3;

    apply {
        // 调用 1.3 的逻辑，完成克隆遥测包与原始业务包的剥离与隔离
        mod_1_3.apply(hdr, meta, eg_intr_md, eg_prsr_md, eg_dprsr_md, eg_port_md);
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
        pkt.emit(hdr.report); // 仅克隆包保留此头部
        pkt.emit(hdr.probe);
    }
}

// =========================================================================
// 系统挂载主干
// =========================================================================
Pipeline(
    IngressParser(),
    Ingress(),
    IngressDeparser(),
    EgressParser(),
    Egress(),
    EgressDeparser()) pipe;

Switch(pipe) main;
