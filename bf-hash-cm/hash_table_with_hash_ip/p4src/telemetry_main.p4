/* -*- P4_16 -*- */
#include <core.p4>
#include <tna.p4>
#include "include/telemetry_defines.p4"
#include "include/telemetry_module_1_1.p4"
#include "include/telemetry_module_1_2.p4"
#include "include/telemetry_module_1_3.p4"
#include "include/telemetry_module_2.p4"
#include "include/telemetry_module_3.p4"

parser IngressParser(
    packet_in pkt,
    out headers_t hdr,
    out my_ingress_metadata_t meta,
    out ingress_intrinsic_metadata_t ig_intr_md) {
    state start {
        meta.is_new_flow = 0;       meta.do_clone = 0;
        meta.is_probe = 0;          meta.is_test_pkt = 0;
        // --- 模块 1.1：布隆过滤器 ---
        meta.h1_raw = 0;       meta.h2_raw = 0;       meta.h3_raw = 0;
        meta.bf_idx_1 = 0;     meta.bf_idx_2 = 0;     meta.bf_idx_3 = 0;
        meta.bf_out_1 = 0;     meta.bf_out_2 = 0;     meta.bf_out_3 = 0;
        // --- 模块 1.2：IP缓存 ---
        meta.current_cnt = 0;  meta.cache_ctrl = 0;
        meta.prep_src = 0;     meta.prep_dst = 0;
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
            PROBE_PORT: parse_probe;
            default: accept;
        }
    }
    state parse_probe {
        pkt.extract(hdr.probe);
        transition accept;
    }
}

control Ingress(
    inout headers_t hdr,
    inout my_ingress_metadata_t meta,
    in    ingress_intrinsic_metadata_t ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {
    Module_1_1_BloomFilter() mod_1_1;
    Module_1_2_IPCache()     mod_1_2;
    Module_2_Sketch()        mod_2;
    Module_3_TrainData()     mod_3;

    action send(bit<9> port) { ig_tm_md.ucast_egress_port = port; }
    table port_match {
        key = { ig_intr_md.ingress_port: exact; }
        actions = { send; }
        size = 64;
    }
    apply {
        mod_1_1.apply(hdr, meta, ig_intr_md, ig_prsr_md, ig_dprsr_md, ig_tm_md);
        mod_1_2.apply(hdr, meta, ig_intr_md, ig_prsr_md, ig_dprsr_md, ig_tm_md);
        mod_2.apply(hdr, meta, ig_intr_md, ig_prsr_md, ig_dprsr_md, ig_tm_md);
        mod_3.apply(hdr, meta, ig_intr_md, ig_prsr_md, ig_dprsr_md, ig_tm_md);
        port_match.apply();
    }
}

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
        pkt.emit(hdr.report);
        pkt.emit(hdr.probe);
        if (meta.do_clone == 1) {
            mirror.emit(MIRROR_SESSION_ID);
        }
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
    Module_1_3_Egress() mod_1_3;
    apply {
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
        pkt.emit(hdr.report);
        pkt.emit(hdr.probe);
    }
}

Pipeline(
    IngressParser(),
    Ingress(),
    IngressDeparser(),
    EgressParser(),
    Egress(),
    EgressDeparser()) pipe;
Switch(pipe) main;
