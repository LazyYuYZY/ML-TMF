/* -*- P4_16 -*- */
// =========================================================================
// 模块 1.3：业务与遥测绝对隔离 (Egress 逻辑)
// 克隆包改写端口发往收集器，原始业务包剥离遥测头部保证用户无感
// =========================================================================
control Module_1_3_Egress(
    inout headers_t hdr,
    inout my_egress_metadata_t meta,
    in    egress_intrinsic_metadata_t eg_intr_md,
    in    egress_intrinsic_metadata_from_parser_t eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t eg_port_md) {

    apply {
        if (eg_intr_md.egress_rid != 0) {
            // 这是一个由 1.2 满载触发的克隆遥测包
            if (hdr.report.isValid()) {
                // 修改发往收集服务器的专有端口 (9998)
                hdr.udp.dst_port = REPORT_PORT;
                // 必须置零 Checksum，否则在服务器侧可能被网卡静默丢弃
                hdr.udp.checksum = 0;
            }
        } else {
            // 这是一个原始业务包，强行剥离上报头部，保证用户无感
            hdr.report.setInvalid();
        }
    }
}
