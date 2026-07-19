/* -*- P4_16 -*- */
#ifndef _TELEMETRY_DEFINES_P4_
#define _TELEMETRY_DEFINES_P4_
#include <core.p4>
#include <tna.p4>

// =========================================================================
// 1. 全局常量定义
// =========================================================================
const bit<16> ETHERTYPE_IPV4 = 0x0800;
const bit<8>  IP_PROTO_UDP   = 17;

// 遥测与探针专用端口与配置
const bit<16> REPORT_PORT       = 9998; // 模块 1.3: 上报克隆包目的端口
const bit<16> PROBE_PORT        = 9999; // 模块 2 & 3: 拉取数据的探针目的端口
const bit<10> MIRROR_SESSION_ID = 1;    // 模块 1.3: 镜像 Session ID

// =========================================================================
// 2. 报文头部定义
// =========================================================================
// --- 标准网络头部 ---
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

// --- 模块一自定义：新流批量上报头部 (V3.0: 4行，每行存源/目IP) ---
header report_t {
    bit<32> src_0; bit<32> dst_0;
    bit<32> src_1; bit<32> dst_1;
    bit<32> src_2; bit<32> dst_2;
    bit<32> src_3; bit<32> dst_3;
}

// --- 模块二/三自定义：数据拉取探针头部 ---
header probe_t {
    bit<32> seq_id;
    bit<16> sketch_1_val;
    bit<16> sketch_2_val;
    bit<16> sketch_3_val;
    bit<32> train_size;
    bit<32> train_key;      // 32 位完整哈希 Key
}

// 全局 Header 结构体封装
struct headers_t {
    ethernet_h ethernet;
    ipv4_h     ipv4;
    udp_h      udp;
    report_t   report;
    probe_t    probe;
}

// =========================================================================
// 3. 全局元数据总线
// =========================================================================
struct my_ingress_metadata_t {
    // --------------------------------------------------
    // 【总线控制信号】(跨模块通信)
    // --------------------------------------------------
    bit<1>  is_new_flow;      // [模块 1.1 -> 1.2] 标记当前流是否为新流
    bit<1>  do_clone;         // [模块 1.2 -> 1.3] 缓存满载，触发上报克隆信号
    bit<1>  is_probe;         // [模块 2 -> 3] 标记是否为主动探针包
    bit<1>  is_test_pkt;      // [测试标记]

    // --------------------------------------------------
    // 【模块 1.1：布隆过滤器】(V3.0: 哈希输入改为源目IP)
    // --------------------------------------------------
    bit<32> h1_raw;           // 哈希1原值
    bit<32> h2_raw;           // 哈希2原值
    bit<32> h3_raw;           // 哈希3原值
    bit<16> bf_idx_1;         // 哈希1截取的16位物理索引
    bit<16> bf_idx_2;         // 哈希2截取的16位物理索引
    bit<16> bf_idx_3;         // 哈希3截取的16位物理索引
    bit<1>  bf_out_1;         // 布隆过滤器1读出值
    bit<1>  bf_out_2;         // 布隆过滤器2读出值
    bit<1>  bf_out_3;         // 布隆过滤器3读出值

    // --------------------------------------------------
    // 【模块 1.2：IP缓存】(V3.0: 4行IP对)
    // --------------------------------------------------
    bit<8>  current_cnt;      // 环形计数器当前指针 (0~3)
    bit<3>  cache_ctrl;       // 位宽折叠控制字 = {is_new_flow, current_cnt[1:0]}
    bit<32> prep_src;         // 预处理暂存：源IP
    bit<32> prep_dst;         // 预处理暂存：目的IP

    // --------------------------------------------------
    // 【模块 2：Count-Min Sketch】
    // --------------------------------------------------
    bit<14> s1_idx;
    bit<14> s2_idx;
    bit<14> s3_idx;
    bit<16> s1_out;
    bit<16> s2_out;
    bit<16> s3_out;

    // --------------------------------------------------
    // 【模块 3：训练数据收集】
    // --------------------------------------------------
    bit<32> flow_key;         // 流的全局唯一标志 (32位完整哈希)
    bit<10> train_idx;        // 1024行精确表的物理寻址索引
    bit<1>  train_key_match;
    bit<32> train_key_out;
    bit<32> train_size_out;
}

struct my_egress_metadata_t {
    bit<8> dummy;
}

#endif // _TELEMETRY_DEFINES_P4_
