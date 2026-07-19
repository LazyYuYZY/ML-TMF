/* -*- P4_16 -*- */
#include <core.p4>
#include <tna.p4>

const bit<16> ETHERTYPE_IPV4 = 0x0800;
const bit<8>  IP_PROTO_UDP   = 17;
const bit<16> REPORT_PORT    = 9998;

// 克隆镜像 Session ID（需在控制面 setup.py 中同步配置 mirror session 1）
const bit<10> MIRROR_SESSION_ID = 1;

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

// 模块 1.3 核心处理对象：新流批量上报头部 (适配 1.2 模块的 2 个槽位五元组拆分块)
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

// 全局元数据总线（通信接口）
struct my_ingress_metadata_t {
    // 【输入接口】监听来自 1.2 模块的信号
    bit<1>  do_clone;                // 模拟测试用的标记
    bit<1>  is_test_pkt;
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
        meta.do_clone = 0;
        meta.is_test_pkt = 0;
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
    
    // 实例化硬件镜像引擎
    Mirror() mirror; 

    apply {
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.udp);
        
        // 注意：这里正常发射可能被 1.2 模块填充过数据的 report 头部
        pkt.emit(hdr.report); 

        // 模块 1.3 核心：按需克隆
        // 监听总线，当 1.2 模块判定槽位满载时，触发克隆到 Egress
        if (meta.do_clone == 1) {
            mirror.emit(MIRROR_SESSION_ID);
        }
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
    
    // --- 模拟 1.2 模块的控制流 ---
    // 为了让 1.3 可以独立编译并打流测试，我们写一个简单的 Mock 逻辑：
    // 如果报文是特定的 UDP 包（例如长度为指定值，或源端口为特定值），我们假装它触发了克隆。
    action mock_clone_trigger() {
        meta.do_clone = 1;
        
        // 假装 1.2 模块已经把提取的五元组填入了上报包头
        hdr.report.setValid();
        
        // Slot 0
        hdr.report.src_addr_0 = hdr.ipv4.src_addr;
        hdr.report.dst_addr_0 = hdr.ipv4.dst_addr;
        hdr.report.ports_0 = hdr.udp.src_port ++ hdr.udp.dst_port;
        hdr.report.proto_0 = (bit<32>)hdr.ipv4.protocol;
        
        // Slot 1
        hdr.report.src_addr_1 = hdr.ipv4.src_addr;
        hdr.report.dst_addr_1 = hdr.ipv4.dst_addr;
        hdr.report.ports_1 = hdr.udp.src_port ++ hdr.udp.dst_port;
        hdr.report.proto_1 = (bit<32>)hdr.ipv4.protocol;
    }
    
    table t_mock_trigger {
        actions = { mock_clone_trigger; }
        default_action = mock_clone_trigger();
        size = 1;
    }

    action send(bit<9> port) {
        ig_tm_md.ucast_egress_port = port;
    }
    
    table port_match {
        key = { ig_intr_md.ingress_port: exact; }
        actions = { send; }
        size = 64;
    }

    apply {
        if (hdr.ipv4.isValid() && hdr.udp.isValid()) {
            // 这里留好接口，后续替换为真实的 t_check_bf.apply() 和 t_route_cache_slot.apply()
            
            // 为了独立验证 1.3 模块，暂时开启 Mock 触发
            // 如果你想看正常包转发，可以把它用 if 包裹或者注释掉
            t_mock_trigger.apply(); 
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
        pkt.emit(hdr.report); // 只有 Valid 的 report 会被真正发往物理线缆
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
        // 模块 1.3 核心：业务流与遥测流隔离
        // 判断当前处理的包是原始包还是 Clone 过来的包
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

/*************************************************************************
 ************************ P I P E L I N E   M A I N **********************
 *************************************************************************/
Pipeline(
    IngressParser(), Ingress(), IngressDeparser(),
    EgressParser(), Egress(), EgressDeparser()
) pipe;

Switch(pipe) main;
