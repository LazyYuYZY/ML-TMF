/* -*- P4_16 -*- */

/*************************************************************************
 ************* C O N S T A N T S    A N D   T Y P E S  *******************
**************************************************************************/
const bit<16> ETHERTYPE_IPV4 = 0x0800;
const bit<16> ETHERTYPE_MPLS = 0x8847;
const bit<16> ETHERTYPE_TO_CPU = 0xBF01;

/* Table Sizes */
const int SEND_TO_OVS_SIZE = 16;
const int PORT_MATCH_SIZE  = 64;

enum bit<16> ether_type_t {
    IPV4 = 0x0800,
    MPLS = 0x8847
}

enum bit<8>  ip_proto_t {
    TCP = 6,
    UDP = 17
}

/*************************************************************************
 ***********************  H E A D E R S  *********************************
 *************************************************************************/

/*  Define all the headers the program will recognize             */
/*  The actual sets of headers processed by each gress can differ */
header ethernet_h {
    bit<48>   dst_addr;
    bit<48>   src_addr;
    bit<16>   ether_type;
}

header ipv4_h {
    bit<4>   version;
    bit<4>   ihl;
    bit<8>   diffserv;
    bit<16>  total_len;
    bit<16>  identification;
    bit<3>   flags;
    bit<13>  frag_offset;
    bit<8>   ttl;
    ip_proto_t   protocol;
    bit<16>  hdr_checksum;
    bit<32>  src_addr;
    bit<32>  dst_addr;
}

header mpls_h {
    bit<20> label;
    bit<3> exp;
    bit<1> bos;
    bit<8> ttl;
}

header tcp_h {
    bit<16>  src_port;
    bit<16>  dst_port;
    bit<32>  seq_no;
    bit<32>  ack_no;
    bit<4>   data_offset;
    bit<4>   res;
    bit<8>   flags;
    bit<16>  window;
    bit<16>  checksum;
    bit<16>  urgent_ptr;
}

header udp_h {
    bit<16>  src_port;
    bit<16>  dst_port;
    bit<16>  len;
    bit<16>  checksum;
}

const bit<3> ING_PORT_MIRROR = 0; 

/*** Internal Headers ***/

typedef bit<4> header_type_t; 
typedef bit<4> header_info_t; 

const header_type_t HEADER_TYPE_BRIDGE         = 0xB;
const header_type_t HEADER_TYPE_MIRROR_INGRESS = 0xC;
const header_type_t HEADER_TYPE_MIRROR_EGRESS  = 0xD;
const header_type_t HEADER_TYPE_RESUBMIT       = 0xA;

#define INTERNAL_HEADER         \
    header_type_t header_type;  \
    header_info_t header_info

header inthdr_h {
    INTERNAL_HEADER;
}

header ing_port_mirror_h {
    INTERNAL_HEADER;
    bit<7> pad0;  PortId_t    ingress_port;
    bit<6> pad1;  MirrorId_t  mirror_session;
}

header to_cpu_h {
    INTERNAL_HEADER;
    bit<6>    pad0; MirrorId_t  mirror_session;
    bit<7>    pad1; PortId_t    ingress_port;

    bit<32> src_addr_1;
    bit<32> dst_addr_1;
    bit<8> protocol_1;
    bit<16> src_port_1;
    bit<16> dst_port_1;
    bit<32> src_addr_2;
    bit<32> dst_addr_2;
    bit<8> protocol_2;
    bit<16> src_port_2;
    bit<16> dst_port_2;
    bit<32> src_addr_3;
    bit<32> dst_addr_3;
    bit<8> protocol_3;
    bit<16> src_port_3;
    bit<16> dst_port_3;
    bit<32> src_addr_4;
    bit<32> dst_addr_4;
    bit<8> protocol_4;
    bit<16> src_port_4;
    bit<16> dst_port_4;
}

// header my_protocol_h {
//     bit<32> src_addr_1;
//     bit<32> dst_addr_1;
//     bit<8> protocol_1;
//     bit<16> src_port_1;
//     bit<16> dst_port_1;
//     bit<32> src_addr_2;
//     bit<32> dst_addr_2;
//     bit<8> protocol_2;
//     bit<16> src_port_2;
//     bit<16> dst_port_2;
//     bit<32> src_addr_3;
//     bit<32> dst_addr_3;
//     bit<8> protocol_3;
//     bit<16> src_port_3;
//     bit<16> dst_port_3;
//     bit<32> src_addr_4;
//     bit<32> dst_addr_4;
//     bit<8> protocol_4;
//     bit<16> src_port_4;
//     bit<16> dst_port_4;
// }

/*************************************************************************
 **************  I N G R E S S   P R O C E S S I N G   *******************
 *************************************************************************/
 
    /***********************  H E A D E R S  ************************/

struct my_ingress_headers_t {
    ethernet_h     ethernet;
    ipv4_h         ipv4;
    tcp_h          tcp;
    udp_h          udp;
}

struct l4_lookup_t {
    bit<16>  src_port;
    bit<16>  dst_port;
}

/******  G L O B A L   I N G R E S S   M E T A D A T A  *********/

struct my_ingress_metadata_t {
    header_type_t  mirror_header_type;
    header_info_t  mirror_header_info;
    PortId_t       ingress_port;
    MirrorId_t     mirror_session;
    
    bit<48> timestamp;

    l4_lookup_t   l4_lookup;

    bit<32> index1;
    bit<32> index2;
    bit<32> index3;

    bit<1>   bloomfilter_flag1;
    bit<1>   bloomfilter_flag2;
    bit<1>   bloomfilter_flag3;
}

/*************************************************************************
 ****************  E G R E S S   P R O C E S S I N G   *******************
 *************************************************************************/

    /***********************  H E A D E R S  ************************/

struct my_egress_headers_t {
    ethernet_h     ethernet;
    ipv4_h         ipv4;
    tcp_h          tcp;
    udp_h          udp;
    // my_protocol_h  my_protocol;
    ethernet_h   cpu_ethernet;
    to_cpu_h     to_cpu;
}

    /********  G L O B A L   E G R E S S   M E T A D A T A  *********/

struct my_egress_metadata_t {
    ing_port_mirror_h  ing_port_mirror;

    l4_lookup_t   l4_lookup;

    bit<8>   count;

    bit<32>  src_addr_1;
    bit<32>  dst_addr_1;
    bit<8>   protocol_1;
    bit<16>  src_port_1;
    bit<16>  dst_port_1;
    bit<32>  src_addr_2;
    bit<32>  dst_addr_2;
    bit<8>   protocol_2;
    bit<16>  src_port_2;
    bit<16>  dst_port_2;
    bit<32>  src_addr_3;
    bit<32>  dst_addr_3;
    bit<8>   protocol_3;
    bit<16>  src_port_3;
    bit<16>  dst_port_3;
    bit<32>  src_addr_4;
    bit<32>  dst_addr_4;
    bit<8>   protocol_4;
    bit<16>  src_port_4;
    bit<16>  dst_port_4;
}