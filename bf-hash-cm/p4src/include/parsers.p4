/* -*- P4_16 -*- */

/*************************************************************************
 **************  I N G R E S S   P R O C E S S I N G   *******************
 *************************************************************************/

    /***********************  P A R S E R  **************************/
parser IngressParser(packet_in        pkt,
    /* User */    
    out my_ingress_headers_t          hdr,
    out my_ingress_metadata_t         meta,
    /* Intrinsic */
    out ingress_intrinsic_metadata_t  ig_intr_md)
{
    /* This is a mandatory state, required by Tofino Architecture */
     state start {
        pkt.extract(ig_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        transition meta_init;
    }

    state meta_init {
        meta.timestamp         = 0;

        meta.l4_lookup         = { 0, 0 };

        meta.index1     = 0;
        meta.index2     = 0;
        meta.index3     = 0;

        meta.bloomfilter_flag1 = 0;
        meta.bloomfilter_flag2 = 0;
        meta.bloomfilter_flag3 = 0;

        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            (bit<16>)ether_type_t.IPV4 :  parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        meta.l4_lookup = pkt.lookahead<l4_lookup_t>();
        transition select(hdr.ipv4.frag_offset, hdr.ipv4.protocol) {
            ( 0, ip_proto_t.TCP  ) : parse_tcp;
            ( 0, ip_proto_t.UDP  ) : parse_udp;
            default : accept;
        }
    }

    state parse_tcp {
        pkt.extract(hdr.tcp);
        transition accept;
    }
    
    state parse_udp {
        pkt.extract(hdr.udp);
        transition accept;
    }

}

    /*********************  D E P A R S E R  ************************/

control IngressDeparser(packet_out pkt,
    /* User */
    inout my_ingress_headers_t                       hdr,
    in    my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md)
{
    Mirror() ing_port_mirror;

    apply {
        if (ig_dprsr_md.mirror_type == ING_PORT_MIRROR) {
            ing_port_mirror.emit<ing_port_mirror_h>(
                meta.mirror_session,
                {
                    meta.mirror_header_type, meta.mirror_header_info,
                    0,
                    meta.ingress_port,
                    0,
                    meta.mirror_session
                });
        }
        pkt.emit(hdr);
    }
}


/*************************************************************************
 ****************  E G R E S S   P R O C E S S I N G   *******************
 *************************************************************************/

    /***********************  P A R S E R  **************************/

parser EgressParser(packet_in        pkt,
    /* User */
    out my_egress_headers_t          hdr,
    out my_egress_metadata_t         meta,
    /* Intrinsic */
    out egress_intrinsic_metadata_t  eg_intr_md)
{
    inthdr_h inthdr;

    /* This is a mandatory state, required by Tofino Architecture */
    state start {
        pkt.extract(eg_intr_md);
        transition meta_init;
    }

    state meta_init {
        meta.src_addr_1 = 0;
        meta.dst_addr_1 = 0;
        meta.protocol_1 = 0;
        meta.src_port_1 = 0;
        meta.dst_port_1 = 0;
        meta.src_addr_2 = 0;
        meta.dst_addr_2 = 0;
        meta.protocol_2 = 0;
        meta.src_port_2 = 0;
        meta.dst_port_2 = 0;
        meta.src_addr_3 = 0;
        meta.dst_addr_3 = 0;
        meta.protocol_3 = 0;
        meta.src_port_3 = 0;
        meta.dst_port_3 = 0;
        meta.src_addr_4 = 0;
        meta.dst_addr_4 = 0;
        meta.protocol_4 = 0;
        meta.src_port_4 = 0;
        meta.dst_port_4 = 0;

        transition parse_mirror;
    }

    state parse_mirror{
        pkt.extract(meta.ing_port_mirror);
        transition accept;
    }
}

    /*********************  D E P A R S E R  ************************/

control EgressDeparser(packet_out pkt,
    /* User */
    inout my_egress_headers_t                       hdr,
    in    my_egress_metadata_t                      meta,
    /* Intrinsic */
    in    egress_intrinsic_metadata_for_deparser_t  eg_dprsr_md)
{
    apply {
        pkt.emit(hdr);
    }
}