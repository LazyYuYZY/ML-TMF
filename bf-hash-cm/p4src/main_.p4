/* -*- P4_16 -*- */

#include <core.p4>
#include <tna.p4>

#include "include/headers.p4"
#include "include/parsers.p4"
#include "include/hash.p4"
#include "include/bloom_filter.p4"
#include "include/cm_sketch.p4"
#include "include/hash_table.p4"
#include "include/5-tuple-egress.p4"

/*************************************************************************
 **************  I N G R E S S   P R O C E S S I N G   *******************
 *************************************************************************/

    /***************** M A T C H - A C T I O N  *********************/

control Ingress(
    /* User */
    inout my_ingress_headers_t                       hdr,
    inout my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_t               ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t   ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t        ig_tm_md)
{

    action send(PortId_t port) {
        ig_tm_md.ucast_egress_port = port;
    }

    action drop() {
        ig_dprsr_md.drop_ctl = 1;
    }

    action gettimestamp() {
        meta.timestamp = ig_intr_md.ingress_mac_tstamp;
    }

    table port_match {
        key = {
            ig_intr_md.ingress_port: exact;
        }
        actions = {
            send;
        }
        size = PORT_MATCH_SIZE;
    }

    action acl_mirror() {
        ig_dprsr_md.mirror_type = ING_PORT_MIRROR;

        meta.mirror_header_type = HEADER_TYPE_MIRROR_INGRESS;
        meta.mirror_header_info = (header_info_t)ING_PORT_MIRROR;

        meta.ingress_port   = ig_intr_md.ingress_port;
        meta.mirror_session = 1;
    }

    hash() hash;
    cm_sketch() cm_sketch;
    hash_table() hash_table;
    bloom_filter() bloom_filter;

    apply{
        if(hdr.ipv4.isValid() && (hdr.tcp.isValid() || hdr.udp.isValid())){
            hash.apply(hdr,meta);
            cm_sketch.apply(meta);
            hash_table.apply(hdr,meta);
            bloom_filter.apply(meta);
            if(meta.bloomfilter_flag3 == 0){
                acl_mirror();
            }
            send(180);
        }
    }
}

/*************************************************************************
 ****************  E G R E S S   P R O C E S S I N G   *******************
 *************************************************************************/

    /***************** M A T C H - A C T I O N  *********************/

control Egress(
    /* User */
    inout my_egress_headers_t                          hdr_egress,
    inout my_egress_metadata_t                         meta_egress,
    /* Intrinsic */    
    in    egress_intrinsic_metadata_t                  eg_intr_md,
    in    egress_intrinsic_metadata_from_parser_t      eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t     eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t  eg_oport_md)
{
    Register<bit<8>, bit<1>>(1,0) counter;
    RegisterAction<bit<8>, bit<1>, bit<8>>(counter)
    set_count = {
        void apply(inout bit<8> register_data, out bit<8> result) {
            result = register_data;
            if(register_data == 7){
                register_data = 0;
            }else{
                register_data = register_data |+| 1;
            }
        }
    };

    

    save_5_tuple_egress() save_5_tuple_egress;

    apply {
        meta_egress.count = set_count.execute(0);
        save_5_tuple_egress.apply(hdr_egress,meta_egress);
    }
}

/************ F I N A L   P A C K A G E ******************************/
Pipeline(
    IngressParser(),
    Ingress(),
    IngressDeparser(),
    EgressParser(),
    Egress(),
    EgressDeparser()
) pipe;

Switch(pipe) main;