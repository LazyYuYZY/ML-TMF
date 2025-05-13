/* -*- P4_16 -*- */

/* Hash table */
const int HASH_BUCKET_LENGTH_WIDTH = 10;
typedef bit<(HASH_BUCKET_LENGTH_WIDTH)> HASH_BUCKET_LENGTH_t;
const int HASH_BUCKET_LENGTH = 1 << (HASH_BUCKET_LENGTH_WIDTH);


const int HASH_COUNT_BIT_WIDTH = 16;
typedef bit<(HASH_COUNT_BIT_WIDTH)> HASH_COUNT_BIT_WIDTH_t;


control hash_table(
    in       my_ingress_headers_t    hdr,
    inout    my_ingress_metadata_t   meta)
{

    Register<bit<32>, HASH_BUCKET_LENGTH_t>(HASH_BUCKET_LENGTH,0) hash_src_addr_1;
    RegisterAction<bit<32>, HASH_BUCKET_LENGTH_t, bit<32>>(hash_src_addr_1)
    set_hash_src_addr_1 = {
        void apply(inout bit<32> register_data) {
            register_data = hdr.ipv4.src_addr;
        }
    };

    Register<bit<32>, HASH_BUCKET_LENGTH_t>(HASH_BUCKET_LENGTH,0) hash_dst_addr_1;
    RegisterAction<bit<32>, HASH_BUCKET_LENGTH_t, bit<32>>(hash_dst_addr_1)
    set_hash_dst_addr_1 = {
        void apply(inout bit<32> register_data) {
            register_data = hdr.ipv4.dst_addr;
        }
    };

    Register<bit<8>, HASH_BUCKET_LENGTH_t>(HASH_BUCKET_LENGTH,0) hash_protocol_1;
    RegisterAction<bit<8>, HASH_BUCKET_LENGTH_t, bit<8>>(hash_protocol_1)
    set_hash_protocol_1 = {
        void apply(inout bit<8> register_data) {
            register_data = hdr.ipv4.protocol;
        }
    };

    Register<bit<16>, HASH_BUCKET_LENGTH_t>(HASH_BUCKET_LENGTH,0) hash_src_port_1;
    RegisterAction<bit<16>, HASH_BUCKET_LENGTH_t, bit<16>>(hash_src_port_1)
    set_hash_src_port_1 = {
        void apply(inout bit<16> register_data) {
            register_data = meta.l4_lookup.src_port;
        }
    };

    Register<bit<16>, HASH_BUCKET_LENGTH_t>(HASH_BUCKET_LENGTH,0) hash_dst_port_1;
    RegisterAction<bit<16>, HASH_BUCKET_LENGTH_t, bit<16>>(hash_dst_port_1)
    set_hash_dst_port_1 = {
        void apply(inout bit<16> register_data) {
            register_data = meta.l4_lookup.dst_port;
        }
    };


    Register<HASH_COUNT_BIT_WIDTH_t, HASH_BUCKET_LENGTH_t>(HASH_BUCKET_LENGTH,0) hash_count;
    RegisterAction<HASH_COUNT_BIT_WIDTH_t, HASH_BUCKET_LENGTH_t, HASH_COUNT_BIT_WIDTH_t>(hash_count)
    set_hash_count = {
        void apply(inout HASH_COUNT_BIT_WIDTH_t register_data) {
            register_data = register_data + 1;
        }
    };
   
    apply {
        set_hash_src_addr_1.execute(meta.index1[(HASH_BUCKET_LENGTH_WIDTH - 1):0]);
        set_hash_dst_addr_1.execute(meta.index1[(HASH_BUCKET_LENGTH_WIDTH - 1):0]);
        set_hash_protocol_1.execute(meta.index1[(HASH_BUCKET_LENGTH_WIDTH - 1):0]);
        set_hash_src_port_1.execute(meta.index1[(HASH_BUCKET_LENGTH_WIDTH - 1):0]);
        set_hash_dst_port_1.execute(meta.index1[(HASH_BUCKET_LENGTH_WIDTH - 1):0]);
        set_hash_count.execute(meta.index1[(HASH_BUCKET_LENGTH_WIDTH - 1):0]);
    }
}