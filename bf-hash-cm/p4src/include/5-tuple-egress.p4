/* -*- P4_16 -*- */

control save_5_tuple_egress(
    inout    my_egress_headers_t   hdr,
    inout    my_egress_metadata_t  meta)
{
    // 第一组
    Register<bit<32>, bit<1>>(1,0) src_addr_1;
    RegisterAction<bit<32>, bit<1>, bit<32>>(src_addr_1)
    set_src_addr_1 = {
        void apply(inout bit<32> register_data, out bit<32> result) {
            register_data = hdr.ipv4.src_addr;
            result = register_data;
        }
    };

    Register<bit<32>, bit<1>>(1,0) dst_addr_1;
    RegisterAction<bit<32>, bit<1>, bit<32>>(dst_addr_1)
    set_dst_addr_1 = {
        void apply(inout bit<32> register_data, out bit<32> result) {
            register_data = hdr.ipv4.dst_addr;
            result = register_data;
        }
    };

    Register<bit<8>, bit<1>>(1,0) protocol_1;
    RegisterAction<bit<8>, bit<1>, bit<8>>(protocol_1)
    set_protocol_1 = {
        void apply(inout bit<8> register_data, out bit<8> result) {
            register_data = hdr.ipv4.protocol;
            result = register_data;
        }
    };

    Register<bit<16>, bit<1>>(1,0) src_port_1;
    RegisterAction<bit<16>, bit<1>, bit<16>>(src_port_1)
    set_src_port_1 = {
        void apply(inout bit<16> register_data, out bit<16> result) {
            register_data = meta.l4_lookup.src_port;
            result = register_data;
        }
    };

    Register<bit<16>, bit<1>>(1,0) dst_port_1;
    RegisterAction<bit<16>, bit<1>, bit<16>>(dst_port_1)
    set_dst_port_1 = {
        void apply(inout bit<16> register_data, out bit<16> result) {
            register_data = meta.l4_lookup.dst_port;
            result = register_data;
        }
    };

    // 第二组
    Register<bit<32>, bit<1>>(1,0) src_addr_2;
    RegisterAction<bit<32>, bit<1>, bit<32>>(src_addr_2)
    set_src_addr_2 = {
        void apply(inout bit<32> register_data, out bit<32> result) {
            register_data = hdr.ipv4.src_addr;
            result = register_data;
        }
    };

    Register<bit<32>, bit<1>>(1,0) dst_addr_2;
    RegisterAction<bit<32>, bit<1>, bit<32>>(dst_addr_2)
    set_dst_addr_2 = {
        void apply(inout bit<32> register_data, out bit<32> result) {
            register_data = hdr.ipv4.dst_addr;
            result = register_data;
        }
    };

    Register<bit<8>, bit<1>>(1,0) protocol_2;
    RegisterAction<bit<8>, bit<1>, bit<8>>(protocol_2)
    set_protocol_2 = {
        void apply(inout bit<8> register_data, out bit<8> result) {
            register_data = hdr.ipv4.protocol;
            result = register_data;
        }
    };

    Register<bit<16>, bit<1>>(1,0) src_port_2;
    RegisterAction<bit<16>, bit<1>, bit<16>>(src_port_2)
    set_src_port_2 = {
        void apply(inout bit<16> register_data, out bit<16> result) {
            register_data = meta.l4_lookup.src_port;
            result = register_data;
        }
    };

    Register<bit<16>, bit<1>>(1,0) dst_port_2;
    RegisterAction<bit<16>, bit<1>, bit<16>>(dst_port_2)
    set_dst_port_2 = {
        void apply(inout bit<16> register_data, out bit<16> result) {
            register_data = meta.l4_lookup.dst_port;
            result = register_data;
        }
    };

    // 第三组
    Register<bit<32>, bit<1>>(1,0) src_addr_3;
    RegisterAction<bit<32>, bit<1>, bit<32>>(src_addr_3)
    set_src_addr_3 = {
        void apply(inout bit<32> register_data, out bit<32> result) {
            register_data = hdr.ipv4.src_addr;
            result = register_data;
        }
    };

    Register<bit<32>, bit<1>>(1,0) dst_addr_3;
    RegisterAction<bit<32>, bit<1>, bit<32>>(dst_addr_3)
    set_dst_addr_3 = {
        void apply(inout bit<32> register_data, out bit<32> result) {
            register_data = hdr.ipv4.dst_addr;
            result = register_data;
        }
    };

    Register<bit<8>, bit<1>>(1,0) protocol_3;
    RegisterAction<bit<8>, bit<1>, bit<8>>(protocol_3)
    set_protocol_3 = {
        void apply(inout bit<8> register_data, out bit<8> result) {
            register_data = hdr.ipv4.protocol;
            result = register_data;
        }
    };

    Register<bit<16>, bit<1>>(1,0) src_port_3;
    RegisterAction<bit<16>, bit<1>, bit<16>>(src_port_3)
    set_src_port_3 = {
        void apply(inout bit<16> register_data, out bit<16> result) {
            register_data = meta.l4_lookup.src_port;
            result = register_data;
        }
    };

    Register<bit<16>, bit<1>>(1,0) dst_port_3;
    RegisterAction<bit<16>, bit<1>, bit<16>>(dst_port_3)
    set_dst_port_3 = {
        void apply(inout bit<16> register_data, out bit<16> result) {
            register_data = meta.l4_lookup.dst_port;
            result = register_data;
        }
    };

    // 第四组
    Register<bit<32>, bit<1>>(1,0) src_addr_4;
    RegisterAction<bit<32>, bit<1>, bit<32>>(src_addr_4)
    set_src_addr_4 = {
        void apply(inout bit<32> register_data, out bit<32> result) {
            register_data = hdr.ipv4.src_addr;
            result = register_data;
        }
    };

    Register<bit<32>, bit<1>>(1,0) dst_addr_4;
    RegisterAction<bit<32>, bit<1>, bit<32>>(dst_addr_4)
    set_dst_addr_4 = {
        void apply(inout bit<32> register_data, out bit<32> result) {
            register_data = hdr.ipv4.dst_addr;
            result = register_data;
        }
    };

    Register<bit<8>, bit<1>>(1,0) protocol_4;
    RegisterAction<bit<8>, bit<1>, bit<8>>(protocol_4)
    set_protocol_4 = {
        void apply(inout bit<8> register_data, out bit<8> result) {
            register_data = hdr.ipv4.protocol;
            result = register_data;
        }
    };

    Register<bit<16>, bit<1>>(1,0) src_port_4;
    RegisterAction<bit<16>, bit<1>, bit<16>>(src_port_4)
    set_src_port_4 = {
        void apply(inout bit<16> register_data, out bit<16> result) {
            register_data = meta.l4_lookup.src_port;
            result = register_data;
        }
    };

    Register<bit<16>, bit<1>>(1,0) dst_port_4;
    RegisterAction<bit<16>, bit<1>, bit<16>>(dst_port_4)
    set_dst_port_4 = {
        void apply(inout bit<16> register_data, out bit<16> result) {
            register_data = meta.l4_lookup.dst_port;
            result = register_data;
        }
    };

    action send_to_cpu() {
        hdr.cpu_ethernet.setValid();
        hdr.cpu_ethernet.dst_addr   = 0xFFFFFFFFFFFF;
        hdr.cpu_ethernet.src_addr   = 0xAAAAAAAAAAAA;
        hdr.cpu_ethernet.ether_type = ETHERTYPE_TO_CPU;
    }
   
    apply {
        if(meta.count == 0 || meta.count == 3){
            meta.src_addr_1 = set_src_addr_1.execute(0);
            meta.dst_addr_1 = set_dst_addr_1.execute(0);
            meta.protocol_1 = set_protocol_1.execute(0);
            meta.src_port_1 = set_src_port_1.execute(0);
            meta.dst_port_1 = set_dst_port_1.execute(0);
        }
        if(meta.count == 1 || meta.count == 3){
            meta.src_addr_2 = set_src_addr_2.execute(0);
            meta.dst_addr_2 = set_dst_addr_2.execute(0);
            meta.protocol_2 = set_protocol_2.execute(0);
            meta.src_port_2 = set_src_port_2.execute(0);
            meta.dst_port_2 = set_dst_port_2.execute(0);
        }
        if(meta.count == 2 || meta.count == 3){
            meta.src_addr_3 = set_src_addr_3.execute(0);
            meta.dst_addr_3 = set_dst_addr_3.execute(0);
            meta.protocol_3 = set_protocol_3.execute(0);
            meta.src_port_3 = set_src_port_3.execute(0);
            meta.dst_port_3 = set_dst_port_3.execute(0);
        }
        if(meta.count == 3 && meta.ing_port_mirror.mirror_session == 1){
            hdr.to_cpu.setValid();
            send_to_cpu();
            hdr.to_cpu.src_addr_1 = meta.src_addr_1;
            hdr.to_cpu.dst_addr_1 = meta.dst_addr_1;
            hdr.to_cpu.protocol_1 = meta.protocol_1;
            hdr.to_cpu.src_port_1 = meta.src_port_1;
            hdr.to_cpu.dst_port_1 = meta.dst_port_1;

            hdr.to_cpu.src_addr_2 = meta.src_addr_2;
            hdr.to_cpu.dst_addr_2 = meta.dst_addr_2;
            hdr.to_cpu.protocol_2 = meta.protocol_2;
            hdr.to_cpu.src_port_2 = meta.src_port_2;
            hdr.to_cpu.dst_port_2 = meta.dst_port_2;

            hdr.to_cpu.src_addr_3 = meta.src_addr_3;
            hdr.to_cpu.dst_addr_3 = meta.dst_addr_3;
            hdr.to_cpu.protocol_3 = meta.protocol_3;
            hdr.to_cpu.src_port_3 = meta.src_port_3;
            hdr.to_cpu.dst_port_3 = meta.dst_port_3;

            hdr.to_cpu.src_addr_4 = set_src_addr_4.execute(0);
            hdr.to_cpu.dst_addr_4 = set_dst_addr_4.execute(0);
            hdr.to_cpu.protocol_4 = set_protocol_4.execute(0);
            hdr.to_cpu.src_port_4 = set_src_port_4.execute(0);
            hdr.to_cpu.dst_port_4 = set_dst_port_4.execute(0);
        }
    }
}