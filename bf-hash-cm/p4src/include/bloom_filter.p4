/* -*- P4_16 -*- */

/* Bloom Filter */
const int FILTER_BUCKET_LENGTH_WIDTH = 18;
typedef bit<(FILTER_BUCKET_LENGTH_WIDTH)> FILTER_BUCKET_LENGTH_t;
const int FILTER_BUCKET_LENGTH = 1 << (FILTER_BUCKET_LENGTH_WIDTH);


const int FILTER_COUNT_BIT_WIDTH = 1;
typedef bit<(FILTER_COUNT_BIT_WIDTH)> FILTER_COUNT_BIT_WIDTH_t;

control bloom_filter(
    inout    my_ingress_metadata_t  meta)
{
    // bloom filter
    Register<FILTER_COUNT_BIT_WIDTH_t, FILTER_BUCKET_LENGTH_t>(FILTER_BUCKET_LENGTH,0) bf_count1;
    Register<FILTER_COUNT_BIT_WIDTH_t, FILTER_BUCKET_LENGTH_t>(FILTER_BUCKET_LENGTH,0) bf_count2;
    Register<FILTER_COUNT_BIT_WIDTH_t, FILTER_BUCKET_LENGTH_t>(FILTER_BUCKET_LENGTH,0) bf_count3;

    RegisterAction<FILTER_COUNT_BIT_WIDTH_t, FILTER_BUCKET_LENGTH_t, FILTER_COUNT_BIT_WIDTH_t>(bf_count1)
    get_bf_1 = {
        void apply(inout FILTER_COUNT_BIT_WIDTH_t register_data, out FILTER_COUNT_BIT_WIDTH_t result) {
            result = register_data;
            register_data = 1;
        }
    };

    RegisterAction<FILTER_COUNT_BIT_WIDTH_t, FILTER_BUCKET_LENGTH_t, FILTER_COUNT_BIT_WIDTH_t>(bf_count2)
    get_bf_2 = {
        void apply(inout FILTER_COUNT_BIT_WIDTH_t register_data, out FILTER_COUNT_BIT_WIDTH_t result) {
            result = register_data;
            register_data = 1;
        }
    };

    RegisterAction<FILTER_COUNT_BIT_WIDTH_t, FILTER_BUCKET_LENGTH_t, FILTER_COUNT_BIT_WIDTH_t>(bf_count3)
    get_bf_3 = {
        void apply(inout FILTER_COUNT_BIT_WIDTH_t register_data, out FILTER_COUNT_BIT_WIDTH_t result) {
            result = register_data;
            register_data = 1;
        }
    };
   
    apply {
        meta.bloomfilter_flag1 = get_bf_1.execute(meta.index1[(FILTER_BUCKET_LENGTH_WIDTH - 1):0]);
        meta.bloomfilter_flag2 = meta.bloomfilter_flag1 & get_bf_2.execute(meta.index2[(FILTER_BUCKET_LENGTH_WIDTH - 1):0]);
        meta.bloomfilter_flag3 = meta.bloomfilter_flag2 & get_bf_3.execute(meta.index3[(FILTER_BUCKET_LENGTH_WIDTH - 1):0]);
    }
}