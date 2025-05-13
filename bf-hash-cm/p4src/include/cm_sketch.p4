/* -*- P4_16 -*- */

/* CM Sketch */
const int SKETCH_BUCKET_LENGTH_WIDTH = 18;
typedef bit<(SKETCH_BUCKET_LENGTH_WIDTH)> SKETCH_BUCKET_LENGTH_t;
const int SKETCH_BUCKET_LENGTH = 1 << (SKETCH_BUCKET_LENGTH_WIDTH);


const int SKETCH_COUNT_BIT_WIDTH = 16;
typedef bit<(SKETCH_COUNT_BIT_WIDTH)> SKETCH_COUNT_BIT_WIDTH_t;


control cm_sketch(
    in  my_ingress_metadata_t   meta)
{

    Register<SKETCH_COUNT_BIT_WIDTH_t, SKETCH_BUCKET_LENGTH_t>(SKETCH_BUCKET_LENGTH,0) sketch_count1;

    RegisterAction<SKETCH_COUNT_BIT_WIDTH_t, SKETCH_BUCKET_LENGTH_t, SKETCH_COUNT_BIT_WIDTH_t>(sketch_count1)
    leave_count1 = {
        void apply(inout SKETCH_COUNT_BIT_WIDTH_t register_data) {
            register_data = register_data + 1;
        }
    };


    Register<SKETCH_COUNT_BIT_WIDTH_t, SKETCH_BUCKET_LENGTH_t>(SKETCH_BUCKET_LENGTH,0) sketch_count2;

    RegisterAction<SKETCH_COUNT_BIT_WIDTH_t, SKETCH_BUCKET_LENGTH_t, SKETCH_COUNT_BIT_WIDTH_t>(sketch_count2)
    leave_count2 = {
        void apply(inout SKETCH_COUNT_BIT_WIDTH_t register_data) {
            register_data = register_data + 1;
        }
    };



    Register<SKETCH_COUNT_BIT_WIDTH_t, SKETCH_BUCKET_LENGTH_t>(SKETCH_BUCKET_LENGTH,0) sketch_count3;

    RegisterAction<SKETCH_COUNT_BIT_WIDTH_t, SKETCH_BUCKET_LENGTH_t, SKETCH_COUNT_BIT_WIDTH_t>(sketch_count3)
    leave_count3 = {
        void apply(inout SKETCH_COUNT_BIT_WIDTH_t register_data) {
            register_data = register_data + 1;
        }
    };

   
    apply {
        leave_count1.execute(meta.index1[(SKETCH_BUCKET_LENGTH_WIDTH - 1):0]);
        leave_count2.execute(meta.index2[(SKETCH_BUCKET_LENGTH_WIDTH - 1):0]);
        leave_count3.execute(meta.index3[(SKETCH_BUCKET_LENGTH_WIDTH - 1):0]);

    }
}