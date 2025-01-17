from sketchs import *

if __name__ == '__main__':
    # 是否有数据集
    dataset_over = 0

    # 是否采用已有cm参数
    flag=0
    cm_d=3
    w=1 * 10 ** 4
    flag_dict={1:"old",0:"new"}
    # memory_list=[1,2,4,8,12,16,20,24,32]
    memory_list = [32]
    # 获取流映射
    flow_index_path = "./testdata_set_5s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flow_index_path)

    for i in memory_list:
        cm_w = int(w * i)
        # 创建使用的 Sketch
        cm_sketch_now = np.full((cm_d, cm_w * (2 ** (cm_d - 1))), 0)  # cm存储的counter值

        sketch_path = "./sketch_params/" + str(cm_w).zfill(6) + "_cm_sketch.txt"
        if not os.path.exists(sketch_path):
            cm_used = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=0, dict_cm={},
                                      cm_sketch_load=cm_sketch_now)
            cm_used.save(file_name=sketch_path)
        else:
            dict_cm = rw_files.get_dict(sketch_path)
            cm_used = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm,
                                      cm_sketch_load=cm_sketch_now)
        # continue


        # 选择要输入的流的位置
        for file_now in range(10):
            # 读取流文件
            json_file_path = "./testdata_set_5s/testdata_flows_5s/" + str(file_now).zfill(5) + ".txt"
            flows_data = rw_files.get_dict(json_file_path)

            # 获取每个时刻的flows映射与数目
            flows_data_onetime = {}
            for one_flow in flows_data.keys():
                flows_data_onetime[flows_alltime_dict[one_flow]] = int(flows_data[one_flow])

            # 流通过cm sketch
            cm_used.insert_dict(flows_data_onetime)

            # 查询cm的counter值:(d,w)
            cm_sketch_now = cm_used.Matrix  # cm的counter值

            '''导出数据'''
            # 将矩阵保存为文本文件
            # 拆分拼接字符串 创建txt文件
            dict_list = json_file_path.split('testdata_flows_5s')
            cm_path = dict_list[0] + "testdata_"+str(cm_w).zfill(6) +"_cm_5s" + dict_list[1]
            folder_path=cm_path.split(dict_list[1])[0]

            # 判断文件夹是否存在
            if not os.path.exists(folder_path):
                print("文件夹不存在")
                os.mkdir(folder_path)

            np.savetxt(cm_path, cm_sketch_now, fmt='%d')

            # 查询所有流在cm sketch中的值
            cm_flows_query_d = cm_used.query_d(list(flows_data_onetime.keys()))
            # DNN dataset:每一行的查询值(n,d)
            dataset_x = cm_flows_query_d.T
            # 流的查询值，即对每行求min:(n,1)
            flows_query_cm = np.min(dataset_x, axis=1)
            # dataset_y即为flows_data的value:(n,1)
            dataset_y = np.array(list(flows_data_onetime.values())).T

            # #获取cm的相对误差
            cm_num_array = flows_query_cm.T
            real_num_array = np.array(list(flows_data_onetime.values()))
            absolute_cm_error = np.abs((real_num_array - cm_num_array))

            relative_cm_error = np.abs((real_num_array - cm_num_array) / real_num_array)
            print("max absolut error:", np.max(absolute_cm_error))
            print("absolut error:" + "{:.6f}".format(np.mean(absolute_cm_error)))
            print("relative error:" + "{:.6f}".format(np.mean(relative_cm_error)))

            cm_used.clear()
        dataset_over = 1
