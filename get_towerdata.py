from sketchs import *

if __name__ == '__main__':
    # 是否有数据集
    dataset_over = 0

    # 是否采用已有tower参数
    tower_d=3
    w=1 * 10 ** 4
    # memory_list=[1,2,4,8,12,16,20,24,32]
    memory_list = [32]
    # 获取流映射
    flow_index_path = "./testdata_set_5s/flow_index.txt"
    # flow_index_path = "./traindata_set_5s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flow_index_path)

    for i in memory_list:
        tower_w=int(w*i)
        # 创建使用的 Sketch
        tower_sketch_now=np.full((tower_d, tower_w*(2**(tower_d-1))), 0)  # tower存储的counter值

        sketch_path = "./sketch_params/" + str(tower_w).zfill(6) + "_tower_sketch.txt"
        if not os.path.exists(sketch_path):
            tower_used = tower_sketch(tower_d=tower_d, tower_w=tower_w, flag=0, dict_tower={},
                                      tower_sketch_load=tower_sketch_now)
            tower_used.save(file_name=sketch_path)
        else:
            dict_tower = rw_files.get_dict(sketch_path)
            tower_used = tower_sketch(tower_d=tower_d, tower_w=tower_w, flag=1, dict_tower=dict_tower,
                                      tower_sketch_load=tower_sketch_now)


        # 选择要输入的流的位置
        for file_now in range(10):
            # 读取流文件
            json_file_path = "./testdata_set_5s/testdata_flows_5s/" + str(file_now).zfill(5) + ".txt"
            # json_file_path = "./traindata_set_5s/traindata_flows_5s/" + str(file_now).zfill(5) + ".txt"
            flows_data = rw_files.get_dict(json_file_path)

            # 获取每个时刻的flows映射与数目
            flows_data_onetime = {}
            for one_flow in flows_data.keys():
                flows_data_onetime[flows_alltime_dict[one_flow]] = flows_data[one_flow]

            # 流通过tower sketch
            tower_used.insert_dict(flows_data_onetime)

            # 查询tower的counter值:(d,w)
            tower_sketch_now = tower_used.Matrix  # tower的counter值

            '''导出数据'''
            # 将矩阵保存为文本文件
            # 拆分拼接字符串 创建txt文件
            dict_list = json_file_path.split('testdata_flows_5s')
            # dict_list = json_file_path.split('traindata_flows_5s')
            tower_path = dict_list[0] + "testdata_"+str(tower_w).zfill(6) +"_tower_5s" + dict_list[1]
            # tower_path = dict_list[0] + "traindata_" + str(tower_w).zfill(6) + "_tower_5s" + dict_list[1]
            folder_path=tower_path.split(dict_list[1])[0]

            # 判断文件夹是否存在
            if not os.path.exists(folder_path):
                print("文件夹不存在")
                os.mkdir(folder_path)

            np.savetxt(tower_path, tower_sketch_now, fmt='%d')

            # 查询所有流在tower sketch中的值
            tower_flows_query_d = tower_used.query_d(list(flows_data_onetime.keys()))
            # DNN dataset:每一行的查询值(n,d)
            dataset_x = tower_flows_query_d.T
            # 流的查询值，即对每行求min:(n,1)
            flows_query_tower = np.min(dataset_x, axis=1)
            # dataset_y即为flows_data的value:(n,1)
            dataset_y = np.array(list(flows_data_onetime.values())).T

            # #获取tower的相对误差
            tower_num_array = flows_query_tower.T
            real_num_array = np.array(list(flows_data_onetime.values()))
            absolute_tower_error = np.abs((real_num_array - tower_num_array))

            relative_tower_error = np.abs((real_num_array - tower_num_array) / real_num_array)
            print("max absolut error:", np.max(absolute_tower_error))
            print("absolut error:" + "{:.6f}".format(np.mean(absolute_tower_error)))
            print("relative error:" + "{:.6f}".format(np.mean(relative_tower_error)))

            tower_used.clear()
        dataset_over = 1
