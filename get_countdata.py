
from sketchs import *


if __name__ == '__main__':
    # 是否采用已有cm参数
    flag = 1
    count_d = 3
    w = 1 * 10 ** 4
    memory_list = [16]
    # 若没有数据集处理pcap文件
    for i in memory_list:
        # 获取旧的cm参数
        count_w = w * i
        count_parameter_path = "./sketch_params/" + str(count_w).zfill(6) + "_cm_sketch.txt"
        dict_count = rw_files.get_dict(count_parameter_path)
        count_w = w * i
        # 创建使用的Count Sketch
        count_sketch_now=np.full((count_d, count_w), 0)  # count存储的counter值
        count_used = count_sketch(count_d=count_d, count_w=count_w, flag=flag, dict_count=dict_count, count_sketch_load=count_sketch_now)

        # 获取流映射
        # flow_index_path = "./traindata_set_5s/flow_index.txt"
        flow_index_path = "./testdata_set_5s/flow_index.txt"
        flows_alltime_dict = rw_files.get_dict(flow_index_path)

        # 选择要输入的流的位置
        for file_now in range(10):
            # 读取流文件
            # json_file_path = "./traindata_set_5s/traindata_flows_5s/" + str(file_now).zfill(5) + ".txt"
            json_file_path = "./testdata_set_5s/testdata_flows_5s/" + str(file_now).zfill(5) + ".txt"
            flows_data = rw_files.get_dict(json_file_path)

            # 获取每个时刻的flows映射与数目
            flows_data_onetime = {}
            for one_flow in flows_data.keys():
                flows_data_onetime[flows_alltime_dict[one_flow]] = flows_data[one_flow]

            # 流通过count sketch
            count_used.insert_dict(flows_data_onetime)

            # 查询count的counter值:(d,w)
            count_sketch_now = count_used.Matrix  # count的counter值

            '''导出数据'''
            # 将矩阵保存为文本文件
            # 拆分拼接字符串 创建txt文件
            dict_list = json_file_path.split('testdata_flows_5s')
            count_path = dict_list[0] + "testdata_"+str(count_w).zfill(6) +"_count_5s" + dict_list[1]
            folder_path=count_path.split(dict_list[1])[0]

            # 判断文件夹是否存在
            if not os.path.exists(folder_path):
                print("文件夹不存在")
                os.mkdir(folder_path)

            np.savetxt(count_path, count_sketch_now, fmt='%d')

            # 查询所有流在count sketch中的值
            count_flows_query_d = count_used.query_d(list(flows_data_onetime.keys()))
            # DNN dataset:每一行的查询值(n,d)
            dataset_x = count_flows_query_d.T
            id_x = np.array(list(flows_data_onetime.keys())).T
            dataset_x[id_x%2!=0,:]=-dataset_x[id_x%2!=0,:]
            # 流的查询值，即对每行求min:(n,1)
            flows_query_count = np.median(dataset_x,axis=1)
            # dataset_y即为flows_data的value:(n,1)
            dataset_y = np.array(list(flows_data_onetime.values())).T

            # #获取count的相对误差
            count_num_array = flows_query_count.T
            real_num_array = np.array(list(flows_data_onetime.values()))
            absolute_count_error = np.abs((real_num_array - count_num_array))

            relative_count_error = np.abs((real_num_array - count_num_array) / real_num_array)
            print("max absolut error:", np.max(absolute_count_error))
            print("absolut error:" + "{:.6f}".format(np.mean(absolute_count_error)))
            print("relative error:" + "{:.6f}".format(np.mean(relative_count_error)))

            count_used.clear()
