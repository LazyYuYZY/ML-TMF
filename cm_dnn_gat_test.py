import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import matplotlib.pyplot as plt

from DNN1.DNN1 import *
from GAT.GAT_model import GAT
from sketchs import *
from Metrics import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
T_slice=5
P_flowdrop=0

def conflict_gat(index_np,dict_cm,w=16 * 10 ** 4,d=3,max_conf=3):
    a = np.array(dict_cm['a'])
    b = np.array(dict_cm['b'])
    p = np.array(dict_cm['p'])
    offset = dict_cm['offset']
    cm_d_id=(a*(index_np+offset)+b)%p%w
    node_num=index_np.shape[0]
    edge_index_cm = []

    for i in range(d):
        unique_values, counts = np.unique(cm_d_id[i], return_counts=True)
        conf_items = unique_values[counts >= 1]
        for conf_cm_id in conf_items:  # 冲突流在sketch中的索引
            conf_flows_id = np.where(np.isin(cm_d_id[i], conf_cm_id))[0]  # 单个conter冲突流在sketch的索引
            conf_flows_id_list = conf_flows_id.tolist()
            for conf_flow_id in conf_flows_id_list:
                k=0
                for flow_j in conf_flows_id_list:
                    if flow_j==conf_flow_id:
                        continue
                    edge_index_cm.append([flow_j, conf_flow_id])
                    k=k+1
                    if k>=max_conf:
                        break;
                while k<max_conf:
                    dummy_node=node_num+i * max_conf+k
                    edge_index_cm.append([dummy_node,conf_flow_id])
                    k=k+1

    return torch.tensor(edge_index_cm, dtype=torch.float, device=device)

def conflict_dnn(index_np, dict_cm, w=22 * 10 ** 4, d=3, max_conf=2):
    a = np.array(dict_cm['a'])
    b = np.array(dict_cm['b'])
    p = np.array(dict_cm['p'])
    offset = dict_cm['offset']
    cm_d_id = (a * (index_np + offset) + b) % p % w
    data_x_d = np.full((index_np.shape[0], d * max_conf), -1)

    for i in range(d):
        unique_values, counts = np.unique(cm_d_id[i], return_counts=True)
        conf_items = unique_values[counts > 1]
        for conf_cm_id in conf_items:  # 冲突流在sketch中的索引
            conf_flows_id = np.where(np.isin(cm_d_id[i], conf_cm_id))[0]  # 单个conter冲突流在sketch的索引
            conf_flows_id_list = conf_flows_id.tolist()
            for conf_flow_id in conf_flows_id_list:
                k = 0
                for j in range(1 + max_conf):
                    if j >= len(conf_flows_id_list):
                        break
                    if conf_flows_id_list[j] != conf_flow_id:
                        data_x_d[conf_flow_id][i * max_conf + k] = index_np[conf_flows_id_list[j]]
                        k = k + 1
                        if k == max_conf:
                            break

    return data_x_d

def select_indices(index_array,p):
    # 计算要选择的行数
    num_rows =index_array.shape[0]
    num_selected_rows = int(num_rows * p)
    # 生成要选择的行索引
    selected_indices = np.random.choice(num_rows, size=num_selected_rows, replace=False)
    return selected_indices

def solution_real(w=1* 10 ** 4,flows_path="",file_now=0,real_sketch=False,dnn_conf_num=2,gat_conf_num=3):
    dnn_pkl="dnn_d_params_"+str(dnn_conf_num)+".pkl"
    gat_pkl="gat_d_params_"+str(gat_conf_num)+".pkl"
    # dnn_pkl="cm_dnn_d_params_"+str(dnn_conf_num)+".pkl"
    # gat_pkl="cm_gat_d_params_"+str(gat_conf_num)+".pkl"

    '''流数据集'''
    test_flows_data = rw_files.get_dict(flows_path)
    test_flows_data_list = list(test_flows_data.values())

    '''处理cm数据'''
    # 获取流映射
    flow_index_path = "./testdata_set_"+str(T_slice)+"s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flow_index_path)
    # 获取每个时刻的flows映射与数目
    flows_data_onetime = {}
    for one_flow in test_flows_data.keys():
        flows_data_onetime[flows_alltime_dict[one_flow]] = test_flows_data[one_flow]

    # 获取cm参数
    file_name = "./sketch_params/" + str(w).zfill(6) + "_cm_sketch.txt"
    dict_cm = rw_files.get_dict(file_name)
    cm_d = 3
    cm_w=dict_cm['w']
    new_dir="./testdata_set_"+str(T_slice)+"s/testdata_" + str(cm_w).zfill(6) + "_cm_"+str(T_slice)+"s"
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    cm_test_path = new_dir +"/"+ str(file_now).zfill(5) + ".txt"
    # 若有对应cm数据集导入，没有就插入cm导出数据集
    if real_sketch:
        cm_sketch_load = np.loadtxt(cm_test_path)
        cm_sketch_now = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_load)
    else:
        cm_sketch_load = np.full((cm_d, cm_w), 0)  # cm存储的counter值
        cm_sketch_now = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_load)
        cm_sketch_now.insert_dict(flows_data_onetime)
        cm_sketch_load = cm_sketch_now.Matrix  # cm的counter值
        '''导出数据'''
        np.savetxt(cm_test_path, cm_sketch_load, fmt='%d')

    '''获取test时刻dnn_cm的输入,即冲突流的id'''
    flows_index_path = "./testdata_set_"+str(T_slice)+"s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flows_index_path)
    get_value = np.vectorize(lambda x: flows_alltime_dict.get(x, 0))
    test_x_key = list(test_flows_data.keys())
    test_index_array = get_value(np.array(test_x_key))

    selected_indices = select_indices(test_index_array, 1 - P_flowdrop)
    test_index_array = test_index_array[selected_indices]

    test_x_ids = conflict_dnn(index_np=test_index_array, dict_cm=dict_cm,w=cm_w, d=cm_d,max_conf=dnn_conf_num)

    '''获取所在时刻冲突流的d个查询值'''
    test_x = np.full((test_x_ids.shape[0], cm_d * (1 + test_x_ids.shape[1])), 0)
    for i in range(cm_d * dnn_conf_num):
        conf_flows_id = np.where(test_x_ids[:, i] != -1)[0]
        test_x[conf_flows_id, i * cm_d:(i + 1) * cm_d] = cm_sketch_now.query_d_np(test_x_ids[conf_flows_id, i]).T
    test_x[:, cm_d * test_x_ids.shape[1]:] = cm_sketch_now.query_d_np(test_index_array).T

    '''test dataset'''
    test_x = np.log(1 + test_x)
    test_y = np.array(test_flows_data_list).reshape(-1, 1)[selected_indices]
    test_cm = np.amin(cm_sketch_now.query_d_np(test_index_array).T, axis=1).reshape(-1, 1)

    '''加载DNN模型'''
    dnn_model_path = dnn_pkl
    # dnn_model_object = DNN1(input_size=test_x.shape[1], deep_l=4 - 2,num=100)
    dnn_model_object = DNN0(input_size=test_x.shape[1], deep_l=4 - 2, num=100)
    dnn_model_object.load_state_dict(torch.load(dnn_model_path,map_location=device))

    '''DNN预测'''
    start_time=time.time()
    dnn_predictions = dnn_model_object(torch.tensor(test_x, dtype=torch.float, device=device))
    dnn_pre_arr = dnn_predictions.detach().numpy()
    test_dnn = np.round(np.exp(dnn_pre_arr))
    end_time = time.time()
    print(end_time - start_time)

    '''加载GAT模型'''
    gat_model_path = gat_pkl
    gat_model_object = GAT(
        num_of_layers= 3,
        num_heads_per_layer= [4, 4, 1],
        num_features_per_layer= [3, 9, 27, 81],
        add_skip_connection= True,
        bias= True,
        dropout= 0.0,
        layer_type= 2,
        log_attention_weights= False,
        dnn_layer_num= 4  # no need to store attentions, used only in playground.py for visualizations
    ).to(device)
    gat_model_object.load_state_dict(torch.load(gat_model_path))

    '''获取图结构'''
    node_features = cm_sketch_now.query_d_np(test_index_array).T
    dummy_node_features = np.zeros((gat_conf_num * cm_d, cm_d))
    node_features = np.concatenate((node_features, dummy_node_features), axis=0)
    node_features = np.log(1 + node_features)
    node_features = torch.tensor(node_features, dtype=torch.float, device=device)
    edge_index = conflict_gat(index_np=test_index_array,dict_cm=dict_cm,w=cm_w,d=cm_d,max_conf=gat_conf_num).t()
    graph_data = (node_features, edge_index.to(device))

    '''GAT预测'''
    start_time=time.time()
    gat_predictions = gat_model_object(graph_data)[0]
    gat_pre_arr = gat_predictions.detach().numpy()
    test_gat = np.round(np.exp(gat_pre_arr))
    end_time = time.time()
    print(end_time - start_time)

    '''性能评估'''
    gat_metrics = Metrics(real_val=test_y, pre_val=test_gat[:-cm_d*gat_conf_num])
    dnn_metrics = Metrics(real_val=test_y,pre_val=test_dnn)
    cm_metrics = Metrics(real_val=test_y,pre_val=test_cm)

    gat_metrics.get_allval()
    dnn_metrics.get_allval()
    cm_metrics.get_allval()

    return gat_metrics,dnn_metrics,cm_metrics

def solution_zipf(w=1* 10 ** 4,flows_path="",alpha=2.0,n=160000,zifp_sketch=False,numbering=0,dnn_conf_num=2,gat_conf_num=3):
    dnn_pkl="dnn_d_params_"+str(dnn_conf_num)+".pkl"
    gat_pkl="gat_d_params_"+str(gat_conf_num)+".pkl"

    '''流数据集'''
    test_flows_data_str = rw_files.get_dict(flows_path)
    test_flows_data={}
    for key, value in test_flows_data_str.items():
        new_key = int(key)
        test_flows_data[new_key] = value
    test_flows_data_list = list(test_flows_data.values())

    '''处理cm数据'''# zipf需要进一步处理
    file_name = "./sketch_params/" + str(w).zfill(6) + "_cm_sketch.txt"
    dict_cm = rw_files.get_dict(file_name)
    cm_d = 3
    cm_w = dict_cm['w']
    cm_test_path = "./zipf_testdata_set/cm/" + str(cm_w).zfill(6) + "_" + str(int(alpha * 10)) + "_" + str(n).zfill(7) + "_" + str(numbering).zfill(2) + ".txt"
    # 若有对应cm数据集导入，没有就插入cm导出数据集
    if zifp_sketch:
        cm_sketch_load = np.loadtxt(cm_test_path)
        cm_sketch_now = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_load)
    else:
        cm_sketch_load = np.full((cm_d, cm_w), 0)  # cm存储的counter值
        cm_sketch_now = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_load)
        cm_sketch_now.insert_dict(test_flows_data)
        cm_sketch_load = cm_sketch_now.Matrix  # cm的counter值
        '''导出数据'''
        np.savetxt(cm_test_path, cm_sketch_load, fmt='%d')

    '''获取test时刻dnn_cm的输入,即冲突流的id'''
    test_index_array = np.array(list(test_flows_data.keys())[1000:])
    test_x_ids = conflict_dnn(index_np=test_index_array, dict_cm=dict_cm, w=cm_w, d=cm_d, max_conf=dnn_conf_num)


    '''获取所在时刻冲突流的d个查询值'''
    test_x = np.full((test_x_ids.shape[0], cm_d * (1 + test_x_ids.shape[1])), 0)
    for i in range(cm_d * dnn_conf_num):
        conf_flows_id = np.where(test_x_ids[:, i] != -1)[0]
        test_x[conf_flows_id, i * cm_d:(i + 1) * cm_d] = cm_sketch_now.query_d_np(test_x_ids[conf_flows_id, i]).T
    test_x[:, cm_d * test_x_ids.shape[1]:] = cm_sketch_now.query_d_np(test_index_array).T

    '''test dataset'''
    test_x = np.log(1 + test_x)
    test_y = np.array(test_flows_data_list).reshape(-1, 1)
    test_cm = np.amin(cm_sketch_now.query_d_np(test_index_array).T, axis=1).reshape(-1, 1)

    '''加载DNN模型'''
    dnn_model_path = dnn_pkl
    dnn_model_object = DNN0(input_size=test_x.shape[1], deep_l=4 - 2)
    dnn_model_object.load_state_dict(torch.load(dnn_model_path, map_location=device))

    '''DNN预测'''
    start_time=time.time()
    dnn_predictions = dnn_model_object(torch.tensor(test_x, dtype=torch.float, device=device))
    dnn_pre_arr = dnn_predictions.detach().numpy()
    test_dnn = np.round(np.exp(dnn_pre_arr))
    end_time = time.time()
    print(end_time - start_time)

    '''加载GAT模型'''
    gat_model_path = 'gat_d_params_3.pkl'
    gat_model_object = GAT(
        num_of_layers= 3,
        num_heads_per_layer= [4, 4, 1],
        num_features_per_layer= [3, 9, 27, 81],
        add_skip_connection= True,
        bias= True,
        dropout= 0.0,
        layer_type= 2,
        log_attention_weights= False,
        dnn_layer_num= 4  # no need to store attentions, used only in playground.py for visualizations
    ).to(device)
    gat_model_object.load_state_dict(torch.load(gat_model_path, map_location=device))

    '''获取图结构'''
    node_features = cm_sketch_now.query_d_np(test_index_array).T
    dummy_node_features = np.zeros((gat_conf_num * cm_d, cm_d))
    node_features = np.concatenate((node_features, dummy_node_features), axis=0)
    node_features = np.log(1 + node_features)
    node_features = torch.tensor(node_features, dtype=torch.float, device=device)
    edge_index = conflict_gat(index_np=test_index_array,dict_cm=dict_cm,w=cm_w,d=cm_d,max_conf=gat_conf_num).t()
    graph_data = (node_features, edge_index.to(device))

    '''GAT预测'''
    start_time = time.time()
    gat_predictions = gat_model_object(graph_data)[0]
    gat_pre_arr = gat_predictions.detach().numpy()
    test_gat = np.round(np.exp(gat_pre_arr))
    end_time = time.time()
    print(end_time - start_time)

    '''性能评估'''
    gat_metrics = Metrics(real_val=test_y, pre_val=test_gat[:-cm_d*gat_conf_num])
    dnn_metrics = Metrics(real_val=test_y,pre_val=test_dnn)
    cm_metrics = Metrics(real_val=test_y,pre_val=test_cm)

    gat_metrics.get_allval()
    dnn_metrics.get_allval()
    cm_metrics.get_allval()

    return gat_metrics,dnn_metrics,cm_metrics

#%%
def draws(x_list,y_np,x_label,real=0):
    if real:
        folder_path = "./results/wide"
    else:
        folder_path = "./results/zipf"
    # 创建画布和子图
    row_num=2
    col_num=3
    fig, axes = plt.subplots(row_num, col_num, figsize=(16, 8))
    sketchs = ["GAT","DNN", "CM"]
    titles = ["ARE(FS)", "AAE(FS)", "F1(FS)", "F1(HeavyHitter)","WMRE(ED)","RE(E)"]
    y_mean = np.mean(y_np, axis=3)
    y_variance = np.var(y_np, axis=3)
    y_err = 1.96 * np.sqrt(y_variance / y_np.shape[3])
    # 遍历六幅图
    for i in range(row_num):
        for j in range(col_num):
            ax = axes[i, j]
            ax.set_title(f"Plot {titles[i * col_num + j]}")
            if 1<i * col_num + j<4:
                ax.set_ylim([0, 1.1])
            else:
                ax.set_ylim([-10, 3])
            # 遍历三条曲线
            for k in range(3):
                fname_mean = x_label[0] + str(k) + str(i * col_num + j) + "mean.txt"
                np.savetxt(folder_path+"/"+fname_mean, y_mean[k, :, i * col_num + j])
                if 1<i * col_num + j<4:
                    ax.plot(x_list, y_mean[k, :, i * col_num + j], label=f"Line {sketchs[k]}")
                    # ax.scatter(x_list, y_mean[k, :, i * col_num + j])
                    ax.errorbar(x_list, y_mean[k, :, i * col_num + j], y_err[k, :, i * col_num + j], fmt='o', linewidth=0.5, capsize=0.5)
                    # plt.fill_between(x_list, y_mean[k, :, i * col_num + j]-y_err[k, :, i * col_num + j], y_mean[k, :, i * col_num + j]+y_err[k, :, i * col_num + j])
                    ax.set_ylabel('rate')
                    fname_err = x_label[0] + str(k) + str(i * col_num + j) + "err.txt"
                    np.savetxt(folder_path+"/"+fname_err, y_err[k, :, i * col_num + j])
                else:
                    y_err_lg=np.full((2,y_np.shape[0],y_np.shape[1],y_np.shape[2]),0)
                    if np.all(y_mean>y_err):
                        y_err_lg[0]=np.log10(y_mean)-np.log10(y_mean-y_err)
                    else:
                        y_err_lg[0]=np.log10(y_mean+y_err)-np.log10(y_mean)
                    y_err_lg[1]=np.log10(y_mean+y_err)-np.log10(y_mean)
                    ax.plot(x_list, np.log10(y_mean[k, :, i * col_num + j]), label=f"Line {sketchs[k]}")
                    # ax.scatter(x_list, np.log10(y_mean[k, :, i * col_num + j]))
                    ax.errorbar(x_list, np.log10(y_mean[k, :, i * col_num + j]), y_err_lg[:, k, :, i * col_num + j], fmt='o', linewidth=0.5, capsize=0.5)
                    # plt.fill_between(x_list, np.log10(y_mean[k, :, i * col_num + j] - y_err[k, :, i * col_num + j]), np.log10(y_mean[k, :, i * col_num + j] + y_err[k, :, i * col_num + j]))
                    ax.set_ylabel('rate(lg)')
                    fname_err_lg=x_label[0]+str(k)+str(i*col_num+j)+"err_lg.txt"
                    np.savetxt(folder_path+"/"+fname_err_lg,y_err_lg[:, k, :, i * col_num + j])

            ax.legend()
            ax.set_xlabel(x_label)

    # 调整子图之间的间距
    fig.tight_layout()
    # 显示图形
    # plt.show()

if __name__ == '__main__':
    dataset_type_dict = {1: "wide", 0: "zipf"}
    test_nums=10
    metrics_num=6
    w0=1*10**4
    # plt.ion()
    sketchs = ["GAT", "DNN" ,"CM"]
    P_flowdrop =0
#%%真实数据集：变化内存240KB 480KB 720KB 960KB 1440KB 1920KB
    memory_list = [8, 12, 16, 24, 32]
    # memory_list = [8,16,32]
    plots_np=np.full((3,len(memory_list),metrics_num,test_nums),0,dtype=float)#类型（sketch种类）k，横坐标i，纵坐标类型j,测试用例数量l
    for i in range(len(memory_list)):#变化内存
        w=int(memory_list[i]*w0)
        for l in range(test_nums):
            T_slice = 5
            file_now = l
            test_flows_path = "./testdata_set_" + str(T_slice) + "s/testdata_flows_" + str(T_slice) + "s/" + str(
                file_now).zfill(5) + ".txt"
            # test_flows_path = "./testdata2_set_" + str(T_slice) + "s/testdata_flows_" + str(T_slice) + "s/" + str(
            #     file_now).zfill(5) + ".txt"
            # test_flows_path = "./testdata3_set_" + str(T_slice) + "s/testdata_flows_" + str(T_slice) + "s/" + str(
            #     file_now).zfill(5) + ".txt"
            # test_flows_path = "./traindata_set_5s/traindata_flows_5s/" + str(file_now).zfill(5) + ".txt"
            list_metrics = solution_real(w=w, flows_path=test_flows_path, file_now=file_now, real_sketch=False,
                                         dnn_conf_num=2, gat_conf_num=3)
            for k in range(3):
                print(k)
                print(list_metrics[k].ARE_val)
                plots_np[k, i, 0, l] = list_metrics[k].ARE_val
                plots_np[k, i, 1, l] = list_metrics[k].AAE_val
                plots_np[k, i, 2, l] = list_metrics[k].F1_val
                plots_np[k, i, 3, l] = list_metrics[k].F1_val_hh
                plots_np[k, i, 4, l] = list_metrics[k].WMRE_val
                plots_np[k, i, 5, l] = list_metrics[k].RE_val
        for k in range(3):
            results_path = "./results/wide/" + "Memory" + "/" + sketchs[k] + "_" + str(w) + ".txt"
            np.savetxt(results_path, plots_np[k, i, :, :])
    #%%
    # draws(memory_list,plots_np,"w(10^4)",real=1)


    # memory_list = [16]
    # plots_np=np.full((3,len(memory_list),metrics_num,test_nums),0,dtype=float)#类型（sketch种类）k，横坐标i，纵坐标类型j,测试用例数量l
    # for i in range(len(memory_list)):#变化内存
    #     w=int(memory_list[i]*w0)
    #     for drop_flow in [0.01,0.001,0.0001,0.00001,0.000001]:
    #         P_flowdrop=drop_flow
    #         print("MEMORY:",memory_list[i]*0.12,"MB,DROP P:",drop_flow)
    #         for l in range(test_nums):
    #             T_slice=5
    #             file_now = l
    #             test_flows_path = "./testdata_set_"+str(T_slice)+"s/testdata_flows_"+str(T_slice)+"s/" + str(file_now).zfill(5) + ".txt"
    #             # test_flows_path = "./testdata2_set_" + str(T_slice) + "s/testdata_flows_" + str(T_slice) + "s/" + str(
    #             #     file_now).zfill(5) + ".txt"
    #             # test_flows_path = "./testdata3_set_" + str(T_slice) + "s/testdata_flows_" + str(T_slice) + "s/" + str(
    #             #     file_now).zfill(5) + ".txt"
    #             # test_flows_path = "./traindata_set_5s/traindata_flows_5s/" + str(file_now).zfill(5) + ".txt"
    #             list_metrics=solution_real(w=w,flows_path=test_flows_path,file_now=file_now,real_sketch=False,dnn_conf_num=2,gat_conf_num=3)
    #             for k in range(3):
    #                 print(k)
    #                 print(list_metrics[k].ARE_val)
    #                 plots_np[k, i, 0, l]=list_metrics[k].ARE_val
    #                 plots_np[k, i, 1, l] = list_metrics[k].AAE_val
    #                 plots_np[k, i, 2, l] = list_metrics[k].F1_val
    #                 plots_np[k, i, 3, l] = list_metrics[k].F1_val_hh
    #                 plots_np[k, i, 4, l] = list_metrics[k].WMRE_val
    #                 plots_np[k, i, 5, l] = list_metrics[k].RE_val
    #         for k in range(3):
    #             results_path = "./results/wide/" + "Bloom" + "/" + sketchs[k] + "_" + str(int(-np.log10(drop_flow))) + ".txt"
    #             np.savetxt(results_path, plots_np[k, i, :, :])
    # #%%
    # draws(memory_list,plots_np,"p",real=1)


    # P_flowdrop =0
# #%%合成数据集：变化内存240KB 480KB 720KB 960KB 1440KB 1920KB
#     memory_list = [4, 8, 12, 16, 24, 32]
#     plots_np = np.full((3, len(memory_list), metrics_num, test_nums), 0, dtype=float)  # 类型（sketch种类）k，横坐标i，纵坐标类型j,测试用例数量l
#     for i in range(len(memory_list)):  # 变化内存
#         w = memory_list[i] * w0
#         for l in range(test_nums):
#             test_flows_path = "./zipf_testdata_set/testdata_flows/20_0160000_" + str(l).zfill(2) + ".txt"
#             list_metrics = solution_zipf(w=w, flows_path=test_flows_path, alpha=2.0, n=160000,
#                                          zifp_sketch=True, numbering=l,dnn_conf_num=2,gat_conf_num=3)
#             for k in range(3):
#                 plots_np[k, i, 0, l] = list_metrics[k].ARE_val
#                 plots_np[k, i, 1, l] = list_metrics[k].AAE_val
#                 plots_np[k, i, 2, l] = list_metrics[k].F1_val
#                 plots_np[k, i, 3, l] = list_metrics[k].F1_val_hh
#                 plots_np[k, i, 4, l] = list_metrics[k].WMRE_val
#                 plots_np[k, i, 5, l] = list_metrics[k].RE_val
#         for k in range(3):
#             results_path = "./results/zipf/" + "Memory" + "/" + sketchs[k] + "_" + str(w) + ".txt"
#             np.savetxt(results_path, plots_np[k, i, :, :])
# # #%%
# #     draws(memory_list, plots_np, "w(10^4)")
#
# #%%合成数据集：变化偏度 [1.2,1.5,2.0,2.5,3.0]
#     alpha_list=[1.2,1.4,1.6,1.8,2.0,2.2,2.4]
#     plots_np = np.full((3, len(alpha_list), metrics_num, test_nums), 0, dtype=float)  # 类型（sketch种类）k，横坐标i，纵坐标类型j,测试用例数量l
#     for i in range(len(alpha_list)):#变化zipf的α
#         for l in range(test_nums):
#             test_flows_path="./zipf_testdata_set/testdata_flows/"+str(int(alpha_list[i]*10))+"_0160000_"+str(l).zfill(2)+".txt"
#             list_metrics=solution_zipf(w=160000,flows_path=test_flows_path,alpha=alpha_list[i],n=160000,zifp_sketch=False,numbering=l,dnn_conf_num=2,gat_conf_num=3)
#             for k in range(3):
#                 print(list_metrics[k].ARE_val)
#                 plots_np[k, i, 0, l]=list_metrics[k].ARE_val
#                 plots_np[k, i, 1, l] = list_metrics[k].AAE_val
#                 plots_np[k, i, 2, l] = list_metrics[k].F1_val
#                 plots_np[k, i, 3, l] = list_metrics[k].F1_val_hh
#                 plots_np[k, i, 4, l] = list_metrics[k].WMRE_val
#                 plots_np[k, i, 5, l] = list_metrics[k].RE_val
#
#         for k in range(3):
#             results_path = "./results/zipf/" + "Alpha" + "/" + sketchs[k] + "_" +str(int(alpha_list[i]*10)) + ".txt"
#             np.savetxt(results_path, plots_np[k, i, :, :])
# # #%%
# #     draws(alpha_list,plots_np,"alpha")
#
#
# #%%真实数据集：变化时间片长度
#     T_list = [5,10,15,20,25,30]
#     plots_np = np.full((3, len(T_list), metrics_num, test_nums), 0, dtype=float)  # 类型（sketch种类）k，横坐标i，纵坐标类型j,测试用例数量l
#     for i in range(len(T_list)):  # 变化zipf的α
#         T_slice=T_list[i]
#         for l in range(test_nums):
#             real_sketch = True
#             file_now=l
#             test_flows_path = "./testdata_set_"+str(T_slice)+"s/testdata_flows_"+str(T_slice)+"s/" + str(file_now).zfill(5) + ".txt"
#             list_metrics = solution_real(w=160000, flows_path=test_flows_path, file_now=file_now, real_sketch=real_sketch,
#                                          dnn_conf_num=2, gat_conf_num=3)
#             for k in range(3):
#                 print(list_metrics[k].ARE_val)
#                 plots_np[k, i, 0, l] = list_metrics[k].ARE_val
#                 plots_np[k, i, 1, l] = list_metrics[k].AAE_val
#                 plots_np[k, i, 2, l] = list_metrics[k].F1_val
#                 plots_np[k, i, 3, l] = list_metrics[k].F1_val_hh
#                 plots_np[k, i, 4, l] = list_metrics[k].WMRE_val
#                 plots_np[k, i, 5, l] = list_metrics[k].RE_val
#
#         for k in range(3):
#             results_path="./results/wide/"+"T"+"/"+sketchs[k]+"_"+str(T_slice)+".txt"
#             np.savetxt(results_path,plots_np[k,i,:,:])
# #%%
#     draws(T_list, plots_np, "T")
#
#     plt.ioff()
#     plt.show()
