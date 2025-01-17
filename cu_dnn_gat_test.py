import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing

from DNN1.DNN1 import *
from GAT.GAT_model import GAT
from sketchs import *
from Metrics import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
T_slice=5

def conflict_gat(index_np,dict_cu,w=16 * 10 ** 4,d=3,max_conf=3):
    a = np.array(dict_cu['a'])
    b = np.array(dict_cu['b'])
    p = np.array(dict_cu['p'])
    offset = dict_cu['offset']
    cu_d_id=(a*(index_np+offset)+b)%p%w
    node_num=index_np.shape[0]
    edge_index_cu = []

    for i in range(d):
        unique_values, counts = np.unique(cu_d_id[i], return_counts=True)
        conf_items = unique_values[counts >= 1]
        for conf_cu_id in conf_items:  # 冲突流在sketch中的索引
            conf_flows_id = np.where(np.isin(cu_d_id[i], conf_cu_id))[0]  # 单个conter冲突流在sketch的索引
            conf_flows_id_list = conf_flows_id.tolist()
            for conf_flow_id in conf_flows_id_list:
                k=0
                for flow_j in conf_flows_id_list:
                    if flow_j==conf_flow_id:
                        continue
                    edge_index_cu.append([flow_j, conf_flow_id])
                    k=k+1
                    if k>=max_conf:
                        break;
                while k<max_conf:
                    dummy_node=node_num+i * max_conf+k
                    edge_index_cu.append([dummy_node,conf_flow_id])
                    k=k+1

    return torch.tensor(edge_index_cu, dtype=torch.float, device=device)

def conflict_dnn(index_np, dict_cu, w=22 * 10 ** 4, d=3, max_conf=2):
    a = np.array(dict_cu['a'])
    b = np.array(dict_cu['b'])
    p = np.array(dict_cu['p'])
    offset = dict_cu['offset']
    cu_d_id = (a * (index_np + offset) + b) % p % w
    data_x_d = np.full((index_np.shape[0], d * max_conf), -1)

    for i in range(d):
        unique_values, counts = np.unique(cu_d_id[i], return_counts=True)
        conf_items = unique_values[counts > 1]
        for conf_cu_id in conf_items:  # 冲突流在sketch中的索引
            conf_flows_id = np.where(np.isin(cu_d_id[i], conf_cu_id))[0]  # 单个conter冲突流在sketch的索引
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

def solution_real(w=1* 10 ** 4,flows_path="",file_now=0,real_sketch=False,dnn_conf_num=2,gat_conf_num=3):
    dnn_pkl="dnn_d_params_"+str(dnn_conf_num)+".pkl"
    gat_pkl="gat_d_params_"+str(gat_conf_num)+".pkl"

    '''流数据集'''
    test_flows_data = rw_files.get_dict(flows_path)
    test_flows_data_list = list(test_flows_data.values())

    '''处理cu数据'''
    # 获取流映射
    flow_index_path = "./testdata_set_"+str(T_slice)+"s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flow_index_path)
    # 获取每个时刻的flows映射与数目
    flows_data_onetime = {}
    for one_flow in test_flows_data.keys():
        flows_data_onetime[flows_alltime_dict[one_flow]] = test_flows_data[one_flow]

    # 获取cu参数
    file_name = "./sketch_params/" + str(w).zfill(6) + "_cm_sketch.txt"
    dict_cu = rw_files.get_dict(file_name)
    cu_d = 3
    cu_w=dict_cu['w']
    new_dir="./testdata_set_"+str(T_slice)+"s/testdata_" + str(cu_w).zfill(6) + "_cu_"+str(T_slice)+"s"
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    cu_test_path = new_dir +"/"+ str(file_now).zfill(5) + ".txt"
    # 若有对应cu数据集导入，没有就插入cu导出数据集
    if real_sketch:
        cu_sketch_load = np.loadtxt(cu_test_path)
        cu_sketch_now = cu_sketch(cu_d=cu_d, cu_w=cu_w, flag=1, dict_cu=dict_cu, cu_sketch_load=cu_sketch_load)
    else:
        cu_sketch_load = np.full((cu_d, cu_w), 0)  # cu存储的counter值
        cu_sketch_now = cu_sketch(cu_d=cu_d, cu_w=cu_w, flag=1, dict_cu=dict_cu, cu_sketch_load=cu_sketch_load)
        cu_sketch_now.insert_dict(flows_data_onetime)
        cu_sketch_load = cu_sketch_now.Matrix  # cu的counter值
        '''导出数据'''
        np.savetxt(cu_test_path, cu_sketch_load, fmt='%d')

    '''获取test时刻dnn_cu的输入,即冲突流的id'''
    flows_index_path = "./testdata_set_"+str(T_slice)+"s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flows_index_path)
    get_value = np.vectorize(lambda x: flows_alltime_dict.get(x, 0))
    test_x_key = list(test_flows_data.keys())
    test_index_array = get_value(np.array(test_x_key))
    test_x_ids = conflict_dnn(index_np=test_index_array, dict_cu=dict_cu,w=cu_w, d=cu_d,max_conf=dnn_conf_num)

    '''获取所在时刻冲突流的d个查询值'''
    test_x = np.full((test_x_ids.shape[0], cu_d * (1 + test_x_ids.shape[1])), 0)
    for i in range(cu_d * dnn_conf_num):
        conf_flows_id = np.where(test_x_ids[:, i] != -1)[0]
        test_x[conf_flows_id, i * cu_d:(i + 1) * cu_d] = cu_sketch_now.query_d_np(test_x_ids[conf_flows_id, i]).T
    test_x[:, cu_d * test_x_ids.shape[1]:] = cu_sketch_now.query_d_np(test_index_array).T

    '''test dataset'''
    test_x = np.log(1 + test_x)
    # test_x = preprocessing.StandardScaler().fit_transform(test_x)
    test_y = np.array(test_flows_data_list).reshape(-1, 1)
    test_cu = np.amin(cu_sketch_now.query_d_np(test_index_array).T, axis=1).reshape(-1, 1)

    '''加载DNN模型'''
    dnn_model_path = dnn_pkl
    dnn_model_object = DNN0(input_size=test_x.shape[1], deep_l=4 - 2)
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
    node_features = cu_sketch_now.query_d_np(test_index_array).T
    dummy_node_features = np.zeros((gat_conf_num * cu_d, cu_d))
    node_features = np.concatenate((node_features, dummy_node_features), axis=0)
    node_features = np.log(1 + node_features)
    node_features = torch.tensor(node_features, dtype=torch.float, device=device)
    edge_index = conflict_gat(index_np=test_index_array,dict_cu=dict_cu,w=cu_w,d=cu_d,max_conf=gat_conf_num).t()
    graph_data = (node_features, edge_index.to(device))

    '''GAT预测'''
    start_time=time.time()
    gat_predictions = gat_model_object(graph_data)[0]
    gat_pre_arr = gat_predictions.detach().numpy()
    test_gat = np.round(np.exp(gat_pre_arr))
    end_time = time.time()
    print(end_time - start_time)

    '''性能评估'''
    gat_metrics = Metrics(real_val=test_y, pre_val=test_gat[:-cu_d*gat_conf_num])
    dnn_metrics = Metrics(real_val=test_y,pre_val=test_dnn)
    cu_metrics = Metrics(real_val=test_y,pre_val=test_cu)

    gat_metrics.get_allval()
    dnn_metrics.get_allval()
    cu_metrics.get_allval()

    return gat_metrics,dnn_metrics,cu_metrics

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
    sketchs = ["GAT","DNN", "CU"]
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
    sketchs = ["GAT", "DNN", "CU"]

#%%真实数据集：变化内存240KB 480KB 720KB 960KB 1440KB 1920KB
    memory_list = [16]
    plots_np=np.full((3,len(memory_list),metrics_num,test_nums),0,dtype=float)#类型（sketch种类）k，横坐标i，纵坐标类型j,测试用例数量l
    for i in range(len(memory_list)):#变化内存
        w=memory_list[i]*w0
        for l in range(test_nums):
            file_now = l
            test_flows_path = "./testdata_set_5s/testdata_flows_5s/" + str(file_now).zfill(5) + ".txt"
            list_metrics=solution_real(w=w,flows_path=test_flows_path,file_now=file_now,real_sketch=True,dnn_conf_num=2,gat_conf_num=3)
            for k in range(3):
                print(k)
                print(list_metrics[k].ARE_val)
                plots_np[k, i, 0, l]=list_metrics[k].ARE_val
                plots_np[k, i, 1, l] = list_metrics[k].AAE_val
                plots_np[k, i, 2, l] = list_metrics[k].F1_val
                plots_np[k, i, 3, l] = list_metrics[k].F1_val_hh
                plots_np[k, i, 4, l] = list_metrics[k].WMRE_val
                plots_np[k, i, 5, l] = list_metrics[k].RE_val
        for k in range(3):
            results_path = "./results/wide/" + "CU" + "/" + sketchs[k]+ "_" + str(w) + ".txt"
            np.savetxt(results_path, plots_np[k, i, :, :])
# #%%
#     draws(memory_list,plots_np,"w(10^4)",real=1)


#     plt.ioff()
#     plt.show()
