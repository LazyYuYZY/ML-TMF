import random
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import os
import sys
import time
from DNN1.DNN1 import *
from GAT.GAT_model import GAT
from sketchs import *
from Metrics import *

class rw_files(object):
    @staticmethod
    def get_dict(json_file_name):
        # 读取JSON文件并解析为字典对象
        with open(json_file_name, 'r') as json_file:
            lines = json_file.readlines()[1:]  # 读取所有行并从第二行开始
            json_data = json.loads("".join(lines))
        return json_data

    @staticmethod
    def change_name(folder_path, old_part, new_part):
        files = os.listdir(folder_path)  # 获取文件夹下的所有文件名

        for file in files:
            file_path = os.path.join(folder_path, file)  # 获取文件的完整路径
            tlist = file.split(old_part)
            new_file_name = tlist[0] + new_part
            new_file_path = os.path.join(folder_path, new_file_name)  # 构建新文件的完整路径
            # 移动文件并重命名
            os.rename(file_path, new_file_path)


def conflict(index_np, dict_cm, w=22 * 10 ** 4, d=3, max_conf=1):
    a = np.array(dict_cm['a'])
    b = np.array(dict_cm['b'])
    p = np.array(dict_cm['p'])
    offset = dict_cm['offset']
    cm_d_id = (a * (index_np + offset) + b) % p % w
    data_x_d = np.full((index_np.shape[0], d * max_conf), -1)

    for i in range(d):
        unique_values, counts = np.unique(cm_d_id[i], return_counts=True)

        # plot_x, plot_y = np.unique(counts, return_counts=True)
        # # plot_x, plot_y = np.unique(counts[counts > 1], return_counts=True)
        # plt.figure()
        # plt.pie(plot_y, labels=plot_x, autopct='%1.1f%%')
        # plt.show()
        # plt.figure()
        # plt.bar(plot_x,plot_y)
        # for plt_i, value in enumerate(plot_y):
        #     plt.text(plt_i+plot_x[0], value, str(value), ha='center', va='bottom')
        # plt.show()

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




def solution_real(w=1* 10 ** 4,flows_path="",file_now=0):
    max_conf = 2

    '''流数据集'''
    test_flows_data = rw_files.get_dict(flows_path)
    test_flows_data_list = list(test_flows_data.values())

    '''处理cm数据'''
    file_name = "./testdata_set_5s/" + str(w).zfill(6) + "_cm_sketch.txt"
    dict_cm = rw_files.get_dict(file_name)
    cm_d = 3
    cm_w=dict_cm['w']
    cm_test_path = "./testdata_set_5s/testdata_" + str(cm_w).zfill(6) + "_cm_5s/20240103_5s_" + str(file_now).zfill(5) + ".txt"
    cm_sketch_load = np.loadtxt(cm_test_path)
    cm_sketch_now = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_load)

    '''处理cu数据'''
    # cu与cm用相同sketch参数
    cu_test_path = "./testdata_set_5s/testdata_" + str(cm_w).zfill(
        6) + "_cu_5s/20240103_5s_" + str(file_now).zfill(5) + ".txt"
    cu_sketch_load = np.loadtxt(cu_test_path)
    cu_sketch_now = cu_sketch(cu_d=cm_d, cu_w=cm_w, flag=1, dict_cu=dict_cm, cu_sketch_load=cu_sketch_load)

    '''处理count数据'''
    count_test_path = "./testdata_set_5s/testdata_" + str(cm_w).zfill(
        6) + "_count_5s/20240103_5s_" + str(file_now).zfill(5) + ".txt"
    count_sketch_load = np.loadtxt(count_test_path)
    count_sketch_now = count_sketch(count_d=cm_d, count_w=cm_w, flag=1, dict_count=dict_cm, count_sketch_load=count_sketch_load)

    '''获取test时刻dnn_cm的输入,即冲突流的id'''
    flows_index_path = "./testdata_set_5s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flows_index_path)
    get_value = np.vectorize(lambda x: flows_alltime_dict.get(x, 0))
    test_x_key = list(test_flows_data.keys())
    test_index_array = get_value(np.array(test_x_key))
    test_x_ids = conflict(index_np=test_index_array, dict_cm=dict_cm,w=cm_w, d=cm_d,max_conf=max_conf)

    '''获取所在时刻冲突流的d个查询值'''
    test_x = np.full((test_x_ids.shape[0], cm_d * (1 + test_x_ids.shape[1])), 0)
    for i in range(cm_d * max_conf):
        conf_flows_id = np.where(test_x_ids[:, i] != -1)[0]
        test_x[conf_flows_id, i * cm_d:(i + 1) * cm_d] = cm_sketch_now.query_d_np(test_x_ids[conf_flows_id, i]).T
    test_x[:, cm_d * test_x_ids.shape[1]:] = cm_sketch_now.query_d_np(test_index_array).T

    '''test dataset'''
    test_x = np.log(1 + test_x)
    test_y = np.array(test_flows_data_list).reshape(-1, 1)
    test_cm = np.amin(cm_sketch_now.query_d_np(test_index_array).T, axis=1).reshape(-1, 1)
    test_cu = np.amin(cu_sketch_now.query_d_np(test_index_array).T, axis=1).reshape(-1, 1)
    test_count = np.median(count_sketch_now.query_d_np(test_index_array).T, axis=1).reshape(-1, 1)

    '''加载模型'''
    model_path = 'dnn_d_params_' + str(max_conf) + '.pkl'
    model_object = DNN1(input_size=test_x.shape[1], deep_l=4 - 2)
    model_object.load_state_dict(torch.load(model_path))



    '''预测'''
    predictions = model_object(torch.tensor(test_x, dtype=torch.float))
    pre_arr = predictions.detach().numpy()
    test_dnn = np.round(np.exp(pre_arr))
    '''性能评估'''
    dnn_metrics=Metrics(real_val=test_y,pre_val=test_dnn)
    cm_metrics=Metrics(real_val=test_y,pre_val=test_cm)
    cu_metrics = Metrics(real_val=test_y, pre_val=test_cu)
    count_metrics = Metrics(real_val=test_y, pre_val=test_count)

    dnn_metrics.get_allval()
    cm_metrics.get_allval()
    cu_metrics.get_allval()
    count_metrics.get_allval()

    return dnn_metrics,cm_metrics,cu_metrics,count_metrics

def solution_zipf(w=1* 10 ** 4,flows_path="",alpha=2.0,n=160000,zifp_sketch=False,numbering=0):
    max_conf = 2

    '''流数据集'''
    test_flows_data_str = rw_files.get_dict(flows_path)
    test_flows_data={}
    for key, value in test_flows_data_str.items():
        new_key = int(key)
        test_flows_data[new_key] = value
    test_flows_data_list = list(test_flows_data.values())

    '''处理cm数据'''
    file_name = "./testdata_set_5s/" + str(w).zfill(6) + "_cm_sketch.txt"
    dict_cm = rw_files.get_dict(file_name)
    cm_d = 3
    cm_w = dict_cm['w']
    cm_test_path = "./zipf_testdata_set/cm/" + str(cm_w).zfill(6) + "_" + str(int(alpha * 10)) + "_" + str(n).zfill(7) + "_" + str(numbering).zfill(2) + ".txt"
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


    '''处理cu数据'''
    # cu与cm用相同sketch参数
    cu_test_path = "./zipf_testdata_set/cu/" + str(cm_w).zfill(6) + "_" + str(
        int(alpha * 10)) + "_" + str(n).zfill(7) + "_" + str(numbering).zfill(2) + ".txt"
    if zifp_sketch:
        cu_sketch_load = np.loadtxt(cu_test_path)
        cu_sketch_now = cu_sketch(cu_d=cm_d, cu_w=cm_w, flag=1, dict_cu=dict_cm, cu_sketch_load=cu_sketch_load)
    else:
        cu_sketch_load = np.full((cm_d, cm_w), 0)  # cm存储的counter值
        cu_sketch_now = cu_sketch(cu_d=cm_d, cu_w=cm_w, flag=1, dict_cu=dict_cm, cu_sketch_load=cu_sketch_load)
        cu_sketch_now.insert_dict(test_flows_data)
        cu_sketch_load = cu_sketch_now.Matrix  # cm的counter值
        '''导出数据'''
        np.savetxt(cu_test_path, cu_sketch_load, fmt='%d')


    '''处理count数据'''
    count_test_path = "./zipf_testdata_set/count/" + str(cm_w).zfill(6) + "_" + str(
        int(alpha * 10)) + "_" + str(n).zfill(7) + "_" + str(numbering).zfill(2) + ".txt"
    if zifp_sketch:
        count_sketch_load = np.loadtxt(count_test_path)
        count_sketch_now = count_sketch(count_d=cm_d, count_w=cm_w, flag=1, dict_count=dict_cm,
                                        count_sketch_load=count_sketch_load)
    else:
        count_sketch_load = np.full((cm_d, cm_w), 0)  # cm存储的counter值
        count_sketch_now = count_sketch(count_d=cm_d, count_w=cm_w, flag=1, dict_count=dict_cm,
                                        count_sketch_load=count_sketch_load)
        count_sketch_now.insert_dict(test_flows_data)
        count_sketch_load = count_sketch_now.Matrix  # cm的counter值
        '''导出数据'''
        np.savetxt(count_test_path, count_sketch_load, fmt='%d')


    '''获取test时刻dnn_cm的输入,即冲突流的id'''
    test_index_array = np.array(list(test_flows_data.keys()))
    test_x_ids = conflict(index_np=test_index_array, dict_cm=dict_cm, w=cm_w, d=cm_d, max_conf=max_conf)

    '''获取所在时刻冲突流的d个查询值'''
    test_x = np.full((test_x_ids.shape[0], cm_d * (1 + test_x_ids.shape[1])), 0)
    for i in range(cm_d * max_conf):
        conf_flows_id = np.where(test_x_ids[:, i] != -1)[0]
        test_x[conf_flows_id, i * cm_d:(i + 1) * cm_d] = cm_sketch_now.query_d_np(test_x_ids[conf_flows_id, i]).T
    test_x[:, cm_d * test_x_ids.shape[1]:] = cm_sketch_now.query_d_np(test_index_array).T

    '''test dataset'''
    test_x = np.log(1 + test_x)
    test_y = np.array(test_flows_data_list).reshape(-1, 1)
    test_cm = np.amin(cm_sketch_now.query_d_np(test_index_array).T, axis=1).reshape(-1, 1)
    test_cu = np.amin(cu_sketch_now.query_d_np(test_index_array).T, axis=1).reshape(-1, 1)
    test_count = np.median(count_sketch_now.query_d_np(test_index_array).T, axis=1).reshape(-1, 1)

    '''加载模型'''
    model_path = 'dnn_d_params_' + str(max_conf) + '.pkl'
    model_object = DNN1(input_size=test_x.shape[1], deep_l=4 - 2)
    model_object.load_state_dict(torch.load(model_path))

    '''预测'''
    start_time = time.time()
    predictions = model_object(torch.tensor(test_x, dtype=torch.float))
    pre_arr = predictions.detach().numpy()
    test_dnn = np.round(np.exp(pre_arr))
    '''性能评估'''
    dnn_metrics = Metrics(real_val=test_y, pre_val=test_dnn)
    cm_metrics = Metrics(real_val=test_y, pre_val=test_cm)
    cu_metrics = Metrics(real_val=test_y, pre_val=test_cu)
    count_metrics = Metrics(real_val=test_y, pre_val=test_count)

    dnn_metrics.get_allval()
    cm_metrics.get_allval()
    cu_metrics.get_allval()
    count_metrics.get_allval()

    # print("DNN")
    # print("ARE:", dnn_metrics.ARE_val)
    # print("AAE:", dnn_metrics.AAE_val)
    # print("F1:", dnn_metrics.F1_val)
    # print("Heavyhitter F1:", dnn_metrics.F1_val_hh)
    #
    # print("CM")
    # print("ARE:", cm_metrics.ARE_val)
    # print("AAE:", cm_metrics.AAE_val)
    # print("F1:", cm_metrics.F1_val)
    # print("Heavyhitter F1:", cm_metrics.F1_val_hh)
    #
    # print("CU")
    # print("ARE:", cu_metrics.ARE_val)
    # print("AAE:", cu_metrics.AAE_val)
    # print("F1:", cu_metrics.F1_val)
    # print("Heavyhitter F1:", cu_metrics.F1_val_hh)
    #
    # print("Count")
    # print("ARE:", count_metrics.ARE_val)
    # print("AAE:", count_metrics.AAE_val)
    # print("F1:", count_metrics.F1_val)
    # print("Heavyhitter F1:", count_metrics.F1_val_hh)

    end_time = time.time()
    # print("running time:", end_time - start_time)
    return dnn_metrics, cm_metrics, cu_metrics, count_metrics

#%%
def draws(x_list,y_np,x_label,real=0):
    if real:
        folder_path = "./zipf_testdata_set/results/wide"
    else:
        folder_path = "./zipf_testdata_set/results/zipf"
    # 创建画布和子图
    row_num=2
    col_num=3
    fig, axes = plt.subplots(row_num, col_num, figsize=(16, 8))
    sketchs = ["DNN", "CM", "CU", "Count"]
    titles = ["ARE", "AAE", "F1", "F1(HeavyHitter)","WMRE","RE"]
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
            # 遍历四条曲线
            for k in range(4):
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
    # memory_list = [1, 2, 4, 8, 12, 16, 20, 24, 32]
    w0=1*10**4
    flows_folder="./testdata_set_5s/testdata_flows_5s"
    plt.ion()
#%%
    memory_list = [4, 8, 12, 16, 24, 32]
    plots_np=np.full((4,len(memory_list),metrics_num,test_nums),0,dtype=float)#类型（sketch种类）k，横坐标i，纵坐标类型j,测试用例数量l
    for i in range(len(memory_list)):#变化内存
        w=memory_list[i]*w0
        for l in range(test_nums):
            file_now = random.randint(0, 180 - 1)
            test_flows_path = flows_folder+"/20240103_5s_" + str(file_now).zfill(5) + ".txt"
            list_metrics=solution_real(w=w,flows_path=test_flows_path,file_now=file_now)
            for k in range(4):
                plots_np[k, i, 0, l]=list_metrics[k].ARE_val
                plots_np[k, i, 1, l] = list_metrics[k].AAE_val
                plots_np[k, i, 2, l] = list_metrics[k].F1_val
                plots_np[k, i, 3, l] = list_metrics[k].F1_val_hh
                plots_np[k, i, 4, l] = list_metrics[k].WMRE_val
                plots_np[k, i, 5, l] = list_metrics[k].RE_val
#%%
    draws(memory_list,plots_np,"w(10^4)",real=1)
# #%%
#     memory_list = [4, 8, 12, 16, 24, 32]
#     plots_np = np.full((4, len(memory_list), metrics_num, test_nums), 0, dtype=float)  # 类型（sketch种类）k，横坐标i，纵坐标类型j,测试用例数量l
#     for i in range(len(memory_list)):  # 变化内存
#         w = memory_list[i] * w0
#         for l in range(test_nums):
#             test_flows_path = "./zipf_testdata_set/testdata_flows/20_0160000_" + str(l).zfill(2) + ".txt"
#             list_metrics = solution_zipf(w=w, flows_path=test_flows_path, alpha=2.0, n=160000,
#                                          zifp_sketch=True, numbering=l)
#             for k in range(4):
#                 plots_np[k, i, 0, l] = list_metrics[k].ARE_val
#                 plots_np[k, i, 1, l] = list_metrics[k].AAE_val
#                 plots_np[k, i, 2, l] = list_metrics[k].F1_val
#                 plots_np[k, i, 3, l] = list_metrics[k].F1_val_hh
#                 plots_np[k, i, 4, l] = list_metrics[k].WMRE_val
#                 plots_np[k, i, 5, l] = list_metrics[k].RE_val
# #%%
#     draws(memory_list, plots_np, "w(10^4)")
# #%%
#     # alpha_list=[1.2,1.5,2.0,2.5,3.0]
#     alpha_list=[1.01,2.2,2.4]
#     plots_np = np.full((4, len(alpha_list), metrics_num, test_nums), 0, dtype=float)  # 类型（sketch种类）k，横坐标i，纵坐标类型j,测试用例数量l
#     for i in range(len(alpha_list)):#变化zipf的α
#         for l in range(test_nums):
#             test_flows_path="./zipf_testdata_set/testdata_flows/"+str(int(alpha_list[i]*10))+"_0160000_"+str(l).zfill(2)+".txt"
#             list_metrics=solution_zipf(w=160000,flows_path=test_flows_path,alpha=alpha_list[i],n=160000,zifp_sketch=False,numbering=l)
#             for k in range(4):
#                 plots_np[k, i, 0, l]=list_metrics[k].ARE_val
#                 plots_np[k, i, 1, l] = list_metrics[k].AAE_val
#                 plots_np[k, i, 2, l] = list_metrics[k].F1_val
#                 plots_np[k, i, 3, l] = list_metrics[k].F1_val_hh
#                 plots_np[k, i, 4, l] = list_metrics[k].WMRE_val
#                 plots_np[k, i, 5, l] = list_metrics[k].RE_val
# #%%
#     draws(alpha_list,plots_np,"alpha")
# #%%
#     n_list = [40000, 80000, 160000, 320000, 640000]
#     plots_np = np.full((4, len(n_list), metrics_num, test_nums), 0, dtype=float)  # 类型（sketch种类）k，横坐标i，纵坐标类型j,测试用例数量l
#     for i in range(len(n_list)):  # 变化zipf的α
#         for l in range(test_nums):
#             test_flows_path = "./zipf_testdata_set/testdata_flows/"+"20_"+str(n_list[i]).zfill(7)+"_" + str(l).zfill(2) + ".txt"
#             list_metrics = solution_zipf(w=160000, flows_path=test_flows_path, alpha=2.0, n=n_list[i],zifp_sketch=True, numbering=l)
#             for k in range(4):
#                 plots_np[k, i, 0, l] = list_metrics[k].ARE_val
#                 plots_np[k, i, 1, l] = list_metrics[k].AAE_val
#                 plots_np[k, i, 2, l] = list_metrics[k].F1_val
#                 plots_np[k, i, 3, l] = list_metrics[k].F1_val_hh
#                 plots_np[k, i, 4, l] = list_metrics[k].WMRE_val
#                 plots_np[k, i, 5, l] = list_metrics[k].RE_val
# #%%
#     draws(n_list, plots_np, "n")
    plt.ioff()
    plt.show()
