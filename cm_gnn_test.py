import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import matplotlib.pyplot as plt
import numpy as np
import torch

# 导入自定义模型和工具
from GNN.GNN import GNNWithDNN          # 之前定义的 GNN + DNN 模型
from DNN1.DNN1 import DNN1,DNN0                # 已有的 DNN 模型
from GAT.GAT_model import GAT             # 已有的 GAT 模型
from sketchs import *
from Metrics import *

device = torch.device("cpu")
T_slice = 5
P_flowdrop = 0


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


def build_graph_gnn(index_np, dict_cm, cm_sketch, max_conf=2):
    """
    为 GNN 构建图数据（与 GAT 类似，但不添加虚拟节点，边仅基于冲突关系）
    返回 (node_features, edge_index)
    """
    a = np.array(dict_cm['a']).flatten()
    b = np.array(dict_cm['b']).flatten()
    p = np.array(dict_cm['p']).flatten()
    offset = dict_cm['offset']
    d = len(a)
    w = cm_sketch.w

    # 计算哈希桶索引
    hash_ids = ((a[:, np.newaxis] * (index_np + offset) + b[:, np.newaxis]) % p[:, np.newaxis] % w).T
    n_nodes = len(index_np)

    # 节点特征：log(1 + CM查询值)
    node_features = cm_sketch.query_d_np(index_np).T
    node_features = np.log(1 + node_features)

    # 构建边
    edge_list = []
    for row in range(d):
        bucket_dict = {}
        for node_idx, bucket in enumerate(hash_ids[:, row]):
            bucket_dict.setdefault(bucket, []).append(node_idx)
        for bucket, nodes in bucket_dict.items():
            if len(nodes) < 2:
                continue
            for i, u in enumerate(nodes):
                others = [v for v in nodes if v != u]
                if len(others) > max_conf:
                    others = np.random.choice(others, max_conf, replace=False)
                for v in others:
                    edge_list.append([u, v])

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(node_features, dtype=torch.float), edge_index

# -------------------- 修改后的测试函数 --------------------
def solution_real(w=1*10**4, flows_path="", file_now=0, real_sketch=False,
                  dnn_conf_num=2, gat_conf_num=3, gnn_conf_num=2):
    """
    真实数据集测试，返回四个模型的 Metrics：GAT, DNN, GNN, CM
    """
    # 模型文件名（可根据训练配置调整）
    dnn_pkl = "cm_dnn_d_params_" + str(dnn_conf_num) + ".pkl"
    gat_pkl = "cm_gat_d_params_" + str(gat_conf_num) + ".pkl"
    gnn_pkl = "cm_gnn_params_" + str(gnn_conf_num) + ".pkl"    # GNN 训练保存的权重

    '''流数据集'''
    test_flows_data = rw_files.get_dict(flows_path)
    test_flows_data_list = list(test_flows_data.values())

    '''流索引映射'''
    flow_index_path = "./testdata_set_" + str(T_slice) + "s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flow_index_path)
    flows_data_onetime = {}
    for one_flow in test_flows_data.keys():
        flows_data_onetime[flows_alltime_dict[one_flow]] = test_flows_data[one_flow]

    '''CM Sketch 参数与数据加载'''
    file_name = "./sketch_params/" + str(w).zfill(6) + "_cm_sketch.txt"
    dict_cm = rw_files.get_dict(file_name)
    cm_d = 3
    cm_w = dict_cm['w']
    new_dir = "./testdata_set_" + str(T_slice) + "s/testdata_" + str(cm_w).zfill(6) + "_cm_" + str(T_slice) + "s"
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    cm_test_path = new_dir + "/" + str(file_now).zfill(5) + ".txt"
    if real_sketch:
        cm_sketch_load = np.loadtxt(cm_test_path)
        cm_sketch_now = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_load)
    else:
        cm_sketch_load = np.full((cm_d, cm_w), 0)
        cm_sketch_now = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_load)
        cm_sketch_now.insert_dict(flows_data_onetime)
        cm_sketch_load = cm_sketch_now.Matrix
        np.savetxt(cm_test_path, cm_sketch_load, fmt='%d')

    '''获取测试流索引'''
    flows_index_path = "./testdata_set_" + str(T_slice) + "s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flows_index_path)
    get_value = np.vectorize(lambda x: flows_alltime_dict.get(x, 0))
    test_x_key = list(test_flows_data.keys())
    test_index_array = get_value(np.array(test_x_key))

    selected_indices = select_indices(test_index_array, 1 - P_flowdrop)
    test_index_array = test_index_array[selected_indices]

    # ---------- DNN 数据准备 ----------
    test_x_ids = conflict_dnn(index_np=test_index_array, dict_cm=dict_cm, w=cm_w, d=cm_d, max_conf=dnn_conf_num)
    test_x = np.full((test_x_ids.shape[0], cm_d * (1 + test_x_ids.shape[1])), 0)
    for i in range(cm_d * dnn_conf_num):
        conf_flows_id = np.where(test_x_ids[:, i] != -1)[0]
        test_x[conf_flows_id, i * cm_d:(i + 1) * cm_d] = cm_sketch_now.query_d_np(test_x_ids[conf_flows_id, i]).T
    test_x[:, cm_d * test_x_ids.shape[1]:] = cm_sketch_now.query_d_np(test_index_array).T
    test_x = np.log(1 + test_x)
    test_y = np.array(test_flows_data_list).reshape(-1, 1)[selected_indices]
    test_cm = np.amin(cm_sketch_now.query_d_np(test_index_array).T, axis=1).reshape(-1, 1)

    # ---------- 加载 DNN 模型并预测 ----------
    dnn_model = DNN1(input_size=test_x.shape[1], deep_l=2, num=100).to(device)
    dnn_model.load_state_dict(torch.load(dnn_pkl, map_location=device))
    dnn_model.eval()
    with torch.no_grad():
        dnn_pred = dnn_model(torch.tensor(test_x, dtype=torch.float, device=device)).cpu().numpy()
    test_dnn = np.round(np.exp(dnn_pred))

    # ---------- 加载 GAT 模型并预测 ----------
    gat_model = GAT(
        num_of_layers=3, num_heads_per_layer=[4,4,1], num_features_per_layer=[3,9,27,81],
        add_skip_connection=True, bias=True, dropout=0.0, layer_type=2,
        log_attention_weights=False, dnn_layer_num=4
    ).to(device)
    gat_model.load_state_dict(torch.load(gat_pkl, map_location=device))
    gat_model.eval()

    node_features_gat = cm_sketch_now.query_d_np(test_index_array).T
    dummy_node_features = np.zeros((gat_conf_num * cm_d, cm_d))
    node_features_gat = np.concatenate((node_features_gat, dummy_node_features), axis=0)
    node_features_gat = np.log(1 + node_features_gat)
    node_features_gat = torch.tensor(node_features_gat, dtype=torch.float, device=device)
    edge_index_gat = conflict_gat(index_np=test_index_array, dict_cm=dict_cm, w=cm_w, d=cm_d, max_conf=gat_conf_num).t().to(device)
    with torch.no_grad():
        gat_pred = gat_model((node_features_gat, edge_index_gat))[0]
    test_gat = np.round(np.exp(gat_pred[:-cm_d * gat_conf_num]))

    # ---------- 加载 GNN 模型并预测 ----------
    gnn_model = GNNWithDNN(
        in_dim=cm_d, gnn_hidden_dim=128, gnn_num_layers=4, gnn_type='gat', heads=4,
        dropout=0.0, dnn_hidden_dim=128, dnn_num_layers=4, output_dim=1
    ).to(device)
    gnn_model.load_state_dict(torch.load(gnn_pkl, map_location=device))
    gnn_model.eval()

    node_features_gnn, edge_index_gnn = build_graph_gnn(test_index_array, dict_cm, cm_sketch_now, max_conf=gnn_conf_num)
    node_features_gnn = node_features_gnn.to(device)
    edge_index_gnn = edge_index_gnn.to(device)
    with torch.no_grad():
        gnn_pred, _ = gnn_model((node_features_gnn, edge_index_gnn))
        gnn_pred = gnn_pred.cpu().numpy()
    test_gnn = np.round(np.exp(gnn_pred))

    # ---------- 评估 ----------
    gat_metrics  = Metrics(real_val=test_y, pre_val=np.asarray(test_gat).reshape(-1, 1))
    dnn_metrics  = Metrics(real_val=test_y, pre_val=np.asarray(test_dnn).reshape(-1, 1))
    gnn_metrics  = Metrics(real_val=test_y, pre_val=np.asarray(test_gnn).reshape(-1, 1))
    cm_metrics   = Metrics(real_val=test_y, pre_val=np.asarray(test_cm).reshape(-1, 1))
    gnn_metrics.get_allval()
    print("GNN ARE:",gnn_metrics.ARE_val)
    gat_metrics.get_allval()
    print("GAT ARE:", gat_metrics.ARE_val)

    for m in [gat_metrics, dnn_metrics, gnn_metrics, cm_metrics]:
        m.get_allval()

    return gat_metrics, dnn_metrics, gnn_metrics, cm_metrics


def solution_zipf(w=1*10**4, flows_path="", alpha=2.0, n=160000, zifp_sketch=False,
                  numbering=0, dnn_conf_num=2, gat_conf_num=3, gnn_conf_num=2):
    """
    Zipf 合成数据集测试，返回四个模型的 Metrics：GAT, DNN, GNN, CM
    """
    dnn_pkl = "dnn_d_params_" + str(dnn_conf_num) + ".pkl"
    gat_pkl = "gat_d_params_" + str(gat_conf_num) + ".pkl"
    gnn_pkl = "gnn_d_params_" + str(gnn_conf_num) + ".pkl"

    '''流数据集'''
    test_flows_data_str = rw_files.get_dict(flows_path)
    test_flows_data = {int(k): v for k, v in test_flows_data_str.items()}
    test_flows_data_list = list(test_flows_data.values())

    '''CM Sketch'''
    file_name = "./sketch_params/" + str(w).zfill(6) + "_cm_sketch.txt"
    dict_cm = rw_files.get_dict(file_name)
    cm_d = 3
    cm_w = dict_cm['w']
    cm_test_path = "./zipf_testdata_set/cm/" + str(cm_w).zfill(6) + "_" + str(int(alpha*10)) + "_" + str(n).zfill(7) + "_" + str(numbering).zfill(2) + ".txt"
    if zifp_sketch:
        cm_sketch_load = np.loadtxt(cm_test_path)
        cm_sketch_now = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_load)
    else:
        cm_sketch_load = np.full((cm_d, cm_w), 0)
        cm_sketch_now = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_load)
        cm_sketch_now.insert_dict(test_flows_data)
        cm_sketch_load = cm_sketch_now.Matrix
        np.savetxt(cm_test_path, cm_sketch_load, fmt='%d')

    test_index_array = np.array(list(test_flows_data.keys())[1000:])

    # ---------- DNN ----------
    test_x_ids = conflict_dnn(index_np=test_index_array, dict_cm=dict_cm, w=cm_w, d=cm_d, max_conf=dnn_conf_num)
    test_x = np.full((test_x_ids.shape[0], cm_d * (1 + test_x_ids.shape[1])), 0)
    for i in range(cm_d * dnn_conf_num):
        conf_flows_id = np.where(test_x_ids[:, i] != -1)[0]
        test_x[conf_flows_id, i * cm_d:(i + 1) * cm_d] = cm_sketch_now.query_d_np(test_x_ids[conf_flows_id, i]).T
    test_x[:, cm_d * test_x_ids.shape[1]:] = cm_sketch_now.query_d_np(test_index_array).T
    test_x = np.log(1 + test_x)
    test_y = np.array(test_flows_data_list).reshape(-1, 1)
    test_cm = np.amin(cm_sketch_now.query_d_np(test_index_array).T, axis=1).reshape(-1, 1)

    dnn_model = DNN0(input_size=test_x.shape[1], deep_l=2).to(device)
    dnn_model.load_state_dict(torch.load(dnn_pkl, map_location=device))
    dnn_model.eval()
    with torch.no_grad():
        dnn_pred = dnn_model(torch.tensor(test_x, dtype=torch.float, device=device)).cpu().numpy()
    test_dnn = np.round(np.exp(dnn_pred))

    # ---------- GAT ----------
    gat_model = GAT(
        num_of_layers=3, num_heads_per_layer=[4,4,1], num_features_per_layer=[3,9,27,81],
        add_skip_connection=True, bias=True, dropout=0.0, layer_type=2,
        log_attention_weights=False, dnn_layer_num=4
    ).to(device)
    gat_model.load_state_dict(torch.load(gat_pkl, map_location=device))
    gat_model.eval()

    node_features_gat = cm_sketch_now.query_d_np(test_index_array).T
    dummy_node_features = np.zeros((gat_conf_num * cm_d, cm_d))
    node_features_gat = np.concatenate((node_features_gat, dummy_node_features), axis=0)
    node_features_gat = np.log(1 + node_features_gat)
    node_features_gat = torch.tensor(node_features_gat, dtype=torch.float, device=device)
    edge_index_gat = conflict_gat(index_np=test_index_array, dict_cm=dict_cm, w=cm_w, d=cm_d, max_conf=gat_conf_num).t().to(device)
    with torch.no_grad():
        gat_pred = gat_model((node_features_gat, edge_index_gat))[0].cpu().numpy()
    test_gat = np.round(np.exp(gat_pred[:-cm_d * gat_conf_num]))

    # ---------- GNN ----------
    gnn_model = GNNWithDNN(
        in_dim=cm_d, gnn_hidden_dim=64, gnn_num_layers=2, gnn_type='gat', heads=4,
        dropout=0.0, dnn_hidden_dim=128, dnn_num_layers=2, output_dim=1
    ).to(device)
    gnn_model.load_state_dict(torch.load(gnn_pkl, map_location=device))
    gnn_model.eval()

    node_features_gnn, edge_index_gnn = build_graph_gnn(test_index_array, dict_cm, cm_sketch_now, max_conf=gnn_conf_num)
    node_features_gnn = node_features_gnn.to(device)
    edge_index_gnn = edge_index_gnn.to(device)
    with torch.no_grad():
        gnn_pred, _ = gnn_model((node_features_gnn, edge_index_gnn))
        gnn_pred = gnn_pred.cpu().numpy()
    test_gnn = np.round(np.exp(gnn_pred))

    # ---------- 评估 ----------
    gat_metrics  = Metrics(real_val=test_y, pre_val=test_gat)
    dnn_metrics  = Metrics(real_val=test_y, pre_val=test_dnn)
    gnn_metrics  = Metrics(real_val=test_y, pre_val=test_gnn)
    cm_metrics   = Metrics(real_val=test_y, pre_val=test_cm)

    print("GNN ARE:",gnn_metrics.ARE_val)

    for m in [gat_metrics, dnn_metrics, gnn_metrics, cm_metrics]:
        m.get_allval()

    return gat_metrics, dnn_metrics, gnn_metrics, cm_metrics


# -------------------- 主程序：变化内存实验 --------------------
if __name__ == '__main__':
    w0 = 1 * 10**4
    test_nums = 10
    metrics_num = 6
    sketchs = ["GAT", "DNN", "GNN", "CM"]
    memory_list = [8, 12, 16, 24, 32]   # 单位为 10^4，即 80K, 120K, 160K, 240K, 320K

    # 注意第一维大小为 4（四种方法）
    plots_np = np.full((4, len(memory_list), metrics_num, test_nums), 0, dtype=float)

    for i, mem in enumerate(memory_list):
        w = int(mem * w0)
        for l in range(test_nums):
            T_slice = 5
            file_now = l
            test_flows_path = "./testdata_set_" + str(T_slice) + "s/testdata_flows_" + str(T_slice) + "s/" + str(file_now).zfill(5) + ".txt"

            print(f"Memory: {mem}*10^4, w={w}, file={file_now}")
            gat_m, dnn_m, gnn_m, cm_m = solution_real(
                w=w, flows_path=test_flows_path, file_now=file_now,
                real_sketch=False, dnn_conf_num=2, gat_conf_num=3, gnn_conf_num=2
            )

            for k, m in enumerate([gat_m, dnn_m, gnn_m, cm_m]):
                plots_np[k, i, 0, l] = m.ARE_val
                plots_np[k, i, 1, l] = m.AAE_val
                plots_np[k, i, 2, l] = m.F1_val
                plots_np[k, i, 3, l] = m.F1_val_hh
                plots_np[k, i, 4, l] = m.WMRE_val
                plots_np[k, i, 5, l] = m.RE_val

        # 保存每个内存配置下的各方法结果
        for k in range(4):
            results_path = "./results/wide/Memory/" + sketchs[k] + "_" + str(w) + ".txt"
            np.savetxt(results_path, plots_np[k, i, :, :])

    # 绘图（需确保 draws 函数支持4条曲线）
    # draws(memory_list, plots_np, "w(10^4)", real=1)
    print("All experiments finished.")