import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

# 导入自定义模块（请根据实际路径调整）
from GNN import GNNWithDNN
from sketchs import *        # 包含 cm_sketch 类及 rw_files 工具
from Metrics import *        # 评估指标类

# ---------- 日志重定向 ----------
class Logger:
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# ---------- 全局配置 ----------
P_FLOWDROP = 0.01
MAX_CONF = 2                     # 每个节点最大冲突邻居数（与 DNN 一致）

def build_graph(index_np, cm_sketch, dict_cm, max_conf=MAX_CONF):
    """
    根据 CM Sketch 的冲突关系构建图
    """
    n_nodes = len(index_np)
    d = cm_sketch.d
    w = cm_sketch.w

    # 哈希参数（确保为一维数组）
    a = np.array(dict_cm['a']).flatten()
    b = np.array(dict_cm['b']).flatten()
    p = np.array(dict_cm['p']).flatten()
    offset = dict_cm['offset']

    # 计算哈希桶索引，形状 (n_nodes, d)
    hash_ids = ((a[:, np.newaxis] * (index_np + offset) + b[:, np.newaxis]) % p[:, np.newaxis] % w).T

    # 节点特征：d 维 CM 查询值
    node_feat = cm_sketch.query_d_np(index_np).T   # (n_nodes, d)

    # 构建边列表
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

    data = Data(x=torch.tensor(node_feat, dtype=torch.float),
                edge_index=edge_index)
    return data

def select_indices(index_array, p):
    num_rows = index_array.shape[0]
    num_selected = int(num_rows * p)
    selected = np.random.choice(num_rows, size=num_selected, replace=False)
    return selected

# ---------- 主程序 ----------
if __name__ == '__main__':
    # 日志设置
    log_path = '../Logs/'
    os.makedirs(log_path, exist_ok=True)
    log_file_name = log_path + 'log-gnn-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    sys.stdout = Logger(log_file_name)
    sys.stderr = Logger(log_file_name)

    # 随机选择两个文件作为训练集和测试集
    file_now = random.randint(0, 9)
    file_now1 = file_now
    while file_now1 == file_now:
        file_now1 = random.randint(0, 9)

    # 数据路径（请根据实际情况修改）
    test_y_path = f"../traindata_set_5s/traindata_flows_5s/{str(file_now).zfill(5)}.txt"
    train_y_path = f"../traindata_set_5s/traindata_flows_5s/{str(file_now1).zfill(5)}.txt"

    # 加载 CM Sketch 参数
    dict_cm = rw_files.get_dict("../sketch_params/160000_cm_sketch.txt")
    cm_d = 3
    cm_w = dict_cm['w']

    # 加载训练集 CM Sketch
    cm_train_path = f"../traindata_set_5s/traindata_160000_cm_5s/{str(file_now1).zfill(5)}.txt"
    cm_train_load = np.loadtxt(cm_train_path)
    sketch_train = cm_sketch(cm_d=3, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_train_load)

    # 加载测试集 CM Sketch
    cm_test_path = f"../traindata_set_5s/traindata_160000_cm_5s/{str(file_now).zfill(5)}.txt"
    cm_test_load = np.loadtxt(cm_test_path)
    sketch_test = cm_sketch(cm_d=3, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_test_load)

    # 加载真实流量大小
    train_flows = rw_files.get_dict(train_y_path)
    test_flows = rw_files.get_dict(test_y_path)

    # 流索引映射（全局 ID -> 出现顺序 ID）
    flows_index_dict = rw_files.get_dict("../traindata_set_5s/flow_index.txt")
    get_value = np.vectorize(lambda x: flows_index_dict.get(x, 0))

    # ---------- 构建训练图 ----------
    train_keys = list(train_flows.keys())
    train_idx = get_value(np.array(train_keys))
    selected_train = select_indices(train_idx, 1 - P_FLOWDROP)
    train_idx_sel = train_idx[selected_train]

    train_graph = build_graph(train_idx_sel, sketch_train, dict_cm, max_conf=MAX_CONF)
    train_y = np.log(np.array(list(train_flows.values()))[selected_train].reshape(-1, 1))
    train_graph.y = torch.tensor(train_y, dtype=torch.float)

    # ---------- 构建测试图 ----------
    test_keys = list(test_flows.keys())
    test_idx = get_value(np.array(test_keys))
    selected_test = select_indices(test_idx, 1 - P_FLOWDROP)
    test_idx_sel = test_idx[selected_test]

    test_graph = build_graph(test_idx_sel, sketch_test, dict_cm, max_conf=MAX_CONF)
    test_y = np.log(np.array(list(test_flows.values()))[selected_test].reshape(-1, 1))
    test_graph.y = torch.tensor(test_y, dtype=torch.float)

    # 纯 CM 基线误差
    cm_queries = sketch_test.query_d_np(test_idx_sel)
    cm_estimates = np.min(cm_queries, axis=0).reshape(-1, 1)
    real_values = np.array(list(test_flows.values()))[selected_test].reshape(-1, 1)
    cm_are = np.mean(np.abs(cm_estimates / real_values - 1))
    print(f"CM Baseline ARE: {cm_are:.6f}")

    # ---------- 训练配置 ----------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 超参数搜索空间（可自定义）
    gnn_layer_list = [4]
    gnn_hidden_list = [128]
    gnn_type_list = ['gat']               # 也可尝试 'gcn'
    heads_list = [4]
    dropout_list = [0.0]
    dnn_hidden_list = [128]
    dnn_num_layers_list = [4]
    lr_list = [0.01]
    epoch_list = [500]

    for gnn_layers in gnn_layer_list:
        for gnn_hidden in gnn_hidden_list:
            for gnn_type in gnn_type_list:
                for heads in heads_list:
                    for dropout in dropout_list:
                        for dnn_hidden in dnn_hidden_list:
                            for dnn_layers in dnn_num_layers_list:
                                for lr in lr_list:
                                    for epochs in epoch_list:
                                        print(f"\n# GNN层数:{gnn_layers}, 类型:{gnn_type}, 隐藏维:{gnn_hidden}, heads:{heads}, "
                                              f"dropout:{dropout}, DNN隐藏:{dnn_hidden}, DNN层数:{dnn_layers}, lr:{lr}, epochs:{epochs} #")
                                        start_time = time.time()

                                        model = GNNWithDNN(
                                            in_dim=cm_d,
                                            gnn_hidden_dim=gnn_hidden,
                                            gnn_num_layers=gnn_layers,
                                            gnn_type=gnn_type,
                                            heads=heads,
                                            dropout=dropout,
                                            dnn_hidden_dim=dnn_hidden,
                                            dnn_num_layers=dnn_layers,
                                            output_dim=1
                                        ).to(device)

                                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                                        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
                                        loss_fn = nn.MSELoss()

                                        train_graph = train_graph.to(device)
                                        test_graph = test_graph.to(device)

                                        for epoch in range(epochs):
                                            model.train()
                                            optimizer.zero_grad()
                                            pred, _ = model((train_graph.x, train_graph.edge_index))
                                            loss = loss_fn(pred, train_graph.y)
                                            loss.backward()
                                            optimizer.step()
                                            scheduler.step()

                                            if (epoch + 1) % 100 == 1:
                                                model.eval()
                                                with torch.no_grad():
                                                    test_pred, _ = model((test_graph.x, test_graph.edge_index))
                                                    test_loss = loss_fn(test_pred, test_graph.y).item()
                                                print(f"Epoch {epoch+1:3d} | Train Loss: {loss.item():.6f} | Test Loss: {test_loss:.6f}")

                                        # 最终评估
                                        model.eval()
                                        with torch.no_grad():
                                            final_pred, _ = model((test_graph.x, test_graph.edge_index))
                                            pred_exp = np.round(np.exp(final_pred.cpu().numpy()))
                                        true_exp = np.exp(test_y)
                                        print(np.max(pred_exp))

                                        gnn_are = np.mean(np.abs(pred_exp / true_exp - 1))
                                        print(f"GNN ARE: {gnn_are:.6f}")
                                        print(f"Max ARE: {np.max(np.abs(pred_exp / true_exp - 1)):.6f}")
                                        print(f"Training time: {time.time() - start_time:.2f}s")

                                        # 可选：保存模型
                                        torch.save(model.state_dict(), f"cm_gnn_params_{MAX_CONF}.pkl")