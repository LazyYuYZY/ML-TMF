import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import random
import scipy.sparse as sp

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

'''CM Sketch类'''
class cm_sketch(object):
    def __init__(self, cm_d=3, cm_w=10 ** 5, flag=0, dict_cm={}, cm_sketch_load=np.full((3, 10 ** 5), 0)):
        self.flag = flag
        self.d = cm_d
        self.w = cm_w
        self.d_list = list(range(self.d))
        # 构造新的CM Sketch
        if self.flag == 0:
            """h(x) = (a*x + b) % p % w"""
            p_list = [6296197, 2254201, 7672057, 1002343, 9815713, 3436583, 4359587, 5638753, 8155451]
            selected_elements = random.sample(p_list, self.d)
            self.a = np.random.randint(low=10 ** 4, high=10 ** 5, size=(self.d, 1))
            self.b = np.random.randint(low=10 ** 4, high=10 ** 5, size=(self.d, 1))
            self.p = np.array(selected_elements).reshape(self.d, 1)
            self.offset = 10 ** 6 + 1
            self.Matrix = np.full((self.d, self.w), 0)  # cm存储的counter值
        else:  # 导入CM Sketch
            self.a = np.array(dict_cm['a'])
            self.b = np.array(dict_cm['b'])
            self.p = np.array(dict_cm['p'])
            self.offset = dict_cm['offset']
            self.Matrix = cm_sketch_load  # cm存储的counter值

    def insert_list(self, flow_list):

        for x, flow_num in enumerate(flow_list):
            x = x + self.offset
            h = (self.a * x + self.b) % self.p % self.w
            h_list = h.reshape(1, self.d).tolist()
            self.Matrix[self.d_list, h_list[0]] = self.Matrix[self.d_list, h_list[0]] + flow_num

    def insert_dict(self, flow_dict):

        for x, flow_num in flow_dict.items():
            x = x + self.offset
            h = (self.a * x + self.b) % self.p % self.w
            h_list = h.reshape(1, self.d).tolist()
            self.Matrix[self.d_list, h_list[0]] = self.Matrix[self.d_list, h_list[0]] + flow_num

    # 获取d个hash的查询值
    def query_d(self, five_turpe_list):
        x = np.array(five_turpe_list)
        x = x + self.offset
        x = x.reshape(1, -1)
        d_query_result = np.full((self.d, len(five_turpe_list)), 0)
        h = (self.a * x + self.b) % self.p % self.w
        for d in range(self.d):
            d_query_result[d] = self.Matrix[d][h[d]]

        return d_query_result

    def query_d_np(self, five_turpe_np):
        x = five_turpe_np
        x = x + self.offset
        x = x.reshape(1, -1)
        d_query_result = np.full((self.d, x.shape[1]), 0)
        h = (self.a * x + self.b) % self.p % self.w
        for d in range(self.d):
            d_query_result[d] = self.Matrix[d][h[d]]

        return d_query_result

    def query_one(self, five_tuple):
        x = five_tuple
        x = x + self.offset
        h = (self.a * x + self.b) % self.p % self.w
        h_list = h.reshape(1, self.d).tolist()
        d_query_result = self.Matrix[self.d_list, h_list[0]]
        oneflow_result = min(d_query_result)
        return oneflow_result

    # 所有流在cm的查询值 list实现
    def query_all_list(self, flow_list):
        flows_query_list = []
        for key in range(len(flow_list)):
            flows_query_list.append(self.query_one(key))
        return flows_query_list

    # 清空counter
    def clear(self):
        self.Matrix = np.full((self.d, self.w), 0)  # cm存储的counter值

    def save(self, file_name):
        dict_load = {'a': self.a.tolist(), 'b': self.b.tolist(), 'p': self.p.tolist(), 'offset': self.offset}
        rw_files.write_dict(file_name, dict_load)


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

'''ranges'''
CORA_TRAIN_RANGE=[0,100000]
CORA_VAL_RANGE=[100000,110000]
CORA_TEST_RANGE=[110000,120000]
idx_train = torch.arange(CORA_TRAIN_RANGE[0], CORA_TRAIN_RANGE[1], dtype=torch.long, device=device)
idx_val = torch.arange(CORA_VAL_RANGE[0], CORA_VAL_RANGE[1], dtype=torch.long, device=device)
idx_test = torch.arange(CORA_TEST_RANGE[0], CORA_TEST_RANGE[1], dtype=torch.long, device=device)

def conflict_all(index_np, dict_cm, w=22 * 10 ** 4, d=3):
    a = np.array(dict_cm['a'])
    b = np.array(dict_cm['b'])
    p = np.array(dict_cm['p'])
    offset = dict_cm['offset']
    cm_d_id = (a * (index_np + offset) + b) % p % w
    # data_x_d = np.full((index_np.shape[0], d * max_conf), -1)
    edge_index_cm=[]
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
                for flow_j in conf_flows_id_list:
                    edge_index_cm.append([flow_j,conf_flow_id])

    return torch.tensor(edge_index_cm,dtype=torch.float)

'''加载cm图信息'''
def load_cm_data(w=1* 10 ** 4,flows_path="",file_now=0):
    '''流数据集'''
    test_flows_data = rw_files.get_dict(flows_path)
    test_flows_data_list = list(test_flows_data.values())

    '''处理cm数据'''
    file_name = "D:/大文件/DNN_d_complete/testdata_set_5s/" + str(w).zfill(6) + "_cm_sketch.txt"
    dict_cm = rw_files.get_dict(file_name)
    cm_d = 3
    cm_w = dict_cm['w']
    cm_test_path = "D:/大文件/DNN_d_complete/testdata_set_5s/testdata_" + str(cm_w).zfill(
        6) + "_cm_5s/20240103_5s_" + str(file_now).zfill(5) + ".txt"
    cm_sketch_load = np.loadtxt(cm_test_path)
    cm_sketch_now = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_load)

    '''获取test时刻dnn_cm的输入,即冲突流的id'''
    flows_index_path = "D:/大文件/DNN_d_complete/testdata_set_5s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flows_index_path)
    get_value = np.vectorize(lambda x: flows_alltime_dict.get(x, 0))
    test_x_key = list(test_flows_data.keys())
    test_index_array = get_value(np.array(test_x_key))

    node_features = cm_sketch_now.query_d_np(test_index_array).T

    '''test dataset'''
    node_features = np.log(1 + node_features)
    node_features = torch.tensor(node_features,dtype=torch.float)
    test_y = np.array(test_flows_data_list).reshape(-1, 1)
    test_cm = np.amin(cm_sketch_now.query_d_np(test_index_array).T, axis=1).reshape(-1, 1)

    node_labels=torch.tensor(test_y,dtype=torch.float)
    edge_index=conflict_all(index_np=test_index_array, dict_cm=dict_cm,w=cm_w, d=cm_d)

    return node_features, node_labels, edge_index.t()

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class GATLayer(nn.Module):
    """GAT层"""

    def __init__(self, input_feature, output_feature, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.a = nn.Parameter(torch.empty(size=(2 * output_feature, 1)))
        self.w = nn.Parameter(torch.empty(size=(input_feature, output_feature)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.w)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # adj>0的位置使用e对应位置的值替换，其余都为-9e15，这样设定经过Softmax后每个节点对应的行非邻居都会变为0。
        attention = F.softmax(attention, dim=1)  # 每行做Softmax，相当于每个节点做softmax
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.mm(attention, Wh)  # 得到下一层的输入

        if self.concat:
            return F.elu(h_prime)  # 激活
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):

        Wh1 = torch.matmul(Wh, self.a[:self.output_feature, :])  # N*out_size @ out_size*1 = N*1

        Wh2 = torch.matmul(Wh, self.a[self.output_feature:, :])  # N*1

        e = Wh1 + Wh2.T  # Wh1的每个原始与Wh2的所有元素相加，生成N*N的矩阵
        return self.leakyrelu(e)


class GAT(nn.Module):
    """GAT模型"""


    def __init__(self, input_size, hidden_size, output_size, dropout, alpha, nheads, concat=True):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attention = [GATLayer(input_size, hidden_size, dropout=dropout, alpha=alpha, concat=True) for _ in
                          range(nheads)]
        for i, attention in enumerate(self.attention):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATLayer(hidden_size * nheads, output_size, dropout=dropout, alpha=alpha, concat=False)


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attention], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

        return F.log_softmax(x, dim=1)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    model.eval()
    output = model(features, adj)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--hidden', type=int, default=8, help='hidden size')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--nheads', type=int, default=8, help='Number of head attentions')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--seed', type=int, default=17, help='Seed number')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    flows_folder = "D:/大文件/DNN_d_complete/testdata_set_5s/testdata_flows_5s"
    file_now = random.randint(0, 180 - 1)
    test_flows_path = flows_folder + "/20240103_5s_" + str(file_now).zfill(5) + ".txt"
    features, labels,adj= load_cm_data(w=20*10**4,flows_path=test_flows_path)

    model = GAT(input_size=features.shape[1], hidden_size=args.hidden, output_size=int(labels.max()) + 1,
                dropout=args.dropout, nheads=8, alpha=args.alpha)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = 1000 + 1
    best_epoch = 0

    for epoch in range(1000):
        loss_values.append(train(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    compute_test()