import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


from GAT_model import GAT
from utils.data_loading import load_graph_data
from utils.constants import *
import utils.utils as utils


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

flow_size_num=10

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
# CORA_TEST_RANGE=[110000,120000]
CORA_TEST_RANGE=[0,120000]
train_indices = torch.arange(CORA_TRAIN_RANGE[0], CORA_TRAIN_RANGE[1], dtype=torch.long, device=device)
val_indices = torch.arange(CORA_VAL_RANGE[0], CORA_VAL_RANGE[1], dtype=torch.long, device=device)
test_indices = torch.arange(CORA_TEST_RANGE[0], CORA_TEST_RANGE[1], dtype=torch.long, device=device)

# Simple decorator function so that I don't have to pass arguments that don't change from epoch to epoch
def get_main_loop(config, gat, cross_entropy_loss, optimizer, node_features, node_labels, edge_index, train_indices, val_indices, test_indices, patience_period, time_start):

    node_dim = 0  # node axis

    train_labels = node_labels.index_select(node_dim, train_indices)
    val_labels = node_labels.index_select(node_dim, val_indices)
    test_labels = node_labels.index_select(node_dim, test_indices)

    # node_features shape = (N, FIN), edge_index shape = (2, E)
    graph_data = (node_features, edge_index)  # I pack data into tuples because GAT uses nn.Sequential which requires it

    def get_node_indices(phase):
        if phase == LoopPhase.TRAIN:
            return train_indices
        elif phase == LoopPhase.VAL:
            return val_indices
        else:
            return test_indices

    def get_node_labels(phase):
        if phase == LoopPhase.TRAIN:
            return train_labels
        elif phase == LoopPhase.VAL:
            return val_labels
        else:
            return test_labels

    def main_loop(phase, epoch=0):
        global BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT, writer

        # Certain modules behave differently depending on whether we're training the model or not.
        # e.g. nn.Dropout - we only want to drop model weights during the training.
        if phase == LoopPhase.TRAIN:
            gat.train()
        else:
            gat.eval()

        node_indices = get_node_indices(phase)
        gt_node_labels = get_node_labels(phase)  # gt stands for ground truth
        # gt_node_labels = torch.tensor(gt_node_labels,dtype=float)
        gt_node_labels=gt_node_labels.long()
        gt_node_labels = torch.squeeze(gt_node_labels)

        # Do a forwards pass and extract only the relevant node scores (train/val or test ones)
        # Note: [0] just extracts the node_features part of the data (index 1 contains the edge_index)
        # shape = (N, C) where N is the number of nodes in the split (train/val/test) and C is the number of classes
        nodes_unnormalized_scores = gat(graph_data)[0].index_select(node_dim, node_indices)

        # Example: let's take an output for a single node on Cora - it's a vector of size 7 and it contains unnormalized
        # scores like: V = [-1.393,  3.0765, -2.4445,  9.6219,  2.1658, -5.5243, -4.6247]
        # What PyTorch's cross entropy loss does is for every such vector it first applies a softmax, and so we'll
        # have the V transformed into: [1.6421e-05, 1.4338e-03, 5.7378e-06, 0.99797, 5.7673e-04, 2.6376e-07, 6.4848e-07]
        # secondly, whatever the correct class is (say it's 3), it will then take the element at position 3,
        # 0.99797 in this case, and the loss will be -log(0.99797). It does this for every node and applies a mean.
        # You can see that as the probability of the correct class for most nodes approaches 1 we get to 0 loss! <3
        loss = cross_entropy_loss(nodes_unnormalized_scores, gt_node_labels)

        if phase == LoopPhase.TRAIN:
            optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph (.grad fields)
            loss.backward()  # compute the gradients for every trainable weight in the computational graph
            optimizer.step()  # apply the gradients to weights

        # Calculate the main metric - accuracy

        # Finds the index of maximum (unnormalized) score for every node and that's the class prediction for that node.
        # Compare those to true (ground truth) labels and find the fraction of correct predictions -> accuracy metric.
        class_predictions = torch.argmax(nodes_unnormalized_scores, dim=-1)
        class_predictions = torch.unsqueeze(class_predictions, dim=1)

        accuracy = torch.sum(torch.eq(class_predictions, torch.unsqueeze(gt_node_labels,dim=1)).long()).item() / len(gt_node_labels)
        # accuracy=torch.mean(torch.abs(class_predictions-gt_node_labels)/gt_node_labels)
        #
        # Logging
        #

        if phase == LoopPhase.TRAIN:
            # Log metrics
            if config['enable_tensorboard']:
                writer.add_scalar('training_loss', loss.item(), epoch)
                writer.add_scalar('training_acc', accuracy, epoch)

            # Save model checkpoint
            if config['checkpoint_freq'] is not None and (epoch + 1) % config['checkpoint_freq'] == 0:
                ckpt_model_name = f'gat_{config["dataset_name"]}_ckpt_epoch_{epoch + 1}.pth'
                config['test_perf'] = -1
                # torch.save(utils.get_training_state(config, gat), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))

        elif phase == LoopPhase.VAL:
            # Log metrics
            if config['enable_tensorboard']:
                writer.add_scalar('val_loss', loss.item(), epoch)
                writer.add_scalar('val_acc', accuracy, epoch)

            # Log to console
            if config['console_log_freq'] is not None and epoch % config['console_log_freq'] == 0:
                print(f'GAT training: time elapsed= {(time.time() - time_start):.2f} [s] | epoch={epoch + 1} | val acc={accuracy}')

            # The "patience" logic - should we break out from the training loop? If either validation acc keeps going up
            # or the val loss keeps going down we won't stop
            if accuracy > BEST_VAL_PERF or loss.item() < BEST_VAL_LOSS:
                BEST_VAL_PERF = max(accuracy, BEST_VAL_PERF)  # keep track of the best validation accuracy so far
                BEST_VAL_LOSS = min(loss.item(), BEST_VAL_LOSS)  # and the minimal loss
                PATIENCE_CNT = 0  # reset the counter every time we encounter new best accuracy
            else:
                PATIENCE_CNT += 1  # otherwise keep counting

            if PATIENCE_CNT >= patience_period:
                raise Exception('Stopping the training, the universe has no more patience for this training.')

        else:
            return accuracy  # in the case of test phase we just report back the test accuracy

    return main_loop  # return the decorated function


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

def load_cm_data_zipf(w=16* 10 ** 4,flows_path="",alpha=1.4,n=160000,zifp_sketch=False,numbering=0):

    '''流数据集'''
    test_flows_data_str = rw_files.get_dict(flows_path)
    test_flows_data={}
    for key, value in test_flows_data_str.items():
        new_key = int(key)
        test_flows_data[new_key] = value
    test_flows_data_list = list(test_flows_data.values())

    '''处理cm数据'''
    file_name = "D:/大文件/DNN_d_complete/testdata_set_5s/" + str(w).zfill(6) + "_cm_sketch.txt"
    dict_cm = rw_files.get_dict(file_name)
    cm_d = 3
    cm_w = dict_cm['w']
    cm_test_path = "D:/大文件/DNN_d_complete/zipf_testdata_set/cm/" + str(cm_w).zfill(6) + "_" + str(int(alpha * 10)) + "_" + str(n).zfill(7) + "_" + str(numbering).zfill(2) + ".txt"
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
    test_index_array = np.array(list(test_flows_data.keys()))

    node_features = cm_sketch_now.query_d_np(test_index_array).T

    '''test dataset'''
    # node_features = np.log(1+node_features)
    node_features = torch.tensor(node_features, dtype=torch.float)
    test_y = np.array(test_flows_data_list).reshape(-1, 1)
    test_cm = np.amin(cm_sketch_now.query_d_np(test_index_array).T, axis=1).reshape(-1, 1)


    test_y[test_y > flow_size_num] = flow_size_num
    test_y = test_y - 1
    test_test_y = test_y[test_indices]
    test_cm[test_cm > flow_size_num] = flow_size_num
    test_cm = test_cm - 1
    test_test_cm = test_cm[test_indices]
    cm_y = test_test_cm - test_test_y
    node_labels = torch.tensor(test_y, dtype=torch.float)

    print("CM acc=", np.mean(cm_y == 0))
    print("all 1 acc=", np.mean(test_test_y == 0))
    print("all big acc=", np.mean(test_test_y == flow_size_num-1))

    # test_y=np.log(test_y)
    # node_labels = torch.tensor(test_y, dtype=torch.float)
    edge_index= conflict_all(index_np=test_index_array, dict_cm=dict_cm, w=cm_w, d=cm_d)

    return node_features, node_labels, edge_index.t()


'''加载cm图信息'''
def load_cm_data(w=16* 10 ** 4,flows_path="",file_now=0):
    '''流数据集'''
    test_flows_data = rw_files.get_dict(flows_path)
    test_flows_data_list = list(test_flows_data.values())

    '''处理cm数据'''
    file_name = "D:/大文件/DNN_d_complete/testdata_set_5s/" + str(w).zfill(6) + "_cm_sketch.txt"
    dict_cm = rw_files.get_dict(file_name)
    cm_d = 3
    cm_w = dict_cm['w']
    print("W=",cm_w)
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
    # node_features = np.log(1 + node_features)
    node_features = torch.tensor(node_features,dtype=torch.float)
    test_y = np.array(test_flows_data_list).reshape(-1, 1)
    test_cm = np.amin(cm_sketch_now.query_d_np(test_index_array).T, axis=1).reshape(-1, 1)

    test_y[test_y>flow_size_num]=flow_size_num
    test_y=test_y-1
    test_test_y=test_y[test_indices]
    test_cm[test_cm>flow_size_num]=flow_size_num
    test_cm=test_cm-1
    test_test_cm=test_cm[test_indices]
    cm_y=test_test_cm - test_test_y

    print("CM acc=", np.mean(cm_y==0))
    print("all 1 acc=",np.mean(test_test_y==0))
    node_labels=torch.tensor(test_y,dtype=torch.float)
    edge_index=conflict_all(index_np=test_index_array, dict_cm=dict_cm,w=cm_w, d=cm_d)

    return node_features, node_labels, edge_index.t()

def train_gat_cm(config,file_path):
    global BEST_VAL_PERF, BEST_VAL_LOSS

    # Step 1: load the graph data
    node_features, node_labels, edge_index =load_cm_data_zipf(flows_path=file_path)

    # Step 2: prepare the model
    gat = GAT(
        num_of_layers=config['num_of_layers'],
        num_heads_per_layer=config['num_heads_per_layer'],
        num_features_per_layer=config['num_features_per_layer'],
        add_skip_connection=config['add_skip_connection'],
        bias=config['bias'],
        dropout=config['dropout'],
        layer_type=config['layer_type'],
        log_attention_weights=False  # no need to store attentions, used only in playground.py for visualizations
    ).to(device)

    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(gat.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    main_loop = get_main_loop(
        config,
        gat,
        loss_fn,
        optimizer,
        node_features,
        node_labels,
        edge_index,
        train_indices,
        val_indices,
        test_indices,
        config['patience_period'],
        time.time())

    BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT = [0, 0, 0]  # reset vars used for early stopping

    # Step 4: Start the training procedure
    for epoch in range(config['num_of_epochs']):
        # Training loop
        main_loop(phase=LoopPhase.TRAIN, epoch=epoch)

        # Validation loop
        with torch.no_grad():
            try:
                main_loop(phase=LoopPhase.VAL, epoch=epoch)
            except Exception as e:  # "patience has run out" exception :O
                print(str(e))
                break  # break out from the training loop

    # Step 5: Potentially test your model
    # Don't overfit to the test dataset - only when you've fine-tuned your model on the validation dataset should you
    # report your final loss and accuracy on the test dataset. Friends don't let friends overfit to the test data. <3
    if 1:
    # if config['should_test']:
        test_acc = main_loop(phase=LoopPhase.TEST)
        config['test_perf'] = test_acc
        print(f'Test accuracy = {test_acc}')
    else:
        config['test_perf'] = -1

    # # Save the latest GAT in the binaries directory
    # torch.save(
    #     utils.get_training_state(config, gat),
    #     os.path.join(BINARIES_PATH, utils.get_available_binary_name(config['dataset_name']))
    # )


def get_training_args(gat_layer_num=2,gat_cell_num=8,lr=0.001,drop_out=0.2):
    # '''流数据集'''
    # test_flows_data = rw_files.get_dict(file_path)
    # test_flows_data_list = list(test_flows_data.values())
    # CM_NUM_INPUT_FEATURES=len(test_flows_data_list)

    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=300)
    parser.add_argument("--patience_period", type=int, help="number of epochs with no improvement on val before terminating", default=1000)
    parser.add_argument("--lr", type=float, help="model learning rate", default=lr)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=5e-4)
    parser.add_argument("--should_test", action='store_true', help='should test the model on the test dataset? (no by default)')

    # # Dataset related
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='dataset to use for training', default=DatasetType.CORA.name)
    parser.add_argument("--should_visualize", action='store_true', help='should visualize the dataset? (no by default)')

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", action='store_true', help="enable tensorboard logging (no by default)")
    parser.add_argument("--console_log_freq", type=int, help="log to output console (epoch) freq (None for no logging)", default=50)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq (None for no logging)", default=100)
    args = parser.parse_args()

    head_list = [4]
    layer_list = [3]
    for i in range(gat_layer_num-1):
        layer_list.append(gat_cell_num)
        head_list.append(4)
    layer_list.append(flow_size_num)
    # Model architecture related
    gat_config = {
        "num_of_layers": gat_layer_num,
        # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_heads_per_layer": head_list,
        "num_features_per_layer": layer_list,
        "add_skip_connection": False,  # hurts perf on Cora
        "bias": True,  # result is not so sensitive to bias
        "dropout": drop_out,  # result is sensitive to dropout
        "layer_type": LayerType.IMP3  # fastest implementation enabled by default
    }

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    # Add additional config information
    training_config.update(gat_config)

    return training_config


if __name__ == '__main__':

    # Train the graph attention network (GAT)
    flows_folder = "D:/大文件/DNN_d_complete/testdata_set_5s/testdata_flows_5s"
    file_now = 0
    # test_flows_path = flows_folder+"/20240103_5s_" + str(file_now).zfill(5) + ".txt"
    test_flows_path = "D:/大文件/DNN_d_complete/zipf_testdata_set/testdata_flows/" + str(
        int(1.4 * 10)) + "_0160000_" + str(0).zfill(2) + ".txt"
    lr=0.001
    drop_out=0.2
    for gat_layer_num in [2]:
        for gat_cell_num in [16, 32, 64]:
            for flow_size_num in [8, 16]:
                    print("gat_layer_num:", gat_layer_num)
                    print("gat_cell_num:", gat_cell_num)
                    print("classify_num",flow_size_num)
                    train_gat_cm(get_training_args(gat_layer_num=gat_layer_num, gat_cell_num=gat_cell_num,lr=lr,drop_out=drop_out), file_path=test_flows_path)

