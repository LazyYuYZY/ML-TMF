''' 获取真实数据，排序好的序列，标签 '''
import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch
import time

from sklearn import linear_model
from sklearn import preprocessing

from LSTM_model import *
from Seq2seq_model import *
from sketchs import *
from Metrics import *

Lmax=10000
Lmin=1

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
T_slice=5
'''流数据集'''
def getcmdata(flows_path="",file_now=0,real_sketch=True,dataname="testdata2_set_5s"):
    test_flows_data = rw_files.get_dict(flows_path)
    test_flows_data_list = list(test_flows_data.values())

    '''处理cm数据'''
    # 获取流映射
    flow_index_path = "../"+dataname+"/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flow_index_path)
    # 获取每个时刻的flows映射与数目
    flows_data_onetime = {}
    for one_flow in test_flows_data.keys():
        flows_data_onetime[flows_alltime_dict[one_flow]] = test_flows_data[one_flow]

    # 获取cm参数
    file_name = "../sketch_params/160000_cm_sketch.txt"
    dict_cm = rw_files.get_dict(file_name)
    cm_d = 3
    cm_w=dict_cm['w']
    new_dir="../"+dataname+"/testdata_" + str(cm_w).zfill(6) + "_cm_5s"
    cm_test_path = new_dir +"/"+ str(file_now).zfill(5) + ".txt"
    # 若有对应cm数据集导入，没有就插入cm导出数据集
    if real_sketch:
        cm_sketch_load = np.loadtxt(cm_test_path)
        cm_sketch_now = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_load)


    '''获取test时刻dnn_cm的输入,即冲突流的id'''
    flows_index_path = "../"+dataname+"/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flows_index_path)
    get_value = np.vectorize(lambda x: flows_alltime_dict.get(x, 0))
    test_x_key = list(test_flows_data.keys())
    test_index_array = get_value(np.array(test_x_key))


    '''test dataset'''
    test_y = np.array(test_flows_data_list).reshape(-1, 1)

    test_cm_d=cm_sketch_now.query_d_np(test_index_array).T
    test_cm = np.amin(test_cm_d, axis=1).reshape(-1, 1)

    test_sorted_cm= np.take_along_axis(test_cm_d, np.argsort(test_cm_d, axis=1), axis=1)
    Lmax=np.max(test_sorted_cm)
    Lmin=np.min(test_sorted_cm)
    test_sorted_map_cm=(test_sorted_cm-Lmin)/Lmax

    RE=(test_cm-test_y)/test_y
    labels= (RE > 0.1).astype(int)
    print("易错流",np.sum(labels))

    acc = calculate_accuracy(torch.tensor(test_y, dtype=torch.float), torch.tensor(test_cm, dtype=torch.float))

    print("cm accuracy:" + "{:.6f}".format(acc))

    return test_sorted_map_cm,labels,test_y,test_cm


def calculate_accuracy(y_pred, y_true):
    predicted_labels = torch.round(y_pred)
    correct = (predicted_labels == y_true).float()
    accuracy = correct.sum() / len(correct)
    return accuracy

def train_classify(test_sorted_map_cm, labels):
    test_sorted_map_cm=test_sorted_map_cm*Lmax+Lmin
    N, d=test_sorted_map_cm.shape

    # TRAIN_RANGE = [int(N / 5), N]
    TRAIN_RANGE = [0, N]
    TEST_RANGE = [0, int(N / 5)]
    train_indices = np.arange(TRAIN_RANGE[0], TRAIN_RANGE[1])
    test_indices = np.arange(TEST_RANGE[0], TEST_RANGE[1])

    dataset_x = test_sorted_map_cm[train_indices]
    test_x = test_sorted_map_cm[test_indices]

    dataset_y = labels[train_indices]
    test_y = labels[test_indices]


    test_x=torch.tensor(test_x, dtype=torch.float)
    test_x=test_x.T
    test_x=test_x.unsqueeze(-1)

    # 使用批训练方式
    dataset = TensorDataset(torch.tensor(dataset_x, dtype=torch.float), torch.tensor(dataset_y, dtype=torch.float))

    # layers_list=[i for i in range(2,10,1)]
    layers_list = [4]
    # lr_list=[0.01,0.001]
    lr_list = [0.01]
    # epoch_list=[100,500,1000]
    epoch_list = [100]
    # batch_list=[100,1000,10000]
    batch_list = [10000]
    for deep_len in layers_list:
        for set_lr in lr_list:
            for set_epoch in epoch_list:
                for set_batch in batch_list:
                    start_time = time.time()
                    print('#LSTM层数:{},学习率:{},epoch:{},batch:{}#'.format(deep_len, set_lr, set_epoch, set_batch))
                    '''定义优化器和损失函数'''
                    lstm1net = LstmRNN(input_size=1,hidden_size=5,output_size=1,num_layers=4,classif_type=True)
                    optim = torch.optim.Adam(lstm1net.parameters(), lr=set_lr)
                    dataloader = DataLoader(dataset, batch_size=set_batch, shuffle=True)
                    # Loss = torch.nn.BCEWithLogitsLoss()
                    Loss = nn.BCELoss()
                    loss_list = []
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)

                    # 下面开始训练：
                    # 一共训练 1000次
                    for epoch in range(set_epoch):
                        loss = None
                        for batch_x, batch_y in dataloader:
                            batch_x = batch_x.T
                            batch_x = batch_x.unsqueeze(-1)
                            y_predict = lstm1net(batch_x)
                            loss = Loss(y_predict, batch_y)
                            optim.zero_grad()
                            loss.backward()
                            optim.step()
                            # loss_list.append(loss.item())
                        # 每100次 的时候打印一次日志
                        if (epoch + 1) % 10 == 0:
                            loss1 = None
                            predict = lstm1net(test_x)
                            loss1 = Loss(predict, torch.tensor(test_y, dtype=torch.float))
                            # print("易错流",torch.sum(torch.round(predict)))
                            print("step: {0} , train loss: {1} , test loss: {2}".format(epoch + 1, loss.item(),
                                                                                        loss1.item()))
                            scheduler.step()

                    # 使用训练好的模型进行预测
                    predict = lstm1net(test_x)
                    loss1 = Loss(predict, torch.tensor(test_y, dtype=torch.float))
                    print("last loss:{0}".format(loss1.item()))

                    acc=calculate_accuracy(torch.round(predict),torch.tensor(test_y, dtype=torch.float))

                    print("accuracy:" + "{:.6f}".format(acc))


                    end_time = time.time()
                    print('one train time:', end_time - start_time)
                    torch.save(lstm1net.state_dict(), "classify_lstm.pkl")

def train_regress_lstm(test_sorted_map_cm, test_val, labels):


    test_sorted_map_cm=test_sorted_map_cm[np.where(labels == 1)[0]]
    test_val=test_val[np.where(labels == 1)[0]]

    N, d = test_sorted_map_cm.shape

    test_val=(test_val-Lmin)/Lmax



    TRAIN_RANGE = [0, N]
    TEST_RANGE = [0, int(N / 5)]
    train_indices = np.arange(TRAIN_RANGE[0], TRAIN_RANGE[1])
    test_indices = np.arange(TEST_RANGE[0], TEST_RANGE[1])

    dataset_x = test_sorted_map_cm[train_indices]
    test_x = test_sorted_map_cm[test_indices]

    dataset_y = test_val[train_indices]
    test_y = test_val[test_indices]

    test_x=torch.tensor(test_x, dtype=torch.float)
    test_x=test_x.T
    test_x=test_x.unsqueeze(-1)

    # acc_cm = calculate_accuracy(torch.tensor(test_cm[test_indices], dtype=torch.float), torch.tensor(test_y*Lmax+Lmin, dtype=torch.float))
    # print("accuracy:" + "{:.6f}".format(acc_cm))

    # 使用批训练方式
    dataset = TensorDataset(torch.tensor(dataset_x, dtype=torch.float), torch.tensor(dataset_y, dtype=torch.float))

    # layers_list=[i for i in range(2,10,1)]
    layers_list = [4]
    # lr_list=[0.01,0.001]
    lr_list = [0.01]
    # epoch_list=[100,500,1000]
    epoch_list = [100]
    # batch_list=[100,1000,10000]
    batch_list = [10000]
    for deep_len in layers_list:
        for set_lr in lr_list:
            for set_epoch in epoch_list:
                for set_batch in batch_list:
                    start_time = time.time()
                    print('#LSTM层数:{},学习率:{},epoch:{},batch:{}#'.format(deep_len, set_lr, set_epoch, set_batch))
                    '''定义优化器和损失函数'''
                    lstm1net = LstmRNN(input_size=1,hidden_size=30,output_size=1,num_layers=4)
                    optim = torch.optim.Adam(lstm1net.parameters(), lr=set_lr)
                    dataloader = DataLoader(dataset, batch_size=set_batch, shuffle=True)
                    Loss = nn.MSELoss(reduction='mean')
                    loss_list = []
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)

                    # 下面开始训练：
                    # 一共训练 1000次
                    for epoch in range(set_epoch):
                        loss = None
                        for batch_x, batch_y in dataloader:
                            batch_x = batch_x.T
                            batch_x = batch_x.unsqueeze(-1)
                            y_predict = lstm1net(batch_x)
                            loss = Loss(y_predict, batch_y)
                            optim.zero_grad()
                            loss.backward()
                            optim.step()
                            # loss_list.append(loss.item())
                        # 每100次 的时候打印一次日志
                        if (epoch + 1) % 50 == 0:
                            loss1 = None
                            predict = lstm1net(test_x)
                            loss1 = Loss(predict, torch.tensor(test_y, dtype=torch.float))
                            print("step: {0} , train loss: {1} , test loss: {2}".format(epoch + 1, loss.item(),
                                                                                        loss1.item()))
                        scheduler.step()

                    # 使用训练好的模型进行预测
                    predict = lstm1net(test_x)
                    loss1 = Loss(predict, torch.tensor(test_y, dtype=torch.float))
                    print("last loss:{0}".format(loss1.item()))

                    acc=calculate_accuracy(torch.round(predict*Lmax+Lmin),torch.round(torch.tensor(test_y*Lmax+Lmin, dtype=torch.float)))

                    print("accuracy:" + "{:.6f}".format(acc))


                    end_time = time.time()
                    print('one train time:', end_time - start_time)
                    torch.save(lstm1net.state_dict(), "regress_lstm.pkl")

def train_regress_seq2seq(test_sorted_map_cm, test_val):

    N, d = test_sorted_map_cm.shape

    test_val=(test_val-Lmin)/Lmax



    TRAIN_RANGE = [0, N]
    TEST_RANGE = [0, int(N / 5)]
    train_indices = np.arange(TRAIN_RANGE[0], TRAIN_RANGE[1])
    test_indices = np.arange(TEST_RANGE[0], TEST_RANGE[1])

    dataset_x = test_sorted_map_cm[train_indices]
    test_x = test_sorted_map_cm[test_indices]

    dataset_y = test_val[train_indices]
    test_y = test_val[test_indices]

    test_x=torch.tensor(test_x, dtype=torch.float)
    test_x=test_x.T
    test_x=test_x.unsqueeze(-1)

    # acc_cm = calculate_accuracy(torch.tensor(test_cm[test_indices], dtype=torch.float), torch.tensor(test_y*Lmax+Lmin, dtype=torch.float))
    # print("accuracy:" + "{:.6f}".format(acc_cm))

    # 使用批训练方式
    dataset = TensorDataset(torch.tensor(dataset_x, dtype=torch.float), torch.tensor(dataset_y, dtype=torch.float))

    # layers_list=[i for i in range(2,10,1)]
    layers_list = [4]
    # lr_list=[0.01,0.001]
    lr_list = [0.01]
    # epoch_list=[100,500,1000]
    epoch_list = [100]
    # batch_list=[100,1000,10000]
    batch_list = [10000]
    for deep_len in layers_list:
        for set_lr in lr_list:
            for set_epoch in epoch_list:
                for set_batch in batch_list:
                    start_time = time.time()
                    print('#LSTM层数:{},学习率:{},epoch:{},batch:{}#'.format(deep_len, set_lr, set_epoch, set_batch))
                    '''定义优化器和损失函数'''
                    lstm1net = Seq2seqLstm(input_size=1,hidden_size=30,output_size=1,num_layers=2)
                    optim = torch.optim.Adam(lstm1net.parameters(), lr=set_lr)
                    dataloader = DataLoader(dataset, batch_size=set_batch, shuffle=True)
                    Loss = nn.MSELoss(reduction='mean')
                    loss_list = []
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)

                    # 下面开始训练：
                    # 一共训练 1000次
                    for epoch in range(set_epoch):
                        loss = None
                        for batch_x, batch_y in dataloader:
                            batch_x = batch_x.T
                            batch_x = batch_x.unsqueeze(-1)
                            y_predict = lstm1net(batch_x)
                            loss = Loss(y_predict, batch_y)
                            optim.zero_grad()
                            loss.backward()
                            optim.step()
                            # loss_list.append(loss.item())
                        # 每100次 的时候打印一次日志
                        if (epoch + 1) % 50 == 0:
                            loss1 = None
                            predict = lstm1net(test_x)
                            loss1 = Loss(predict, torch.tensor(test_y, dtype=torch.float))
                            print("step: {0} , train loss: {1} , test loss: {2}".format(epoch + 1, loss.item(),
                                                                                        loss1.item()))
                        scheduler.step()

                    # 使用训练好的模型进行预测
                    predict = lstm1net(test_x)
                    loss1 = Loss(predict, torch.tensor(test_y, dtype=torch.float))
                    print("last loss:{0}".format(loss1.item()))

                    acc=calculate_accuracy(torch.round(predict*Lmax+Lmin),torch.round(torch.tensor(test_y*Lmax+Lmin, dtype=torch.float)))

                    print("accuracy:" + "{:.6f}".format(acc))


                    end_time = time.time()
                    print('one train time:', end_time - start_time)
                    torch.save(lstm1net.state_dict(), "regress_seq2seq.pkl")


def eval_classify(test_sorted_map_cm,labels):

    test_x = test_sorted_map_cm*Lmax+Lmin
    test_y=labels

    test_x=torch.tensor(test_x, dtype=torch.float)
    test_x=test_x.T
    test_x=test_x.unsqueeze(-1)

    lstm_model_path = "classify_lstm.pkl"
    lstm1net = LstmRNN(input_size=1,hidden_size=5,output_size=1,num_layers=4,classif_type=True)
    lstm1net.load_state_dict(torch.load(lstm_model_path, map_location=device))

    # 使用训练好的模型进行预测
    predict = lstm1net(test_x)

    print(torch.sum(torch.round(predict)))
    acc = calculate_accuracy(torch.round(predict), torch.tensor(test_y, dtype=torch.float))

    print("classify accuracy:" + "{:.6f}".format(acc))

    return torch.round(predict)

def eval_regress(test_sorted_map_cm, test_val, labels):

    test_x = test_sorted_map_cm[np.where(labels == 1)[0]]
    test_y=np.zeros(test_val.shape)
    test_y[:]=test_val


    test_x=torch.tensor(test_x, dtype=torch.float)
    test_x=test_x.T
    test_x=test_x.unsqueeze(-1)

    lstm_model_path = "regress_lstm.pkl"
    lstm2net = LstmRNN(input_size=1,hidden_size=30,output_size=1,num_layers=4,classif_type=False)
    lstm2net.load_state_dict(torch.load(lstm_model_path, map_location=device))

    # 使用训练好的模型进行预测
    predict = lstm2net(test_x)
    predict=predict*Lmax+Lmin

    test_y[np.where(labels == 1)[0]]=torch.round(predict).detach().numpy()
    test_y[np.where(labels == 0)[0]] = np.amin(test_sorted_map_cm[np.where(labels == 0)[0]]*Lmax+Lmin,axis=1).reshape(-1,1)

    acc = calculate_accuracy(torch.tensor(test_y, dtype=torch.float), torch.tensor(test_val, dtype=torch.float))

    print("talent accuracy:" + "{:.6f}".format(acc))


    return test_y

def eval_regress2(test_sorted_map_cm, test_val, labels):

    test_x = test_sorted_map_cm[np.where(labels == 1)[0]]
    test_y=np.zeros(test_val.shape)
    test_y[:]=test_val


    test_x=torch.tensor(test_x, dtype=torch.float)
    test_x=test_x.T
    test_x=test_x.unsqueeze(-1)

    lstm_model_path = "regress_seq2seq.pkl"
    lstm2net = Seq2seqLstm(input_size=1,hidden_size=30,output_size=1,num_layers=2)
    lstm2net.load_state_dict(torch.load(lstm_model_path, map_location=device))

    # 使用训练好的模型进行预测
    predict = lstm2net(test_x)
    predict=predict*Lmax+Lmin

    test_y[np.where(labels == 1)[0]]=torch.round(predict).detach().numpy()

    acc = calculate_accuracy(torch.tensor(test_y, dtype=torch.float), torch.tensor(test_val, dtype=torch.float))

    print("deep accuracy:" + "{:.6f}".format(acc))


    return test_y


if __name__ == '__main__':
    # file_now=0
    # dataname="testdata_set_5s"
    # flows_path="../"+dataname+"/testdata_flows_5s/" + str(file_now).zfill(5) + ".txt"
    # test_sorted_map_cm, labels, test_val, test_cm=getcmdata(flows_path=flows_path,file_now=file_now,real_sketch=True,dataname=dataname)
    #
    # cm_metrics = Metrics(real_val=test_val, pre_val=test_cm)
    # cm_metrics.get_allval()
    # print("CM", cm_metrics.ARE_val)
    #
    # train_classify(test_sorted_map_cm, labels)
    # train_regress_lstm(test_sorted_map_cm, test_val, labels)
    # train_regress_seq2seq(test_sorted_map_cm, test_val)

    plots_np = np.full((3, 10), 0, dtype=float)
    for file_now in range(10):
        dataname = "testdata_set_5s"
        flows_path = "../testdata_set_5s/testdata_flows_5s/" + str(file_now).zfill(5) + ".txt"
        test_sorted_map_cm, labels, test_val, test_cm = getcmdata(flows_path=flows_path, file_now=file_now,
                                                                  real_sketch=True,dataname=dataname)

        feature_scaler = preprocessing.StandardScaler().fit(test_sorted_map_cm)
        test_x = feature_scaler.transform(test_sorted_map_cm*Lmax+Lmin)
        # clf = linear_model.Ridge(alpha=.5)
        clf = linear_model.LinearRegression()
        clf.fit(test_x, test_val)
        pre = clf.predict(test_x)
        pre=np.round(pre)
        ML_metrics = Metrics(real_val=test_val, pre_val=pre)
        ML_metrics.get_allval()
        print("ML", ML_metrics.ARE_val)
        plots_np[0,file_now]=ML_metrics.ARE_val

        cm_metrics = Metrics(real_val=test_val, pre_val=test_cm)
        cm_metrics.get_allval()
        print("CM", cm_metrics.ARE_val)

        errorflow_id = eval_classify(test_sorted_map_cm, labels)
        lstm_y = eval_regress(test_sorted_map_cm, test_val, errorflow_id.detach().numpy())
        seq2seq_y=eval_regress2(test_sorted_map_cm, test_val, errorflow_id.detach().numpy())
        # test_y = eval_regress2(test_sorted_map_cm, test_val, np.ones(labels.shape))

        lstm_metrics = Metrics(real_val=test_val, pre_val=lstm_y)
        seq2seq_metrics = Metrics(real_val=test_val, pre_val=seq2seq_y)


        lstm_metrics.get_allval()
        seq2seq_metrics.get_allval()

        print("Talentsketch",lstm_metrics.ARE_val)
        print("Deepsketch",seq2seq_metrics.ARE_val)
        plots_np[1, file_now] = lstm_metrics.ARE_val
        plots_np[2, file_now] = seq2seq_metrics.ARE_val
    sketchs=["ML","Talent","Deep"]
    for k in range(3):
        results_path = "../results/wide/ML/" + sketchs[k] + ".txt"
        np.savetxt(results_path, plots_np[k, :])


