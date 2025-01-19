from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import sys
import time
from sklearn import preprocessing

from DNN1 import *
from sketchs import *

P_flowdrop=0.01

# 控制台输出记录到文件
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def conflict(index_np,dict_tower,w=16 * 10 ** 4,d=3,max_conf=1):
    a = np.array(dict_tower['a'])
    b = np.array(dict_tower['b'])
    p = np.array(dict_tower['p'])
    offset = dict_tower['offset']
    w=np.array([w,w*2,w*4]).reshape(-1,1)
    tower_d_id=(a*(index_np+offset)+b)%p%w
    data_x_d= np.full((index_np.shape[0],d*max_conf), -1)

    for i in range(d):
        unique_values, counts = np.unique(tower_d_id[i], return_counts=True)
        conf_items=unique_values[counts>1]
        for conf_tower_id in conf_items:#冲突流在sketch中的索引
            conf_flows_id=np.where(np.isin(tower_d_id[i], conf_tower_id))[0]#单个conter冲突流在sketch的索引
            conf_flows_id_list=conf_flows_id.tolist()
            for conf_flow_id in conf_flows_id_list:
                k = 0
                for j in range(1+max_conf):
                    if j>=len(conf_flows_id_list):
                        break
                    if conf_flows_id_list[j] != conf_flow_id:
                        data_x_d[conf_flow_id][i * max_conf + k] = index_np[conf_flows_id_list[j]]
                        k = k + 1
                        if k==max_conf:
                            break

    return data_x_d

def select_indices(index_array,p):
    # 计算要选择的行数
    num_rows =index_array.shape[0]
    num_selected_rows = int(num_rows * p)
    # 生成要选择的行索引
    selected_indices = np.random.choice(num_rows, size=num_selected_rows, replace=False)
    return selected_indices

if __name__ == '__main__':
    '''自定义目录存放日志文件'''
    log_path = '../Logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 日志文件名按照程序运行时间设置
    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)

    '''准备数据'''
    max_conf=2

    file_now=random.randint(0,10-1)
    # file_now=179
    file_now1=file_now
    while file_now1 == file_now:
        file_now1=random.randint(0,10-1)

    test_y_path = "../traindata_set_5s/traindata_flows_5s/" + str(file_now).zfill(5) + ".txt"
    dataset_y_path="../traindata_set_5s/traindata_flows_5s/"+str(file_now1).zfill(5)+".txt"

    # 处理tower数据

    file_name = "../sketch_params/160000_tower_sketch.txt"
    dict_tower = rw_files.get_dict(file_name)
    tower_d = 3
    tower_w = dict_tower['w']

    tower_test_path = "../traindata_set_5s/traindata_160000_tower_5s/" + str(file_now).zfill(5) + ".txt"
    tower_sketch_now = np.loadtxt(tower_test_path)
    sketch_now = tower_sketch(tower_d=3, tower_w=tower_w, flag=1, dict_tower=dict_tower, tower_sketch_load=tower_sketch_now)


    tower_train_path = "../traindata_set_5s/traindata_160000_tower_5s/" + str(file_now1).zfill(5) + ".txt"
    tower_sketch_now1 = np.loadtxt(tower_train_path)
    sketch_now1 = tower_sketch(tower_d=3, tower_w=tower_w, flag=1, dict_tower=dict_tower, tower_sketch_load=tower_sketch_now1)

    train_flows_data = rw_files.get_dict(dataset_y_path)
    train_flows_data_list = list(train_flows_data.values())
    test_flows_data = rw_files.get_dict(test_y_path)
    test_flows_data_list = list(test_flows_data.values())

    flows_index_path = "../traindata_set_5s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flows_index_path)
    get_value = np.vectorize(lambda x: flows_alltime_dict.get(x, 0))

    #获取train时刻流信息
    dataset_x_key = list(train_flows_data.keys())
    index_array = get_value(np.array(dataset_x_key))

    selected_indices=select_indices(index_array,1-P_flowdrop)
    index_array = index_array[selected_indices]

    dataset_x_ids = conflict(index_np=index_array,dict_tower=dict_tower,w=tower_w,max_conf=max_conf)

    # 获取test时刻流信息
    test_x_key = list(test_flows_data.keys())
    test_index_array = get_value(np.array(test_x_key))

    test_selected_indices = select_indices(test_index_array, 1 - P_flowdrop)
    test_index_array = test_index_array[test_selected_indices]

    test_x_ids = conflict(index_np=test_index_array, dict_tower=dict_tower,w=tower_w,max_conf=max_conf)

    '''获取所在时刻的d个查询值'''
    # 获取train
    dataset_x = np.full((dataset_x_ids.shape[0], tower_d*(1+dataset_x_ids.shape[1])), 0)
    for i in range(tower_d*max_conf):
        conf_flows_id=np.where(dataset_x_ids[:,i]!=-1)[0]
        dataset_x[conf_flows_id,i*tower_d:(i+1)*tower_d]=sketch_now1.query_d_np(dataset_x_ids[conf_flows_id,i]).T
    dataset_x[:,tower_d*dataset_x_ids.shape[1]:]=sketch_now1.query_d_np(index_array).T

    # 获取test
    test_x = np.full((test_x_ids.shape[0], tower_d * (1+test_x_ids.shape[1])), 0)
    for i in range(tower_d * max_conf):
        conf_flows_id = np.where(test_x_ids[:, i] != -1)[0]
        test_x[conf_flows_id, i * tower_d:(i + 1) * tower_d] = sketch_now.query_d_np(test_x_ids[conf_flows_id, i]).T
    test_x[:, tower_d * test_x_ids.shape[1]:] = sketch_now.query_d_np(test_index_array).T

    # 查询所有流在tower sketch中的值
    tower_flows_query_d = sketch_now.query_d_np(test_index_array)
    tower_x = tower_flows_query_d.T
    # 流的查询值，即对每行求min:(n,1)
    flows_query_tower = np.min(tower_x, axis=1)
    tower_y = np.array(list(test_flows_data.values())).T

    # #获取tower的相对误差
    tower_num_array = flows_query_tower.T
    real_num_array = np.array(test_flows_data_list).reshape(-1,1)[test_selected_indices].reshape(1,-1)
    relative_tower_error = np.abs((real_num_array - tower_num_array) / real_num_array)
    print("relative error:" + "{:.6f}".format(np.mean(relative_tower_error)))

    '''train dataset'''
    dataset_x=np.log(1+dataset_x)
    dataset_y = np.log(np.array(train_flows_data_list).reshape(-1, 1)[selected_indices])

    '''test dataset'''
    test_x=np.log(1+test_x)
    test_y = np.log(np.array(test_flows_data_list).reshape(-1,1)[test_selected_indices])

    # dataset_x = preprocessing.StandardScaler().fit_transform(dataset_x)
    # test_x = preprocessing.StandardScaler().fit_transform(test_x)
    # 使用批训练方式
    dataset=TensorDataset(torch.tensor(dataset_x,dtype=torch.float),torch.tensor(dataset_y,dtype=torch.float))

    # layers_list=[i for i in range(3,9,2)]
    layers_list = [4]
    # cell_list = [128,256]
    cell_num_list = [100]
    # lr_list=[0.01,0.001]
    lr_list=[0.01]
    drop_out_list=[0.0]
    # epoch_list=[100,500,1000]
    epoch_list = [25]
    # batch_list=[100,1000,10000]
    batch_list=[10000]
    for deep_len in layers_list:
        for cell_num in cell_num_list:
            for set_lr in lr_list:
                for drop_out in drop_out_list:
                    for set_epoch in epoch_list:
                        for set_batch in batch_list:
                            start_time = time.time()
                            print('#DNN层数:{},神经元数量:{},学习率:{},dropout:{},epoch:{},batch:{}#'.format(deep_len,cell_num,set_lr,drop_out,set_epoch,set_batch))
                            '''定义优化器和损失函数'''
                            dnn1net = DNN1(input_size=dataset_x.shape[1], deep_l=deep_len - 2,num=cell_num,drop_out=drop_out)
                            optim=torch.optim.Adam(DNN1.parameters(dnn1net), lr=set_lr)
                            dataloader = DataLoader(dataset, batch_size=set_batch, shuffle=True)
                            scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)
                            Loss=nn.MSELoss(reduction='mean')
                            loss_list=[]

                            # 下面开始训练：
                            # 一共训练 1000次
                            for epoch in range(set_epoch):
                                loss=None
                                for batch_x,batch_y in dataloader:
                                    y_predict=dnn1net(batch_x)
                                    loss=Loss(y_predict,batch_y)
                                    optim.zero_grad()
                                    loss.backward()
                                    optim.step()
                                    # loss_list.append(loss.item())
                                # 每100次 的时候打印一次日志
                                if (epoch+1)%10==0:
                                    loss1 = None
                                    predict = dnn1net(torch.tensor(test_x, dtype=torch.float))
                                    loss1 = Loss(predict, torch.tensor(test_y, dtype=torch.float))
                                    print("step: {0} , train loss: {1} , test loss: {2}".format(epoch+1,loss.item(),loss1.item()))
                                    test_0=np.zeros(test_y.shape)
                                    if loss1==Loss(torch.tensor(test_0,dtype=torch.float),torch.tensor(test_y,dtype=torch.float)):
                                        break
                                scheduler.step()  # 学习率的更新

                            # 使用训练好的模型进行预测
                            predict=dnn1net(torch.tensor(test_x, dtype=torch.float))
                            loss1=Loss(predict,torch.tensor(test_y,dtype=torch.float))
                            print("last loss:{0}".format(loss1.item()))

                            pre_arr=predict.detach().numpy()
                            # relative_pre_error=np.exp(pre_arr-test_y)-1
                            relative_pre_error=np.round(np.exp(pre_arr))/np.exp(test_y)-1
                            print("relative pre error:"+"{:.6f}".format(np.mean(np.abs(relative_pre_error))))
                            print("max relative pre error:" + "{:.6f}".format(np.max(np.abs(relative_pre_error))))

                            end_time = time.time()
                            print('one train time:',end_time-start_time)

                            torch.save(dnn1net.state_dict(), "tower_dnn_d_params_"+str(max_conf)+".pkl")
