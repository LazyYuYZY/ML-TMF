import argparse

import numpy as np
from torch.optim import Adam
import time

from GAT_model2 import GAT
from utils.constants import *

from sketchs import *
from Metrics import *
from DNN1.DNN1 import *

input_type="lg"
label_type="lg"
flow_size_num=27
conf_num=[4,2,1]
dnn_num=4
device = torch.device("cpu")  # checking whether you have a GPU, I hope so!

'''ranges'''
CORA_TRAIN_RANGE=[0,120000]
CORA_VAL_RANGE=[120000,140000]
CORA_TEST_RANGE=[120000,160000]
train_indices = torch.arange(CORA_TRAIN_RANGE[0], CORA_TRAIN_RANGE[1], dtype=torch.long, device=device)
val_indices = torch.arange(CORA_VAL_RANGE[0], CORA_VAL_RANGE[1], dtype=torch.long, device=device)
test_indices = torch.arange(CORA_TEST_RANGE[0], CORA_TEST_RANGE[1], dtype=torch.long, device=device)
alpha_globe=3.0
flow_num=160000
w_globe=8*10**4
# 4，8，12，16，20


def conflict(index_np,dict_tower,w=16 * 10 ** 4,d=3,max_conf=[4,2,1]):
    a = np.array(dict_tower['a'])
    b = np.array(dict_tower['b'])
    p = np.array(dict_tower['p'])
    offset = dict_tower['offset']
    w = np.array([w, w * 2, w * 4]).reshape(-1, 1)
    tower_d_id=(a*(index_np+offset)+b)%p%w
    node_num=index_np.shape[0]
    edge_index_tower = []
    edge_size_d=[0]


    for i in range(d):
        unique_values, counts = np.unique(tower_d_id[i], return_counts=True)
        conf_items = unique_values[counts >= 1]
        for conf_tower_id in conf_items:  # 冲突流在sketch中的索引
            conf_flows_id = np.where(np.isin(tower_d_id[i], conf_tower_id))[0]  # 单个conter冲突流在sketch的索引
            conf_flows_id_list = conf_flows_id.tolist()
            for conf_flow_id in conf_flows_id_list:
                k=0
                for flow_j in conf_flows_id_list:
                    if flow_j==conf_flow_id:
                        continue
                    # if (flow_j, conf_flow_id) not in edge_index_tower:
                    #     edge_index_tower[(flow_j, conf_flow_id)] = [0, 0, 0]
                    # edge_index_tower[(flow_j, conf_flow_id)][i] = 1
                    edge_index_tower.append([flow_j, conf_flow_id])
                    k=k+1
                    if k>=max_conf[i]:
                        break;
                while k<max_conf[i]:
                    used_dummy_num=0
                    for used_hash in range(i):
                        used_dummy_num=used_dummy_num+max_conf[used_hash]
                    dummy_node=node_num+used_dummy_num+k
                    # edge_index_tower[(flow_j, dummy_node)] = [0, 0, 0]
                    edge_index_tower.append([dummy_node,conf_flow_id])
                    k=k+1
        edge_size_d.append(len(edge_index_tower))
    # # 将键从元组转换为列表
    # keys_list = [list(key) for key in edge_index_tower.keys()]
    # values_list = [value for value in edge_index_tower.values()]

    return torch.tensor(edge_index_tower, dtype=torch.float),torch.tensor(edge_size_d, dtype=torch.int32)



def load_tower_data_wide(w=16* 10 ** 4,flows_path="",file_now=9,real_sketch=False,max_conf=conf_num):

    '''流数据集'''
    flows_index_path = "../traindata_set_5s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flows_index_path)
    get_value = np.vectorize(lambda x: flows_alltime_dict.get(x, 0))
    test_flows_data = rw_files.get_dict(flows_path)
    test_flows_data_list=list(test_flows_data.values())

    '''处理tower数据'''
    file_name = "../sketch_params/" + str(w).zfill(6) + "_tower_sketch_train.txt"
    dict_tower = rw_files.get_dict(file_name)
    tower_d = 3
    tower_w = dict_tower['w']
    tower_test_path = "../traindata_set_5s/traindata_160000_tower_5s/"+str(file_now).zfill(5)+".txt"
    if real_sketch:
        tower_sketch_load = np.loadtxt(tower_test_path)
        tower_sketch_now = tower_sketch(tower_d=tower_d, tower_w=tower_w, flag=1, dict_tower=dict_tower, tower_sketch_load=tower_sketch_load)

    '''获取test时刻dnn_tower的输入,即冲突流的id'''
    test_x_key=list(test_flows_data.keys())
    test_index_array = get_value(np.array(test_x_key))

    node_features = tower_sketch_now.query_d_np(test_index_array).T
    dummy_node_features=np.zeros((sum(max_conf),tower_d))
    node_features = np.concatenate((node_features, dummy_node_features), axis=0)
    '''test dataset'''
    if input_type=="lg":
        node_features = np.log(1+node_features)
    node_features = torch.tensor(node_features, dtype=torch.float)

    test_y = np.array(test_flows_data_list).reshape(-1, 1)
    test_tower = np.amin(tower_sketch_now.query_d_np(test_index_array).T, axis=1).reshape(-1, 1)

    # test_y = test_y - 1
    # test_test_y = test_y[val_indices]

    # test_tower = test_tower - 1
    # test_test_tower = test_tower[val_indices]
    # tower_y = test_test_tower - test_test_y

    #
    # print("CM acc=", np.mean(tower_y == 0))
    # print("all 1 acc=", np.mean(test_test_y == 0))
    # print("all big acc=", np.mean(test_test_y == flow_size_num-1))
    # node_labels = torch.tensor(test_y, dtype=torch.float)
    # test_y=np.log(test_y)

    if label_type=="lg":
        test_y_lg= np.log(test_y)
        node_labels = torch.tensor(test_y_lg, dtype=torch.float)
    else:
        test_y[test_y > flow_size_num] = flow_size_num
        test_tower[test_tower > flow_size_num] = flow_size_num
        node_labels = torch.tensor(test_y, dtype=torch.float)
    print("Tower ARE=", np.mean(np.abs(test_tower / test_y - 1)))
    print("all 1 ARE=", np.mean((test_y - 1) / test_y))
    edge_index, edge_size_d= conflict(index_np=test_index_array, dict_tower=dict_tower, w=tower_w, d=tower_d,max_conf=conf_num)

    return node_features, node_labels, edge_index.t(), edge_size_d



def load_tower_data_zipf(w=16* 10 ** 4,flows_path="",alpha=alpha_globe,n=160000,zifp_sketch=False,numbering=0,max_conf=conf_num):

    '''流数据集'''
    test_flows_data_str = rw_files.get_dict(flows_path)
    test_flows_data={}
    for key, value in test_flows_data_str.items():
        new_key = int(key)
        test_flows_data[new_key] = value
    test_flows_data_list = list(test_flows_data.values())

    '''处理tower数据'''
    file_name = "../sketch_params/" + str(w).zfill(6) + "_tower_sketch.txt"
    dict_tower = rw_files.get_dict(file_name)
    tower_d = 3
    tower_w = dict_tower['w']
    tower_test_path = "../zipf_testdata_set/tower/" + str(tower_w).zfill(6) + "_" + str(int(alpha * 10)) + "_" + str(n).zfill(7) + "_" + str(numbering).zfill(2) + ".txt"
    if zifp_sketch:
        tower_sketch_load = np.loadtxt(tower_test_path)
        tower_sketch_now = tower_sketch(tower_d=tower_d, tower_w=tower_w, flag=1, dict_tower=dict_tower, tower_sketch_load=tower_sketch_load)
    else:
        tower_sketch_load = np.full((tower_d, tower_w*(2**(tower_d-1))), 0)  # tower存储的counter值
        tower_sketch_now = tower_sketch(tower_d=tower_d, tower_w=tower_w, flag=1, dict_tower=dict_tower, tower_sketch_load=tower_sketch_load)
        tower_sketch_now.insert_dict(test_flows_data)
        tower_sketch_load = tower_sketch_now.Matrix  # tower的counter值
        '''导出数据'''
        np.savetxt(tower_test_path, tower_sketch_load, fmt='%d')
    '''获取test时刻dnn_tower的输入,即冲突流的id'''
    test_index_array = np.array(list(test_flows_data.keys()))

    node_features = tower_sketch_now.query_d_np(test_index_array).T
    dummy_node_num=int(sum(max_conf))
    dummy_node_features=np.zeros((dummy_node_num,tower_d))
    node_features = np.concatenate((node_features, dummy_node_features), axis=0)
    '''test dataset'''
    if input_type=="lg":
        node_features = np.log(1+node_features)
    node_features = torch.tensor(node_features, dtype=torch.float)

    test_y = np.array(test_flows_data_list).reshape(-1, 1)
    test_tower = np.amin(tower_sketch_now.query_d_np(test_index_array).T, axis=1).reshape(-1, 1)


    if label_type=="lg":
        test_y_lg= np.log(test_y)
        node_labels = torch.tensor(test_y_lg, dtype=torch.float)
    else:
        test_y[test_y > flow_size_num] = flow_size_num
        test_tower[test_tower > flow_size_num] = flow_size_num
        node_labels = torch.tensor(test_y, dtype=torch.float)
    print("Tower ARE=", np.mean(np.abs(test_tower / test_y - 1)))
    print("all 1 ARE=", np.mean((test_y - 1) / test_y))
    edge_index, edge_size_d = conflict(index_np=test_index_array, dict_tower=dict_tower, w=tower_w, d=tower_d,max_conf=conf_num)

    return node_features, node_labels, edge_index.t(), edge_size_d



# Simple decorator function so that I don't have to pass arguments that don't change from epoch to epoch
def get_main_loop(config, gat, cross_entropy_loss, optimizer, node_features, node_labels, edge_index, edge_size_d, train_indices, val_indices, test_indices, patience_period, time_start,schedule):

    node_dim = 0  # node axis
    Loss = nn.MSELoss(reduction='mean')
    train_labels = node_labels.index_select(node_dim, train_indices)
    val_labels = node_labels.index_select(node_dim, val_indices)
    test_labels = node_labels.index_select(node_dim, test_indices)

    # node_features shape = (N, FIN), edge_index shape = (2, E)
    graph_data = (node_features, edge_index, edge_size_d)  # I pack data into tuples because GAT uses nn.Sequential which requires it

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
        #标签为整数，并且交叉熵其张量维度应为(n,)
        # gt_node_labels = gt_node_labels.long()
        # gt_node_labels = torch.squeeze(gt_node_labels)

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
        # loss = cross_entropy_loss(nodes_unnormalized_scores, gt_node_labels)

        loss=Loss(nodes_unnormalized_scores,gt_node_labels)


        if phase == LoopPhase.TRAIN:
            optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph (.grad fields)
            loss.backward()  # compute the gradients for every trainable weight in the computational graph
            optimizer.step()  # apply the gradients to weights
            if epoch%2==0:
                schedule.step()


        # Calculate the main metric - accuracy

        # Finds the index of maximum (unnormalized) score for every node and that's the class prediction for that node.
        # Compare those to true (ground truth) labels and find the fraction of correct predictions -> accuracy metric.
        # class_predictions = torch.argmax(nodes_unnormalized_scores, dim=-1)
        # accuracy = torch.sum(torch.eq(class_predictions, gt_node_labels).long()).item() / len(gt_node_labels)
        if label_type=="lg":
            accuracy = torch.mean(torch.abs(torch.round(torch.exp(nodes_unnormalized_scores))/torch.exp(gt_node_labels)-1))
        else:
            accuracy = torch.mean(torch.abs(torch.round(nodes_unnormalized_scores) / gt_node_labels - 1))


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


def train_gat_cora(config,file_path=""):
    global BEST_VAL_PERF, BEST_VAL_LOSS


    # Step 1: load the graph data
    node_features, node_labels, edge_index, edge_size_d = load_tower_data_zipf(w=w_globe,flows_path=file_path,alpha=alpha_globe,n=160000,max_conf=conf_num,zifp_sketch=False)
    # node_features, node_labels, edge_index, train_indices, val_indices, test_indices=

    # Step 2: prepare the model
    gat = GAT(
        num_of_layers=config['num_of_layers'],
        num_heads_per_layer=config['num_heads_per_layer'],
        num_features_per_layer=config['num_features_per_layer'],
        add_skip_connection=config['add_skip_connection'],
        bias=config['bias'],
        dropout=config['dropout'],
        layer_type=config['layer_type'],
        log_attention_weights=False,
        dnn_layer_num=dnn_num# no need to store attentions, used only in playground.py for visualizations
    ).to(device)

    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(gat.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    main_loop = get_main_loop(
        config,
        gat,
        loss_fn,
        optimizer,
        node_features,
        node_labels,
        edge_index,
        edge_size_d,
        train_indices,
        val_indices,
        test_indices,
        config['patience_period'],
        time.time(),
        scheduler)

    BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT = [0, 0, 0]  # reset vars used for early stopping

    # Step 4: Start the training procedure
    for epoch in range(config['num_of_epochs']):
        # Training loop
        main_loop(phase=LoopPhase.TRAIN, epoch=epoch)

        # Validation loop
        with torch.no_grad():
            try:
                if config['console_log_freq'] is not None and epoch % config['console_log_freq'] == 0:
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

    torch.save(gat.state_dict(), "tower_gat_d_params_" + str(conf_num[-1]) + ".pkl")

    # # Save the latest GAT in the binaries directory
    # torch.save(
    #     utils.get_training_state(config, gat),
    #     os.path.join(BINARIES_PATH, utils.get_available_binary_name(config['dataset_name']))
    # )


def get_training_args(gat_layer_num=2,dnn_layer_num=32):
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=1000)
    parser.add_argument("--patience_period", type=int, help="number of epochs with no improvement on val before terminating", default=500)
    parser.add_argument("--lr", type=float, help="model learning rate", default=5e-3)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=0)
    parser.add_argument("--should_test", action='store_true', help='should test the model on the test dataset? (no by default)')

    # Dataset related
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='dataset to use for training', default=DatasetType.CORA.name)
    parser.add_argument("--should_visualize", action='store_true', help='should visualize the dataset? (no by default)')

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", action='store_true', help="enable tensorboard logging (no by default)")
    parser.add_argument("--console_log_freq", type=int, help="log to output console (epoch) freq (None for no logging)", default=50)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq (None for no logging)", default=1000)
    args = parser.parse_args()
    head_list = []
    layer_list = [3]
    # Model architecture related
    gat_config = {
        "num_of_layers": 3,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_heads_per_layer": [ 4, 4, 1],
        "num_features_per_layer": [3, 9, 27, 81],
        "add_skip_connection": True,  # hurts perf on Cora
        "bias": True,  # result is not so sensitive to bias
        "dropout": 0.0,  # result is sensitive to dropout
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

    # training
    for w_globe in [16*10**4]:
        for alpha_globe in [2.0]:
            test_flows_path = "../zipf_testdata_set/testdata_flows/20_0320000_00.txt"
            print("w=", w_globe)
            print("alpha=", alpha_globe)
            for dnn_num in [4]:
                print("conf_num=", conf_num)
                print("dnn_num=", dnn_num)
                train_gat_cora(get_training_args(dnn_layer_num=dnn_num), file_path=test_flows_path)


    # test
    # '''加载模型'''
    # config=get_training_args(dnn_layer_num=dnn_num)
    # model_path = 'tower_gat_d_params_3.pkl'
    # model_object = GAT(
    #     num_of_layers=config['num_of_layers'],
    #     num_heads_per_layer=config['num_heads_per_layer'],
    #     num_features_per_layer=config['num_features_per_layer'],
    #     add_skip_connection=config['add_skip_connection'],
    #     bias=config['bias'],
    #     dropout=config['dropout'],
    #     layer_type=config['layer_type'],
    #     log_attention_weights=False,
    #     dnn_layer_num=dnn_num# no need to store attentions, used only in playground.py for visualizations
    # ).to(device)
    # model_object.load_state_dict(torch.load(model_path))
    #
    #
    # file_now=random.randint(0,179)
    # for w_globe in [16*10**4]:
    #     for alpha_globe in [2.0]:
    #         test_flows_path = "../traindata_set_5s/traindata_flows_5s/"+str(file_now).zfill(5)+".txt"
    #         print("w=", w_globe)
    #         print("alpha=", alpha_globe)
    #
    #         node_features, node_labels, edge_index = load_tower_data_wide(w=w_globe, flows_path=test_flows_path,
    #                                                                    file_now=file_now, max_conf=conf_num,real_sketch=True)
    #         graph_data=(node_features,edge_index)
    #         predictions = model_object(
    #             graph_data
    #         )[0]
    #         pre_arr = predictions.detach().numpy()
    #         test_gat = np.round(np.exp(pre_arr))
    #         # print(node_labels[:10])
    #         # print(test_gat[:10])
    #         # print(node_labels[-10:])
    #         # print(test_gat[-20:])
    #         test_y_lg=node_labels.numpy()
    #         test_y=np.exp(test_y_lg)
    #         '''性能评估'''
    #         GAT_metrics = Metrics(real_val=test_y, pre_val=test_gat[:-9])
    #
    #         GAT_metrics.get_allval()
    #         print("GAT ARE:", GAT_metrics.ARE_val)
