from sketchs import *


if __name__ == '__main__':
    '''准备数据'''

    file_now = random.randint(0, 1 - 1)
    # file_now = 179

    test_y_path = "./caida/testdata_flows_5s/" + str(file_now).zfill(
        5) + ".txt"
    # test_y_path = "./traindata_set_5s/traindata_flows_5s/" + str(file_now).zfill(
    #     5) + ".txt"
    test_flows_data = rw_files.get_dict(test_y_path)
    test_flows_keys_list=list(test_flows_data.keys())
    test_flows_values_list=list(test_flows_data.values())
    test_flows_values_np=np.array(test_flows_values_list)
    a=np.sum(test_flows_values_np <= 2)
    rate=np.sum(test_flows_values_np<=2)/test_flows_values_np.shape[0]
    print(rate)
