from sklearn import linear_model
from sklearn import preprocessing
import time

from sketchs import *



if __name__ == '__main__':
    '''准备数据'''

    file_now = random.randint(0, 10 - 1)
    # file_now = 179

    test_y_path = "./testdata_set_5s/testdata_flows_5s/" + str(file_now).zfill(
        5) + ".txt"

    # 处理cm数据
    flows_index_path = "./testdata_set_5s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flows_index_path)
    file_name = "./sketch_params/160000_cm_sketch.txt"
    dict_cm = rw_files.get_dict(file_name)
    cm_d = 3

    cm_w=16*10**4

    cm_test_path = "./testdata_set_5s/testdata_160000_cm_5s/" + str(file_now).zfill(5) + ".txt"
    cm_sketch_now = np.loadtxt(cm_test_path)
    sketch_now = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_now)

    test_flows_data = rw_files.get_dict(test_y_path)
    test_flows_data_list = list(test_flows_data.values())

    get_value = np.vectorize(lambda x: flows_alltime_dict.get(x, 0))

    # 获取test时刻流信息
    test_x_key = list(test_flows_data.keys())
    test_index_array = get_value(np.array(test_x_key))

    '''获取所在时刻的d个查询值'''
    # 获取test
    test_x=sketch_now.query_d_np(test_index_array).T

    '''test dataset'''
    test_y = np.array(test_flows_data_list).reshape(-1, 1)
    test_cm = np.amin(sketch_now.query_d_np(test_index_array).T, axis=1).reshape(-1, 1)

    feature_scaler = preprocessing.StandardScaler().fit(test_x)
    test_x = feature_scaler.transform(test_x)

    # clf = linear_model.Ridge(alpha=.5)
    clf = linear_model.LinearRegression()
    clf.fit(test_x, test_y)
    pre=clf.predict(test_x)

    # 输出预测结果
    relative_ML_error = pre / test_y - 1
    # relative_pre_error = np.exp(pre_arr-test_y) - 1
    relative_cm_error = test_cm / test_y - 1
    print("relative cm error:" + "{:.6f}".format(np.mean(np.abs(relative_cm_error))))
    print("relative pre error:" + "{:.6f}".format(np.mean(np.abs(relative_ML_error))))
    print("cm recall:" + "{:.6f}".format(np.sum(relative_cm_error == 0) / test_y.shape[0]))
    print("pre recall:" + "{:.6f}".format(np.sum(relative_ML_error == 0) / test_y.shape[0]))

    cm_error = np.sum(np.abs(relative_cm_error) < 0.001)
    print("cm rate (error rate less than 0.1%):" + "{:.6f}".format(cm_error / test_y.shape[0]))
    pre_error = np.sum(np.abs(relative_ML_error) < 0.001)
    print("pre rate (error rate less than 0.1%):" + "{:.6f}".format(pre_error / test_y.shape[0]))

    end_time = time.time()



