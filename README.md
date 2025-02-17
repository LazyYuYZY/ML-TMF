Sketch_name={cm, tower, cu, count}

## 真实数据集：

1.wareshake打开终端editcap.exe 切割 MAWI数据***数据集过大就不放置分割后的文件夹***

.\editcap -i 5 .\202311011400.pcap .\traindata_set_5s\traindata_pcap_5s\20240103_5s.pcap

(.\editcap -i $Tslice$ .\ $trace$ $foldername$ \ $filename$)

2.changefname.py 修改文件名

3.get_flows_data.py
将pcap文件读取处理，根据流的五元组与数据包数量构建字典，存入flow(pcap->txt)

4.flows_hash.py 五元组映射为数字
为方便后续处理，将流的五元组映射为整数

5.1.DNN/DNN_$Sketch\_name$.py 训练DNN模型
（用的MAWI231101的数据，随机选择一个时间片下的数据）

5.2.GAT/GAT_DNN_d.py 训练优化后的GAT模型（可选）
（用的zipf的数据，2.0偏度下160000条流）

6.$Sketch\_name$_dnn_gat_test.py 测试性能

## 其他：

合成数据集：get_zipf_flowsdata

其他机器学习方法实现
