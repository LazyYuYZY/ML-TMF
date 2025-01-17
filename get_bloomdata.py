import numpy as np

from sketchs import *
import matplotlib.pyplot as plt
import numpy as np





if __name__ == '__main__':
    # 是否有数据集
    dataset_over = 0

    # 是否采用已有bloom参数
    flag=0
    bloom_d=3
    w=1 * 10 ** 4
    flag_dict={1:"old",0:"new"}
    # memory_list=[1,2,4,8,12,16,20,24,32]
    memory_list = [16]
    memory_register_list=[100,1000,10000,100000,1000000]
    result_loss=[]
    result_cost=[]
    for memory_register in memory_register_list:
        for i in memory_list:
            bloom_w = int(w * i)
            # 创建使用的 Sketch
            bloom_sketch_now = np.full((bloom_d, bloom_w * (2 ** (bloom_d - 1))), 0)  # bloom存储的counter值

            sketch_path = "./sketch_params/" + str(bloom_w).zfill(6) + "_bloom_sketch.txt"
            if not os.path.exists(sketch_path):
                bloom_used = bloom_sketch(bloom_d=bloom_d, bloom_w=bloom_w, flag=0, dict_bloom={},
                                          bloom_sketch_load=bloom_sketch_now)
                bloom_used.save(file_name=sketch_path)
            else:
                dict_bloom = rw_files.get_dict(sketch_path)
                bloom_used = bloom_sketch(bloom_d=bloom_d, bloom_w=bloom_w, flag=1, dict_bloom=dict_bloom,
                                          bloom_sketch_load=bloom_sketch_now)
            # continue


            # 选择要输入的流的位置
            for file_now in range(1):
                # 读取流文件
                all_recive=set()
                register=[]
                json_file_path = "./test-bloom-register.text"

                cost=0

                flows_data=np.loadtxt(json_file_path).tolist()
                for packet in flows_data:
                    packet=int(packet)
                    if bloom_used.query_one(packet)==0:
                        bloom_used.insert_one(packet)
                        register.append(packet)
                        all_recive.add(packet)
                        cost=cost+1

                    if len(register)==memory_register:
                        register=[]
                        bloom_used.clear()
                print("number of flow in register:",memory_register)
                print("rate of loss flows:",1-len(all_recive)/174990)
                print("cost of sendpkt(KB):",13*cost/1000,"(base:",174990*13/1000,"KB)")
                result_loss.append(1-len(all_recive)/174990)
                result_cost.append(13*cost/1e6)
    result_loss_path="./results/sendpkt_loss"
    result_cost_path="./results/sendpkt_cost"
    np.savetxt(result_loss_path, np.array(result_loss), fmt='%d')
    np.savetxt(result_cost_path, np.array(result_cost), fmt='%d')

    # 生成示例数据
    x = np.log10(np.array(memory_register_list))
    y1 = np.array(result_loss)
    y2 = np.array(result_cost)

    # 创建图形和第一个坐标轴
    fig, ax1 = plt.subplots()

    # 绘制第一个数据系列
    ax1.plot(x, y1, label='loss',linewidth=1.0, color='b',
                            marker='o', markersize=5, markerfacecolor="none")
    ax1.set_xlabel('lg(N)')
    ax1.set_ylabel('loss', color='b')
    ax1.tick_params('y', colors='b')

    ax1.set_xlim([1.5, 6.5])  # 设置 x 轴范围从 0 到 10
    ax1.set_ylim([-0.05, 1])  # 设置 y1 轴范围从 -1.5 到 1.5


    # 创建第二个坐标轴，与第一个坐标轴共享 x 轴
    ax2 = ax1.twinx()

    # 绘制第二个数据系列
    ax2.plot(x, y2, label='overhead',linewidth=1.0, color='r',
                            marker='*', markersize=5, markerfacecolor="none")
    ax2.set_ylabel('cost(MB)', color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_ylim([-0.25, 5])  # 设置 y1 轴范围从 -1.5 到 1.5

    # 显示图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best', ncol=1, fontsize=14,
                        prop={'family': 'Times New Roman', 'size': 12})

    # ax1.set_xlabel(x_label, fontsize=14, fontname='Times New Roman')
    # 设置字体和线宽s
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    # 设置图例
    # legend = ax1.legend(loc='upper center', bbox_to_anchor=(0.45, 1.2), ncol=3, fontsize=14,
    #                    prop={'family': 'Times New Roman', 'size': 12})
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontname('Times New Roman')
    # legend = ax2.legend(loc='upper center', bbox_to_anchor=(0.45, 1.2), ncol=3, fontsize=14,
    #                     prop={'family': 'Times New Roman', 'size': 12})

    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontname('Times New Roman')
    ax1.spines['top'].set_linewidth(2.0)
    ax1.spines['right'].set_linewidth(2.0)
    ax1.spines['bottom'].set_linewidth(2.0)
    ax1.spines['left'].set_linewidth(2.0)

    # 设置坐标轴标签
    ax1.set_xlabel(f"-lg(S)", fontsize=14, fontname='Times New Roman')
    ax1.set_ylabel(f"Loss", fontsize=14, fontname='Times New Roman')
    ax2.set_ylabel(f"Overhead(MB)", fontsize=14, fontname='Times New Roman')

    # 设置网格和边框
    ax1.grid(True)
    ax1.set_facecolor('none')
    fig.patch.set_facecolor('none')
    ax1.patch.set_facecolor('none')
    fig.set_size_inches(3, 4)

    fig.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.2)

    # 显示图形
    plt.show()
