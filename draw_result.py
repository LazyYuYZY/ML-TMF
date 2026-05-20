import numpy as np
import matplotlib.pyplot as plt

color1 = "#002c53"
color2 = "#ffa510"
color3 = "#0c84c6"
color4 = "#ffbd66"
color5 = "#f74d4d"
color6 = "#2455a4"
color7 = "#41b7ac"
# color_list=[color3,color2,color7,color5]
color_list = [(35 / 255, 86 / 255, 167/ 255), (133/ 255, 182 / 255, 225 / 255), (237 / 255, 52/ 255, 47 / 255),
              (246 / 255, 154 / 255, 155 / 255), (23 / 255, 134 / 255, 66 / 255), (160 / 255, 210 / 255, 147 / 255)]

sketchs_list = ["CM", "Tower", "DNN", "DNN_Tower", "GAT", "GAT_Tower"]


def draw_bar():
    cm = np.array([112, 228, 288, 400, 512, 600]) * 0.12
    ssc = np.array([20, 40, 56, 80, 100, 128]) * 0.12
    mmc = np.array([14, 28, 40, 56, 70, 84]) * 0.12
    x = np.array([5,10,15,20,25,30])
    # ax = [3 32 0 80]

    sketchs = ["CM", "Tower", "SSC-CM", "SSC-Tower", "MMC-CM", "MMC-Tower"]
    fmts = ['o', 'v', 'p', 'd', 'h', '*']
    xl = 'The length of measurement periods(s)'
    yl = 'Memory(MB)'
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(x,cm,  linewidth=1.0, color=color_list[0], marker=fmts[0],
            markersize=5, markerfacecolor="none", label=f"{sketchs[0]}")
    ax.plot(x,ssc, linewidth=1.0, color=color_list[2], marker=fmts[2],
            markersize=5, markerfacecolor="none", label=f"{sketchs[2]}")
    ax.plot(x,mmc, linewidth=1.0, color=color_list[4], marker=fmts[4],
            markersize=5, markerfacecolor="none", label=f"{sketchs[4]}")
    ax.set_ylim([0,80])
    # ax.set_xlabel(x_label, fontsize=14, fontname='Times New Roman')
    # 设置字体和线宽
    ax.tick_params(axis='both', which='major', labelsize=12)
    # 设置图例
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=14,
                       prop={'family': 'Times New Roman', 'size': 12})

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    # 设置坐标轴标签
    ax.set_xlabel(xl, fontsize=14, fontname='Times New Roman')
    ax.set_ylabel(yl, fontsize=14, fontname='Times New Roman')

    # 设置网格和边框
    ax.grid(True)
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    ax.patch.set_facecolor('none')
    fig.set_size_inches(3, 4)
    fig.subplots_adjust(left=0.145, right=0.99, top=0.85, bottom=0.2)
    # 调整子图之间的间距
    # fig.tight_layout()
    # 显示图形
    plt.show()


def draws_all(x_list, y_np, x_label, real=0, X_Label=""):
    if real:
        folder_path = "./results/wide"
    else:
        folder_path = "./results/zipf"
    # 创建画布和子图
    row_num = 2
    col_num = 3
    if x_label=="w(10^4)":
        x_list=np.array(x_list)*0.12
    sketchs = ["CM", "Tower", "SSC-CM", "SSC-Tower", "MMC-CM", "MMC-Tower"]
    # titles = ["ARE(FS)", "AAE(FS)", "F1(FS)", "F1(HeavyHitter)","WMRE(ED)","RE(E)"]
    Y_Label = ["lg(ARE)", "lg(AAE)", "F1 score", "F1 score", "lg(WMRE)", "lg(RE)"]
    ylims = [[-5, 2], [-3, 3], [0, 1.1], [0, 1.1], [-4, 1], [-20, 3]]
    fmts = ['o', 'v', 'p', 'd', 'h', '*']

    y_mean = np.mean(y_np, axis=3)
    y_variance = np.var(y_np, axis=3)
    y_err = 1.96 * np.sqrt(y_variance / y_np.shape[3])
    print(x_list.shape, y_np.shape, y_mean.shape)
    # 遍历六幅图
    for i in range(row_num):
        for j in range(col_num):
            if i * col_num + j >= 1 and ((real==0) or (X_Label!="Memory(MB)")):
                break
            fig, ax = plt.subplots(figsize=(3, 3))
            # if row_num==1:
            #     ax =axes[j]
            # else:
            #     ax = axes[i, j]
            # ax.set_title(f"Plot {titles[i * col_num + j]}")
            ax.set_ylim(ylims[i * col_num + j])
            # 遍历sketch结果
            for k in range(len(sketchs)):
                fname_mean = x_label[0] + str(k) + str(i * col_num + j) + "mean.txt"
                np.savetxt(folder_path + "/" + fname_mean, y_mean[k, :, i * col_num + j])
                if 1 < i * col_num + j < 4:
                    ax.plot(x_list, y_mean[k, :, i * col_num + j], linewidth=1.0, color=color_list[k], marker=fmts[k],
                            markersize=5, markerfacecolor="none", label=f"{sketchs[k]}")
                    ax.errorbar(x_list, y_mean[k, :, i * col_num + j], y_err[k, :, i * col_num + j],
                                color=color_list[k], linewidth=1, capsize=2, ms=4)
                    ax.set_ylabel('rate')
                    fname_err = x_label[0] + str(k) + str(i * col_num + j) + "err.txt"
                    np.savetxt(folder_path + "/" + fname_err, y_err[k, :, i * col_num + j])

                else:
                    y_err_lg = np.full((2, y_np.shape[0], y_np.shape[1], y_np.shape[2]), 0)
                    if np.all(y_mean > y_err):
                        y_err_lg[0] = np.log10(y_mean) - np.log10(y_mean - y_err)
                    else:
                        y_err_lg[0] = np.log10(y_mean + y_err) - np.log10(y_mean)
                    y_err_lg[1] = np.log10(y_mean + y_err) - np.log10(y_mean)
                    ax.plot(x_list, np.log10(y_mean[k, :, i * col_num + j]), linewidth=1.0, color=color_list[k],
                            marker=fmts[k], markersize=5, markerfacecolor="none", label=f"{sketchs[k]}")
                    ax.errorbar(x_list, np.log10(y_mean[k, :, i * col_num + j]), y_err_lg[:, k, :, i * col_num + j],
                                color=color_list[k], linewidth=1, capsize=2, ms=4)
                    # ax.set_ylabel('rate(lg)')
                    fname_err_lg = x_label[0] + str(k) + str(i * col_num + j) + "err_lg.txt"
                    np.savetxt(folder_path + "/" + fname_err_lg, y_err_lg[:, k, :, i * col_num + j])

            ax.set_xlabel(x_label, fontsize=14, fontname='Times New Roman')
            # 设置字体和线宽
            ax.tick_params(axis='both', which='major', labelsize=12)
            # 设置图例
            legend = ax.legend(loc='upper center', bbox_to_anchor=(0.45, 1.2), ncol=3, fontsize=14,
                               prop={'family': 'Times New Roman','size':12})

            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontname('Times New Roman')
            ax.spines['top'].set_linewidth(2.0)
            ax.spines['right'].set_linewidth(2.0)
            ax.spines['bottom'].set_linewidth(2.0)
            ax.spines['left'].set_linewidth(2.0)

            # 设置坐标轴标签
            ax.set_xlabel(X_Label, fontsize=14, fontname='Times New Roman')
            ax.set_ylabel(Y_Label[i * col_num + j], fontsize=14, fontname='Times New Roman')

            # 设置网格和边框
            ax.grid(True)
            ax.set_facecolor('none')
            fig.patch.set_facecolor('none')
            ax.patch.set_facecolor('none')
            fig.set_size_inches(3, 4)

            if i * col_num + j in [0, 1, 4]:
                fig.subplots_adjust(left=0.12, right=0.99, top=0.85, bottom=0.2)
            else:
                fig.subplots_adjust(left=0.145, right=0.99, top=0.85, bottom=0.2)
    # 调整子图之间的间距
    # fig.tight_layout()
    # 显示图形
    plt.show()


def plot_bar_with_errorbars(sketch="Count"):
    fold_path = "./results/wide/" + sketch + "/"
    paths = [fold_path + sketch + "_160000.txt", fold_path + "DNN_160000.txt", fold_path + "GAT_160000.txt"]
    # 计算每个序列的平均值和标准差
    plots_np = np.full((3, 10), 0, dtype=float)
    for i in range(3):
        plots_np[i] = np.loadtxt(paths[i])[0]
    y_mean = np.mean(plots_np, axis=1)
    y_variance = np.var(plots_np, axis=1)
    y_err = 1.96 * np.sqrt(y_variance / plots_np.shape[1])

    # 设置柱状图的位置
    x = [sketch, "SSC-" + sketch, "MMC-" + sketch]

    # 绘制柱状图
    fig, ax = plt.subplots()
    ax.bar(x, y_mean, yerr=y_err, capsize=5, align='center', alpha=0.5,
           color=[color_list[0], color_list[2], color_list[4]])

    # 设置字体和线宽
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    # 设置坐标轴标签
    # ax.set_xlabel('Approaches', fontsize=12, fontname='Times New Roman')
    ax.set_ylabel('ARE', fontsize=12, fontname='Times New Roman')

    # 设置网格和背景颜色
    ax.grid(True)
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    ax.patch.set_facecolor('none')

    # 设置图形窗口大小和位置
    fig.set_size_inches(3, 3)  # Adjust this based on your requirement
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.2)

    # 设置x轴刻度标签
    ax.set_xticks(x)

    # 显示图形
    plt.show()


def plot_bar_MLsketch():
    fold_path = "./results/wide/ML/"
    paths = [fold_path + "ML", fold_path + "Talent", fold_path + "Deep", fold_path + "DNN", fold_path + "GAT"]
    # 计算每个序列的平均值和标准差
    plots_np = np.full((len(paths), 10), 0, dtype=float)
    for i in range(len(paths)):
        plots_np[i] = np.loadtxt(paths[i] + ".txt")[0]
    y_mean = np.mean(plots_np, axis=1)
    y_variance = np.var(plots_np, axis=1)
    y_err = 1.96 * np.sqrt(y_variance / plots_np.shape[1])

    y_mean_lg = np.log10(y_mean)
    y_err_lg = np.full((2, len(paths)), 0)
    if np.all(y_mean > y_err):
        y_err_lg[0] = np.log10(y_mean) - np.log10(y_mean - y_err)
    else:
        y_err_lg[0] = np.log10(y_mean + y_err) - np.log10(y_mean)
    y_err_lg[1] = np.log10(y_mean + y_err) - np.log10(y_mean)
    # 设置柱状图的位置
    # x = np.arange(len(paths))  # 序列数量
    x = ["ML-sketch", "Talentsketch", "Deepsketch", "SSC-CM", "MMC-CM"]

    fig, ax = plt.subplots()
    ax.bar(x, -y_mean_lg, yerr=-y_err_lg, capsize=5, align='center', alpha=0.5,
           color=[color1, color2, color7, color_list[2], color_list[4]])

    # 设置字体和线宽
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    # 设置坐标轴标签
    # ax.set_xlabel('Schemes', fontsize=12, fontname='Times New Roman')
    ax.set_ylabel('-lg(ARE)', fontsize=12, fontname='Times New Roman')

    # 设置网格和背景颜色
    ax.grid(True)
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    ax.patch.set_facecolor('none')

    # 设置图形窗口大小和位置
    fig.set_size_inches(5, 3)  # Adjust this based on your requirement
    fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    # 设置x轴刻度标签
    ax.set_xticks(x)

    # 显示图形
    plt.show()

    fname_mean = "mean.txt"
    np.savetxt(fold_path + fname_mean, y_mean)
    fname_err = "err.txt"
    np.savetxt(fold_path + fname_err, y_err)

def plot_bar_fourtarces():
    y = np.array([[48, 10, 8],
                  [96, 20, 12],
                  [128, 28, 20],
                  [112, 20, 14]]) * 0.12
    xl = 'MAWI traces'
    yl = 'Memory(MB)'

    # 绘图2
    fig, ax = plt.subplots()

    # 绘制柱状图
    bar_width = 0.25
    index = np.arange(y.shape[0])

    bars1 = ax.bar(index, y[:, 0], bar_width, label='CM', color=color_list[0])
    bars2 = ax.bar(index + bar_width, y[:, 1], bar_width, label='SSC-CM', color=color_list[2])
    bars3 = ax.bar(index + 2 * bar_width, y[:, 2], bar_width, label='MMC-CM', color=color_list[4])

    # 设置x轴刻度标签
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(['2023/01/03', '2023/07/03', '2023/11/01', '2024/01/03'])

    ax.tick_params(axis='both', which='major', labelsize=12)
    # 设置图例
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=14,
                       prop={'family': 'Times New Roman', 'size': 12})

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    # 设置坐标轴标签
    ax.set_xlabel(xl, fontsize=14, fontname='Times New Roman')
    ax.set_ylabel(yl, fontsize=14, fontname='Times New Roman')

    # 设置网格和边框
    ax.grid(True)
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    ax.patch.set_facecolor('none')

    # 设置图形窗口大小和位置
    fig.set_size_inches(3, 4)  # Adjust this based on your requirement
    fig.subplots_adjust(left=0.145, right=0.99, top=0.85, bottom=0.2)

    # 显示图形
    plt.show()


def draw_are_vs_memory(w0=10000):
    """
    绘制 CM-SSC, CM-MMC, ELASTIC, FCM, Seqsketch, Embedsketch 六种方法的 lg(ARE) 随内存变化的折线图
    """
    memory_list = [8, 12, 16, 24, 32]
    w_list = [m * w0 for m in memory_list]          # 用于 CM-SSC/MMC 的文件名参数
    x_mb = np.array(memory_list) * 0.12             # 横坐标内存大小 (MB)

    # 六种方法的显示标签与对应的数据读取规则
    # 前两种：CM-SSC, CM-MMC
    methods = [
        ("CM-SSC",  "Memory/DNN",  True),
        ("CM-MMC",  "Memory/GAT",  True),
        ("ELASTIC", "elastic/elastic_memory", False),
        ("FCM",     "fcm/fcm_memory", False),
        ("Seqsketch", "seqsketch/seq_memory_errors", False),
        ("Embedsketch", "embedsketch/embed_memory_errors", False)
    ]

    n_methods = len(methods)
    n_mem = len(memory_list)

    # 确定重复实验次数（通过读取第一个文件第一行的列数获得）
    # 先尝试读取一个 CM-SSC 文件获取重复次数
    try:
        sample_data = np.loadtxt(f"./results/wide/Memory/DNN_{w_list[0]}.txt")
        if sample_data.ndim == 1:
            n_repeats = len(sample_data)
        else:
            n_repeats = sample_data.shape[1]  # 第一行作为多次实验
    except:
        n_repeats = 10  # 默认值

    # 初始化数据数组 shape: (n_methods, n_mem, n_repeats)
    data_all = np.full((n_methods, n_mem, n_repeats), np.nan)

    for i, (label, rel_path, per_mem_file) in enumerate(methods):
        if i>3:
            continue
        if per_mem_file:
            # 每个内存一个独立文件
            for j, w in enumerate(w_list):
                path = f"./results/wide/{rel_path}_{w}.txt"
                try:
                    arr = np.loadtxt(path)
                    if arr.ndim == 1:
                        vals = arr
                    else:
                        vals = arr[0, :]        # 取第一行
                    data_all[i, j, :len(vals)] = vals
                except Exception as e:
                    print(f"读取失败: {path}, {e}")
        else:
            # 汇总文件，每行对应一个内存档位
            path = f"./results/wide/{rel_path}.txt"
            try:
                arr = np.loadtxt(path)
                for j in range(n_mem):
                    if arr.ndim == 1:
                        vals = arr if j == 0 else np.array([])
                    else:
                        vals = arr[j, :] if j < arr.shape[0] else np.array([])
                    data_all[i, j, :len(vals)] = vals
            except Exception as e:
                print(f"读取失败: {path}, {e}")

    # 计算均值、方差和置信区间
    y_mean = np.nanmean(data_all, axis=2)
    y_var  = np.nanvar(data_all, axis=2)
    y_err  = 1.96 * np.sqrt(y_var / n_repeats)

    # 对数坐标下的误差棒转换
    # 如果 ARE 可能为0或负，需要处理，假设 ARE 为正数
    y_mean_lg = np.log10(y_mean)
    y_err_lg = np.full((2, n_methods, n_mem), 0.0)
    for i in range(n_methods):
        for j in range(n_mem):
            mean_val = y_mean[i, j]
            err_val = y_err[i, j]
            if mean_val > err_val:
                lower = np.log10(mean_val) - np.log10(mean_val - err_val)
            else:
                lower = np.log10(mean_val + err_val) - np.log10(mean_val)
            upper = np.log10(mean_val + err_val) - np.log10(mean_val)
            y_err_lg[0, i, j] = lower
            y_err_lg[1, i, j] = upper

    # 绘图设置
    fmts = ['o', 'v', 'p', 'd', 'h', '*']
    fig, ax = plt.subplots(figsize=(5, 4))

    for i, (label, _, _) in enumerate(methods):
        if i>3:
            continue
        ax.errorbar(x_mb, y_mean_lg[i, :],
                    yerr=y_err_lg[:, i, :],
                    color=color_list[i % len(color_list)],
                    marker=fmts[i], markersize=7, markerfacecolor='none',
                    linewidth=1.5, capsize=3, label=label)

    # 坐标轴与字体设置（参考 draws_all）
    ax.set_xlabel("Memory (MB)", fontsize=14, fontname='Times New Roman')
    ax.set_ylabel("lg(ARE)", fontsize=14, fontname='Times New Roman')
    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # 边框加粗
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    # 图例置于顶部
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2,
              prop={'family': 'Times New Roman', 'size': 12})

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    fig.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.15)

    plt.show()

def draw_ml_are_vs_memory(w0=10000):
    """
    绘制 CM, CM-SSC, CM-GNN, CM-MMC 四种机器学习相关方法的 lg(ARE) 随内存变化的折线图
    """
    memory_list = [8, 12, 16, 24, 32]
    w_list = [m * w0 for m in memory_list]
    x_mb = np.array(memory_list) * 0.12   # 内存（MB）

    # 四种方法的标签、相对路径及是否为每个内存独立文件
    # 请根据你的实际文件名修改路径
    methods = [
        ("CM",      "Memory/CM",      True),
        ("CM-SSC",  "Memory/DNN",     True),   # 或 "Memory/CM-SSC"
        ("CM-GNN",  "Memory/GNN",     True),   # 假设 GNN 文件名为 GNN
        ("CM-MMC",  "Memory/GAT",     True),   # 或 "Memory/CM-MMC"
    ]

    n_methods = len(methods)
    n_mem = len(memory_list)

    # 尝试获取重复实验次数
    try:
        sample_data = np.loadtxt(f"./results/wide/Memory/CM_{w_list[0]}.txt")
        if sample_data.ndim == 1:
            n_repeats = len(sample_data)
        else:
            n_repeats = sample_data.shape[1]
    except:
        n_repeats = 10

    # 初始化数据数组
    data_all = np.full((n_methods, n_mem, n_repeats), np.nan)

    for i, (label, rel_path, per_mem_file) in enumerate(methods):
        if per_mem_file:
            for j, w in enumerate(w_list):
                path = f"./results/wide/{rel_path}_{w}.txt"
                try:
                    arr = np.loadtxt(path)
                    if arr.ndim == 1:
                        vals = arr
                    else:
                        vals = arr[0, :]          # 第一行为该内存下的多次实验
                    data_all[i, j, :len(vals)] = vals
                except Exception as e:
                    print(f"读取失败: {path}, {e}")
        else:
            # 如果将来有汇总文件也可处理
            path = f"./results/wide/{rel_path}.txt"
            try:
                arr = np.loadtxt(path)
                for j in range(n_mem):
                    if arr.ndim == 1:
                        vals = arr if j == 0 else np.array([])
                    else:
                        vals = arr[j, :] if j < arr.shape[0] else np.array([])
                    data_all[i, j, :len(vals)] = vals
            except Exception as e:
                print(f"读取失败: {path}, {e}")

    # 计算均值、方差、95%置信区间
    y_mean = np.nanmean(data_all, axis=2)
    y_var  = np.nanvar(data_all, axis=2)
    y_err  = 1.96 * np.sqrt(y_var / n_repeats)

    # 转换为对数坐标及误差棒
    y_mean_lg = np.log10(y_mean)
    y_err_lg = np.full((2, n_methods, n_mem), 0.0)
    for i in range(n_methods):
        for j in range(n_mem):
            mean_val = y_mean[i, j]
            err_val = y_err[i, j]
            if mean_val > err_val:
                lower = np.log10(mean_val) - np.log10(mean_val - err_val)
            else:
                lower = np.log10(mean_val + err_val) - np.log10(mean_val)
            upper = np.log10(mean_val + err_val) - np.log10(mean_val)
            y_err_lg[0, i, j] = lower
            y_err_lg[1, i, j] = upper

    # 绘图
    fmts = ['o', 'v', 'p', 'd', 'h', '*']
    fig, ax = plt.subplots(figsize=(5, 4))

    for i, (label, _, _) in enumerate(methods):
        ax.errorbar(x_mb, y_mean_lg[i, :],
                    yerr=y_err_lg[:, i, :],
                    color=color_list[i % len(color_list)],
                    marker=fmts[i], markersize=7, markerfacecolor='none',
                    linewidth=1.5, capsize=3, label=label)

    # 样式设置
    ax.set_xlabel("Memory (MB)", fontsize=14, fontname='Times New Roman')
    ax.set_ylabel("lg(ARE)", fontsize=14, fontname='Times New Roman')
    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2,
              prop={'family': 'Times New Roman', 'size': 12})
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    fig.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.15)

    plt.show()


if __name__ == '__main__':
    dataset_type_dict = {1: "wide", 0: "zipf"}

    w0 = 1 * 10 ** 4
    memory_list = [8, 12, 16, 24, 32]
    alpha_list = [1.6, 1.8, 2.0, 2.2, 2.4]
    T_list = [5, 10, 15, 20, 25, 30]
    p_drop=[0.1,0.01, 0.001, 0.0001, 0.00001]
    memory_bloom_list=[16,24,32,40,48]
    # draw_bar()
    # plot_bar_fourtarces()
    # plot_bar_MLsketch()
    # plot_bar_with_errorbars(sketch="Count")
    # plot_bar_with_errorbars(sketch="CU")
    #
    # plots_np = np.full((len(sketchs_list), len(memory_list), 6, 10), 0, dtype=float)
    # for i in range(len(memory_list)):
    #     for k in range(len(sketchs_list)):
    #         w=memory_list[i]*w0
    #         results_path = "./results/wide/" + "Memory" + "/" + sketchs_list[k] + "_" + str(w) + ".txt"
    #         plots_np[k, i, :, :]=np.loadtxt(results_path)
    # draws_all(x_list=memory_list,y_np=plots_np, x_label="w(10^4)", real=1,X_Label="Memory(MB)")
    # #
    # plots_np = np.full((len(sketchs_list), len(memory_list), 6, 10), 0, dtype=float)
    # for i in range(len(memory_list)):
    #     for k in range(len(sketchs_list)):
    #         w = memory_list[i] * w0
    #         results_path = "./results/zipf/" + "Memory" + "/" + sketchs_list[k] + "_" + str(w) + ".txt"
    #         plots_np[k, i, :, :] = np.loadtxt(results_path)
    # draws_all(x_list=memory_list, y_np=plots_np, x_label="w(10^4)", real=0, X_Label="Memory(MB)")
    #
    # plots_np = np.full((len(sketchs_list), len(alpha_list), 6, 10), 0, dtype=float)
    # for i in range(len(alpha_list)):
    #     for k in range(len(sketchs_list)):
    #         results_path = "./results/zipf/" + "Alpha" + "/" + sketchs_list[k] + "_" + str(
    #             int(alpha_list[i] * 10)) + ".txt"
    #         plots_np[k, i, :, :] = np.loadtxt(results_path)
    # draws_all(x_list=alpha_list, y_np=plots_np, x_label="alpha", real=0, X_Label="Skewness parameter ($\\alpha$)")
    #
    # plots_np = np.full((len(sketchs_list), len(T_list), 6, 10), 0, dtype=float)
    # for i in range(len(T_list)):
    #     for k in range(len(sketchs_list)):
    #         T_slice = T_list[i]
    #         results_path = "./results/wide/" + "T" + "/" + sketchs_list[k] + "_" + str(T_slice) + ".txt"
    #         plots_np[k, i, :, :] = np.loadtxt(results_path)
    # draws_all(x_list=T_list, y_np=plots_np, x_label="T", real=1, X_Label="The length of measurement periods(s)")

    # plots_np = np.full((len(sketchs_list), len(p_drop), 6, 10), 0, dtype=float)
    # for i in range(len(p_drop)):
    #     for k in range(len(sketchs_list)):
    #         drop_flow = p_drop[i]
    #         results_path = "./results/wide/" + "Bloom" + "/" + sketchs_list[k] + "_" + str(int(-np.log10(drop_flow))) + ".txt"
    #         plots_np[k, i, :, :] = np.loadtxt(results_path)
    # print(-np.log10(np.array(p_drop)))
    # p_drop=-np.log10(np.array(p_drop))
    # draws_all(x_list=p_drop, y_np=plots_np, x_label="p", real=1, X_Label="-lg(r)")

    # draw_are_vs_memory()

    draw_ml_are_vs_memory()
