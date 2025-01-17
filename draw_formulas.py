import numpy as np
import matplotlib.pyplot as plt

color1="#002c53"
color2="#ffa510"
color3="#0c84c6"
color4="#ffbd66"
color5="#f74d4d"
color6="#2455a4"
color7="#41b7ac"
# color_list=[color3,color2,color7,color5]
color_list=[(75/255,101/255,175/255),(244/255,111/255,68/255),(164/255,5/255,69/255),(127/255,203/255,164/255),(255/255,5/255,6/255),(1/255,203/255,164/255)]

sketchs_list=["CM", "Tower", "DNN", "DNN_Tower", "GAT", "GAT_Tower"]


def draw_formula1():
    # 定义x的范围
    x = np.linspace(0, 2, 100)

    # 计算三个函数的y值
    y0 = np.exp(-x)
    y1 = np.exp(-x) + x * np.exp(-x)
    y2 = np.exp(-x) + x * np.exp(-x) + (x ** 2) / 2 * np.exp(-x)
    y3 = np.exp(-x) + x * np.exp(-x) + (x ** 2) / 2 * np.exp(-x) + (x ** 3) / 6 * np.exp(-x)

    # 绘制三个函数的图形
    fig, ax = plt.subplots(figsize=(3, 3))
    plt.plot(x, y0**3, label='$n_c=0$ (CM sketch)')
    plt.plot(x, y1**3, label='$n_c=1$')
    plt.plot(x, y2**3, label='$n_c=2$')
    plt.plot(x, y3**3, label='$n_c=3$')

    # 添加标题和标签
    ax.legend()
    ax.set_xlabel("$\lambda=\\frac{N-1}{w}$",fontsize=12, fontname='Times New Roman')
    # 设置图例
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=10,
                       prop={'family': 'Times New Roman'})

    # 设置字体和线宽
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    # 设置坐标轴标签
    # ax.set_xlabel(X_Label, fontsize=12, fontname='Times New Roman')
    ax.set_ylabel("$r$", fontsize=12, fontname='Times New Roman')

    # 设置图形窗口大小和位置
    fig.set_size_inches(3, 3)  # Adjust this based on your requirement
    fig.subplots_adjust(left=0.12, right=0.95, top=0.85, bottom=0.2)

    # 设置网格和边框
    ax.grid(True)
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    ax.patch.set_facecolor('none')

    # 显示图形
    plt.show()

def draw_formula2():
    # 定义x的范围
    x = np.linspace(0, 2, 100)

    def calculate_y(lam, k):
        d=3
        total = 0
        for i in range(k + 1):
            total+=(lam**i) / np.math.factorial(i) * np.exp(-lam) * ((1 - (1 - np.exp(-lam))**(d-1))**i)
        return 1 - (1 - total)**d

    y0 = calculate_y(x, 0)
    y1 = calculate_y(x, 1)
    y2 = calculate_y(x, 2)
    y3 = calculate_y(x, 3)

    # 绘制三个函数的图形
    fig, ax = plt.subplots(figsize=(3, 3))
    plt.plot(x, y0, label='$n_c=0$ (CM sketch)')
    plt.plot(x, y1, label='$n_c=1$')
    plt.plot(x, y2, label='$n_c=2$')
    plt.plot(x, y3, label='$n_c=3$')

    # 添加标题和标签
    ax.legend()
    ax.set_xlabel("$\lambda=\\frac{N-1}{w}$",fontsize=12, fontname='Times New Roman')
    # 设置图例
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=10,
                       prop={'family': 'Times New Roman'})

    # 设置字体和线宽
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    # 设置坐标轴标签
    # ax.set_xlabel(X_Label, fontsize=12, fontname='Times New Roman')
    ax.set_ylabel("$P$", fontsize=12, fontname='Times New Roman')

    # 设置网格和边框
    ax.grid(True)
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    ax.patch.set_facecolor('none')

    fig.set_size_inches(3, 3)  # Adjust this based on your requirement
    fig.subplots_adjust(left=0.12, right=0.95, top=0.85, bottom=0.2)

    # 显示图形
    plt.show()

draw_formula1()
draw_formula2()