import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def read_mem_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    mem_list = {}
    for line in lines:
        if ':' in line:
            step, value = line.split(':')
            mem_list[int(step.strip())] = float(value.strip())
    
    return mem_list

def plot_mem_trend(mem_lists, labels, title):
    plt.figure(figsize=(10, 6))
    
    for mem_list, label in zip(mem_lists, labels):
        steps = sorted(mem_list.keys())
        values = [mem_list[step] for step in steps]
        plt.plot(steps, values, label=label, marker='o')
        
        # 在 x=0 时显示 y 轴上的值
        if 0 in mem_list:
            y_value = mem_list[0]
            plt.annotate(f'{y_value:.4f}', xy=(0, y_value), xytext=(5, y_value), textcoords='offset points', fontsize=16, color='red')
    
    plt.xlabel('Steps')
    plt.ylabel('Mem. Ratio (%)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig('memorization/mem_ratio_trend.png')

# 定义文件路径和标签
files_and_labels = [
    # ('memorization/mem_list_-20.txt', 'xt -> xt\'(cc = -20)'),
    ('memorization/mem_list_-20_x0hat.txt', 'xt -> x0(cc = -20)'),
    ('memorization/mem_list_0_x0hat.txt', 'xt -> x0(cc = 0)'),
    # ('memorization/mem_list_cur.txt', 'xt(cc = 0)'),
    # ('memorization/mem_list_-1_-5.txt', 'xt -> xt\'(cc = -1, -5)'),
    # ('memorization/mem_list_-5_-20.txt', 'xt -> xt\'(cc = -5, -20)'),
    # 添加更多文件和标签
    # ('memorization/mem_list_another.txt', 'cc = another'),
]

# 读取所有文件中的数据
mem_lists = [read_mem_list(file) for file, _ in files_and_labels]
labels = [label for _, label in files_and_labels]

# 绘制趋势图
plot_mem_trend(mem_lists, labels, 'Mem. Ratio Trend')