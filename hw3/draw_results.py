import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import numpy as np

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    # 创建一个字典来存储每个step的指标
    tag_dict = {}
    for e in tf.train.summary_iterator(file):
        step = e.step
        if step not in tag_dict:
            tag_dict[step] = {}
        for v in e.summary.value:
            tag_dict[step][v.tag] = v.simple_value
        if len(tag_dict) > 120:
            break
    # 从字典中提取同时包含 'Train_EnvstepsSoFar' 和 'Train_AverageReturn' 的步数
    X = []
    Y = []
    for step in sorted(tag_dict.keys()):
        if 'Train_EnvstepsSoFar' in tag_dict[step] and 'Train_AverageReturn' in tag_dict[step]:
            X.append(tag_dict[step]['Train_EnvstepsSoFar'])
            Y.append(tag_dict[step]['Train_AverageReturn'])
    return X, Y

if __name__ == '__main__':
    dqn_dirs = [
        'data/q1_dqn_1_LunarLander-v3_30-10-2024_22-30-02',
        'data/q1_dqn_2_LunarLander-v3_30-10-2024_22-37-31',
        'data/q1_dqn_3_LunarLander-v3_30-10-2024_22-44-40'
    ]

    ddqn_dirs = [
        'data/q1_doubledqn_1_LunarLander-v3_31-10-2024_00-08-51',
        'data/q1_doubledqn_2_LunarLander-v3_31-10-2024_00-18-01',
        'data/q1_doubledqn_3_LunarLander-v3_31-10-2024_00-26-58'
    ]

    dqn_runs = []
    ddqn_runs = []

    # 提取DQN的数据
    for dqn_dir in dqn_dirs:
        logdir = os.path.join(dqn_dir, 'events*')
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results(eventfile)
        dqn_runs.append((X, Y))

    # 提取DDQN的数据
    for ddqn_dir in ddqn_dirs:
        logdir = os.path.join(ddqn_dir, 'events*')
        eventfile = glob.glob(logdir)[0]
        X, Y = get_section_results(eventfile)
        ddqn_runs.append((X, Y))

    # 定义函数来插值并平均结果
    def average_runs(runs):
        # 找到所有运行的公共X轴范围
        min_x = max(run[0][0] for run in runs if len(run[0]) > 0)
        max_x = min(run[0][-1] for run in runs if len(run[0]) > 0)
        X_common = np.linspace(min_x, max_x, num=100)
        Ys = []
        for X, Y in runs:
            # 确保X和Y的长度一致
            if len(X) != len(Y):
                continue
            # 对每个Y值进行插值，使其对应公共的X轴
            Y_interp = np.interp(X_common, X, Y)
            Ys.append(Y_interp)
        if len(Ys) == 0:
            return X_common, np.zeros_like(X_common), np.zeros_like(X_common)
        Y_mean = np.mean(Ys, axis=0)
        Y_std = np.std(Ys, axis=0)
        return X_common, Y_mean, Y_std

    # 计算DQN和DDQN的平均值和标准差
    dqn_X_avg, dqn_Y_mean, dqn_Y_std = average_runs(dqn_runs)
    ddqn_X_avg, ddqn_Y_mean, ddqn_Y_std = average_runs(ddqn_runs)

    # 绘制6条学习曲线和平均曲线
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'r']
    linestyles = ['-', '--', '-.']

    # 绘制DQN的单个种子曲线
    for i, (X, Y) in enumerate(dqn_runs):
        plt.plot(X, Y, color=colors[i % len(colors)], linestyle='-', alpha=0.3, label=f'DQN Seed {i+1}')

    # 绘制DDQN的单个种子曲线
    for i, (X, Y) in enumerate(ddqn_runs):
        plt.plot(X, Y, color=colors[i % len(colors)], linestyle='--', alpha=0.3, label=f'DDQN Seed {i+1}')

    # 绘制DQN的平均曲线和标准差，加粗线条
    plt.plot(dqn_X_avg, dqn_Y_mean, color='blue', linestyle='-', linewidth=2.5, label='DQN Mean')
    plt.fill_between(dqn_X_avg, dqn_Y_mean - dqn_Y_std, dqn_Y_mean + dqn_Y_std, color='blue', alpha=0.2)

    # 绘制DDQN的平均曲线和标准差，加粗线条
    plt.plot(ddqn_X_avg, ddqn_Y_mean, color='orange', linestyle='--', linewidth=2.5, label='DDQN Mean')
    plt.fill_between(ddqn_X_avg, ddqn_Y_mean - ddqn_Y_std, ddqn_Y_mean + ddqn_Y_std, color='orange', alpha=0.2)

    plt.xlabel('Train Steps')
    plt.ylabel('Train Average Return')
    plt.title('Learning Curves and Average Return of DQN and DDQN')
    plt.legend()
    plt.grid(True)
    plt.show()
