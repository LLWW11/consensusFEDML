"""绘制原始城域拓扑三种方案的平均参与人数和总时间。"""
from pathlib import Path
import csv
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    """读取原始两百轮汇总，校验并导出图像与数据。"""
    base = Path(__file__).resolve().parents[2]
    source = base / 'result-U-6fixedge_epoch200.mat'
    dest = Path(__file__).resolve().parent
    data = loadmat(str(source))
    util = data['total_util'].ravel()
    methods = [('HFLSnF', 'HFLSnF_fix', '#1c6795', 'o', '-'),
               ('HFLnoSnF', 'HFLnoSnF_fix', '#d0752c', 's', '--'),
               ('FLnoSnF', 'FLnoSnF', '#8d4f85', '^', '-.')]
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei'],
                         'axes.unicode_minus': False, 'font.size': 11,
                         'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 5), layout='constrained')
    rows = []
    for name, key, color, marker, line in methods:
        clients = data['cnum_' + key].ravel()
        total = data['toall_time_' + key].ravel()
        # 使用原文件汇总字段，同时独立核对其与逐轮数据的关系。
        assert np.allclose(clients, data['client_num_' + key].mean(axis=0))
        expected_time = data['max_layer_' + key].mean(axis=0) * 16 + 100 + data['time_agg_' + key].mean(axis=0)
        assert np.allclose(total, expected_time)
        axes[0].plot(util, clients, label=name, color=color, marker=marker, linestyle=line, linewidth=2, markersize=6)
        axes[1].plot(util, total, label=name, color=color, marker=marker, linestyle=line, linewidth=2, markersize=6)
        rows.extend((float(u), name, float(c), float(t)) for u,c,t in zip(util,clients,total))
    for axis, title, ylabel in zip(axes, ['（a）平均参与人数', '（b）平均总时间'], ['参与人数（个）', '总时间（秒）']):
        axis.set_title(title, loc='left', fontsize=13, fontweight='bold')
        axis.set_xlabel('网络利用率 U')
        axis.set_ylabel(ylabel)
        axis.set_xticks(util)
        axis.grid(axis='y', color='#d9e1e5', linewidth=0.8)
        axis.spines[['top','right']].set_visible(False)
        axis.legend(frameon=False, fontsize=10)
    axes[0].set_ylim(0, 40)
    axes[1].set_ylim(100, 220)
    fig.suptitle('原始 Metro 拓扑：三种方案随网络利用率的变化', fontsize=16, fontweight='bold')
    fig.text(0.5, -0.02, '每个 U 均为原始 200 轮统计；分层方案采用固定边缘节点。', ha='center', fontsize=10, color='#555555')
    fig.savefig(dest / '原始Metro_参与人数与总时间.png', dpi=220, bbox_inches='tight')
    fig.savefig(dest / '原始Metro_参与人数与总时间.pdf', bbox_inches='tight')
    plt.close(fig)
    with (dest / '原始Metro_参与人数与总时间.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['网络利用率U', '方案', '平均参与人数', '平均总时间_秒'])
        writer.writerows(rows)
    print('已导出图像和数据，共 {} 条记录'.format(len(rows)))


if __name__ == '__main__':
    main()
