import re
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


pattern_title = re.compile(r"(---------------------Start evaluation on MS-COCO dataset test-----------------------)")
# pattern_title = re.compile(r"(-----------Evaluation performance on MS-COCO validation dataset for Epoch \d*----------)")
pattern_cider = re.compile(r"(CIDEr: )(\d*\.\d*)")
file_root = '../logs_1'



def find_pattern(pattern, string):
    results = re.match(pattern, string)
    if results is not None:
        return results.group(0)
    return False


def find_ciders(fp):
    cider_list = []
    with open(fp, 'r') as f:
        content = f.readlines()
        flag = 0
        for cont in content:
            if flag == 1:
                status = find_pattern(pattern_cider, cont)
                if status is not False:
                    flag = 0
                    cider_list.append(float(status.split(': ')[-1]))
            elif find_pattern(pattern_title, cont):
                flag = 1
    maxcider = 0.0
    for i in range(len(cider_list)):
        if cider_list[i] > maxcider:
            maxcider = cider_list[i]
        else:
            cider_list[i] = maxcider
    # print(cider_list)
    return [i + 1 for i in range(len(cider_list))], cider_list


if __name__ == '__main__':
    datasets = ['ucm', 'sydney']
    for dataset in datasets:
        log_list = ['log_%s_pure.txt' % dataset, 'log_%s_sim_2fc.txt' % dataset]
        x_list = []
        y_list = []
        for file_path in log_list:
            file_path = os.path.join(file_root, file_path)
            x, y = find_ciders(file_path)
            x_list.append(x)
            y_list.append(y)

        # pp = PdfPages('../figs/%s.pdf'% dataset)

        plt.figure()

        if dataset == 'sydney':
            plt.ylim((1.2, 2.8))  # Sydney
        else:
            plt.ylim((2.5, 3.7))  # UCM

        plt.xlim((0, 80))
        # plt.xlim((0, max(max(x) for x in x_list)))
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel('CIDEr', fontsize=15)
        linestyles = ['-.', '-', ':', '--']
        colors = ['r', 'b']

        labels = ['Attention', 'Visual Aligning Attention']
        for num, task in enumerate(log_list):
            plt.plot(x_list[num], y_list[num], color=colors[num], linewidth=2.0, linestyle=linestyles[num], label=labels[num])
        plt.legend(loc='lower right')
        plt.savefig('../figs/%s.jpg' % dataset)
        # pp.savefig()
        # pp.close()