from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import parser


def get_event(event_path):
    event_acc = EventAccumulator(event_path)
    event_acc.Reload()
    # print(event_acc.Tags()['scalars'])
    # print(event_acc.Scalars('metrics'))

    w_times, step_nums, vals = zip(*event_acc.Scalars('metrics'))
    print(step_nums, vals)
    # print(len(step_nums))

    x = list()
    y = list()
    # for i in range(0, 50, 1):
    for i in range(0, len(step_nums), 1):
        # x.append(i)
        x.append(step_nums[i])
        y.append(vals[i])
    # json_dict = {
    #     'x': x,
    #     'y': y
    # }
    # with open('figs/fig_loss.json', 'w') as f:
    #     json.dump(json_dict, f, indent=4)
    # plt.figure()
    # plt.plot(x, y)
    # plt.savefig('figs/fig.jpg')
    return x, y

def get_events(tasks, session, metric, event_pattern):
    x_list = []
    y_list = []
    for task in tasks:
        event_path = event_pattern.format(task, session, metric)
        print('event_path: ', event_path)
        x,y = get_event(event_path)
        x_list.append(x)
        y_list.append(y)
    plt.figure()

    plt.ylim((0.0, 3.5))
    plt.xlim((0, 52))
    # plt.xlim((0, max(max(x) for x in x_list)))
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('loss', fontsize=15)
    linestyles = ['-.', '-', ':', '--']
    colors = ['r', 'b']
    for num, task in enumerate(tasks):
        plt.plot(x_list[num], y_list[num], color=colors[num], linewidth=2.0, linestyle=linestyles[num], label=task)
    plt.legend(loc='upper left')
    plt.savefig('figs/ucm.jpg')


if __name__ == '__main__':
    if not os.path.exists('figs'):
        os.mkdir('figs')

    tasks = ['log_ucm_2fc_cos', 'log_ucm_non']
    # tasks = ['log_sydney_2fc_cos', 'log_sydney_non']
    session = '1'
    metric = 'CIDEr'
    # event_id = '1556223156'
    # event_path = '{}/{}/metrics/{}/events.out.tfevents.{}.03fcbf7fd2c8'.format(
    #     tasks, session, metric, event_id)
    # get_event(event_path)
    event_pattern = '{}/{}/metrics/{}'
    get_events(tasks, session, metric, event_pattern)
