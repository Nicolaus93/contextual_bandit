import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import pickle
import os
from os.path import join
import argparse
import itertools


def main():
    parser = argparse.ArgumentParser(
        description='Multi Armed Bandit algorithms.')
    parser.add_argument(dest='dataset', metavar='dataset', type=str, nargs=1,
                        help='the dataset to use')
    parser.add_argument('-cum_regret', dest='cum_regret', action='store_true',
                        default=False,
                        help='whether to plot cumulative regret')

    args = parser.parse_args()

    # loading results
    dataset = args.dataset[0]
    cum_regret = args.cum_regret
    if cum_regret:
        regret = 'cum_regret'
    else:
        regret = 'regret'
    file_path = join(
        os.getcwd(), 'datasets/avazu/' + dataset + '/results/test/' + regret)
    files = [f for f in os.listdir(file_path) if f.endswith(".plot")]
    col = itertools.cycle(seaborn.color_palette())
    for i, bandit in enumerate(files):
        print(join(file_path, bandit))
        fileObject = open(join(file_path, bandit), 'rb')
        try:
            cum_regret = pickle.load(fileObject)
        except Exception as e:
            print(e)
            continue
        name = bandit.split('.')[0]
        if name in ['ExploitSingle', 'ExploitMulti']:
            print(name)
            continue
        time = len(cum_regret)
        plt.figure(1, figsize=(12, 6))
        ax = plt.subplot(111)
        ax.plot(range(time), cum_regret, c=next(col),
                ls='-', label=name, linewidth=1)

    plt.xlabel('time')
    plt.ylabel('regret')

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    p = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                  fancybox=True, shadow=True, frameon=True)
    p.draggable(True)

    plt.show()


if __name__ == '__main__':
    main()
