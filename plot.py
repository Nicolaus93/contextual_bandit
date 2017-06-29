import matplotlib.pyplot as plt
import pickle
import os
from os.path import isfile, join
import argparse


def main():
    parser = argparse.ArgumentParser(description='Multi Armed Bandit algorithms.')
    parser.add_argument(dest='dataset', metavar='dataset', type=str, nargs=1,
                        help='the dataset to use')
    parser.add_argument(dest='regret', metavar='regret', type=str, nargs=1,
                        help='choose the type of regret to plot')
    args = parser.parse_args()

    # loading results
    dataset = args.dataset[0]
    regret = args.regret[0]
    if not regret in ['cum_regret', 'regret']:
        print('please choose cum_regret or regret')
        exit()
    file_path = join(os.sep, os.getcwd(), 'datasets/avazu/' + dataset + '/results/' + regret)
    files = os.listdir(file_path)
    col = ['b', 'g', 'r', 'y', 'm']
    for i, bandit in enumerate(files):
        print(join(file_path, bandit))
        fileObject = open(join(file_path, bandit),'rb')
        try:
            cum_regret = pickle.load(fileObject)
        except:
            continue
        time = len(cum_regret)
        plt.plot(range(time), cum_regret, c=col[i], ls='-', label=str(bandit))
        plt.xlabel('time')
        plt.ylabel('regret')
        axes = plt.gca()
        plt.title("Regret Bound with respect to T")

    p = plt.legend(loc='upper left')
    p.draggable(True)
    plt.show()

if __name__ == '__main__':
    main()