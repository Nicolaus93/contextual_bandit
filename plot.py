import matplotlib.pyplot as plt
import pickle
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Multi Armed Bandit algorithms.')
    parser.add_argument(dest='dataset', metavar='dataset', type=str, nargs=1,
                        help='the dataset to use')
    args = parser.parse_args()

    # loading dataset
    dataset = args.dataset[0]
    file_path = os.path.join(os.sep, os.getcwd(), 'datasets/avazu/' + dataset + '/results/cum_regret')
    # file_Name = 'cum_regret'
    fileObject = open(file_path,'rb')
    cum_regret = pickle.load(fileObject)
    col = ['b', 'g', 'r', 'y']
    for i, bandit in enumerate(cum_regret):
        time = len(cum_regret[bandit])
        plt.plot(range(time), cum_regret[bandit], c=col[i], ls='-', label=bandit)
        plt.xlabel('time')
        plt.ylabel('regret')
        plt.legend(loc='upper left')
        axes = plt.gca()
        plt.title("Regret Bound with respect to T")

    plt.show()

if __name__ == '__main__':
    main()