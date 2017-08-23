import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def create_model(classes, d):
    file_path = os.getcwd()
    model = np.random.random((classes, d))
    directory = os.path.join(file_path, "models")
    if not os.path.exists(directory):
        os.makedirs(directory)
    print("saving..")
    name = str(classes) + "classes_" + str(d) + "d"
    fileObject = open(os.path.join(directory, name), 'wb')
    pickle.dump(model, fileObject)
    fileObject.close()
    return model


if __name__ == "__main__":
    classes = int(sys.argv[1])
    d = int(sys.argv[2])
    model = create_model(classes=classes, d=d)
    sns.set()
    col = sns.color_palette()
    plt.figure(1)
    i, j = model.shape
    first = int(str(i) + "11")  # plot x11
    ax1 = plt.subplot(first)
    plt.setp(ax1.get_xticklabels(), visible=False)
    x = range(d)
    plt.bar(x, height=model[0])
    for k in range(2, classes + 1):
        num = int(str(i) + '1' + str(k))
        plt.subplot(num, sharex=ax1)
        plt.bar(x, height=model[k - 1], color=col[k - 1])
        ax2 = plt.subplot(num, sharex=ax1)
        plt.setp(ax2.get_xticklabels(), visible=False)
    plt.show()
