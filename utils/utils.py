import numpy as np
import matplotlib.pyplot as plt
import itertools

plt.switch_backend('agg')


class RocPack(object):
    def __init__(self, fpr, tpr, auc, name):
        self.fpr = fpr
        self.tpr = tpr
        self.auc = auc
        self.name = name

    def plotSigleRoc(self):
        plt.plot(self.fpr, self.tpr, label=self.name+' (area = {:.3f})'.format(self.auc))
        plt.legend(loc='best')
        plt.show()



def split_feature(X):
    x_list = []
    for i in range(X.shape[1]):
        x_list.append(X[:, [i]])
    return x_list


def data_For_FS_Embedding_Net(X_train, target_dims, bk=False):
    x_list = []
    for i in range(len(target_dims)):
        temp_list = []
        for j in range(target_dims[i]):
            temp_list.append(X_train[i])
        x_list.append(np.concatenate(temp_list, axis=1))
    if bk == True:
        for i in range(len(target_dims)):
            x_list.append(X_train[i])

    return x_list


def plotImage(name, acc, loss, labels, pdf=True, show=False):
    epochs = range(len(acc))
    plt.figure()
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, loss, 'r', label='Traing loss')
    plt.title(labels)
    plt.legend()
    if pdf == True:
        plt.savefig('./img/' + name + '.pdf')
    if show == True:
        plt.show()
    plt.close()
    plt.cla()


def plot_confusion_matrix(name, cm, pdf=True, show=False, classes=['0', '1'],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(name + "_" + title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if pdf == True:
        plt.savefig('./img/'+ name + '.pdf')
    if show == True:
        plt.show()
    plt.close()
    plt.cla()

def plot_feature_image(name, weights, feature_names, pdf=True, show=False):
    pos = np.argsort(-np.abs(weights).reshape([-1, ]))
    x = [float(abs(weights[i])) for i in pos]
    print(len(x))
    print(type(x[0]))
    y = [c for c in range(len(feature_names))]
    tick_label = [feature_names[i] for i in pos]
    #     tick_label.reverse()
    print(tick_label)
    bars = plt.barh(y, x, tick_label=tick_label)
    for bar in bars:
        bar.set_facecolor('cornflowerblue')
    plt.plot(x, y)
    plt.scatter(x, y, c='r', marker='o')
    plt.title(name)
    if pdf == True:
        plt.savefig('./img/' + name + '.pdf')
    if show == True:
        plt.show()
    plt.close()
    plt.cla()


def plot_weight_image(name, weights, pdf=True, show=False):
    x = [c for c in range(42)]
    y = weights

    plt.scatter(x[:2], y[:2], c='r', marker='o')
    plt.scatter(x[2:-1], y[2:-1], c='b', marker='o')
    plt.title(name)
    if pdf == True:
        plt.savefig('./img/'+ name + '.pdf')
    if show == True:
        plt.show()
    plt.close()
    plt.cla()
