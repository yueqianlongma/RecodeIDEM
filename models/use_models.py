from tensorflow.keras.optimizers import RMSprop
import random
import sys

origin_path = sys.path
sys.path.append("..")
sys.path = origin_path
from models.models import *
from utils.utils import *
from data_deal.deal_data import deal_data, overSampling


class UseModels(object):
    def __init__(self, sim_size, filePath, conditionPath=None, train_prob=0.002, test_prob=0.003,
                 test_len=0.25, epochs=100, batch_size=1024, pdf=True, show=True):

        self.X, self.identify_or_embedding, self.orgin_dims, self.target_dims, self.feature_names, \
        self.trainX, self.trainY, self.testX, self.testY, self.featureX, self.featureY, \
        self.feature_trainX, self.feature_testX, self.feature_trainY, self.feature_testY = \
            deal_data(filePath, conditionPath, train_prob, test_prob, test_len)

        print(len(self.feature_trainX))
        print(len(self.feature_trainX))
        self.sim_size = int(sim_size / (len(self.feature_trainX) * (len(self.feature_trainX) + 1) / 2))

        self.epochs = epochs
        self.batch_size = batch_size
        self.pdf = pdf
        self.show = show
        self.roc_values = []

    def dataFor(self):
        print("----------------------------------------------------------dataFor-----------------------------------------------------------\n")
        ####  train
        pp = []
        label = []

        for i in range(len(self.feature_trainX)):
            for j in range(i, len(self.feature_trainX)):
                hdix_train_len = int(len(self.feature_trainX[i])) - 1
                hdjx_train_len = int(len(self.feature_trainX[j])) - 1
                if i == j:
                    for c in range(self.sim_size):
                        t1 = random.randint(0, int(hdix_train_len / 2))
                        t2 = random.randint(int(hdix_train_len / 2), hdix_train_len)
                        pp += [[self.feature_trainX[i][t2], self.feature_trainX[j][t1]]]
                        label.append(1)
                else:
                    for c in range(self.sim_size):
                        t1 = random.randint(0, hdix_train_len)
                        t2 = random.randint(0, hdjx_train_len)
                        pp += [[self.feature_trainX[i][t1], self.feature_trainX[j][t2]]]
                        label.append(0)
        print(len(pp))
        X_train = np.array(pp)
        print(X_train.shape)
        Y_train = np.array(label).reshape([-1, 1])
        print(Y_train.shape)

        return X_train, Y_train

    def _CSDNN(self, epoc):
        print("----------------------------------------------------------CSDNN-----------------------------------------------------------\n")

        model = csdnn(self.trainX.shape[1])
        model.summary()

        history = model.fit(self.trainX, self.trainY, epochs=self.epochs,
                            batch_size=self.batch_size, verbose=1)

        plotImage("csdnn_LS_AC" + str(epoc), history.history['acc'], history.history['loss'],
                  'CSDNN Training Loss and Accuracy', pdf=self.pdf, show=self.show)

        y_pred = model.predict(self.testX)
        y_pred = y_pred > 0.5

        print(classification_report(self.testY, y_pred))
        ##ROC
        fpr, tpr, thresholds_csdnn = roc_curve(self.testY, y_pred)
        auc_csdnn = auc(fpr, tpr)
        plot_confusion_matrix("CSDNN_matrix" + str(epoc), confusion_matrix(self.testY, y_pred), pdf=self.pdf, show=self.show)

        self.roc_values.append(RocPack(fpr, tpr, auc_csdnn, "CSDNN"))

    def _SNN(self, epoc):
        print("---------------------SNN------------------------\n")

        model_siamese = build_SNN(self.trainX.shape[1])
        model_siamese.summary()

        history = model_siamese.fit([self.X_train[:, 0], self.X_train[:, 1]], self.Y_train,
                                    batch_size=self.batch_size,  ##512
                                    epochs=self.epochs,  ##100
                                    validation_split=0.25,
                                    verbose=1)
        plotImage("SNN_LS_AC" + str(epoc), history.history['accuracy'], history.history['loss'],
                  'SNN Training Loss and Accuracy', pdf=self.pdf, show=self.show)

        fpr, tpr, auc = predict_snn(epoc, self.feature_testX, self.testY,
                                    self.feature_trainX, self.feature_trainY, model_siamese, pdf=self.pdf, show=self.show)
        self.roc_values.append(RocPack(fpr, tpr, auc, "SNN"))

        return model_siamese

    def _FSSNN(self, epoc):
        print("--------------------FSSNN-----------------------\n")

        X_train_list = []
        for i in range(len(self.X_train_list_0)):
            X_train_list.append(self.X_train_list_0[i])
        for i in range(len(self.X_train_list_1)):
            X_train_list.append(self.X_train_list_1[i])
        print(len(X_train_list))

        model = build_FS_Siamese_Model(self.target_dims, 0.001 / 7)
        rms = RMSprop()
        model.compile(loss=contrastive_loss,
                      optimizer=rms,
                      metrics=[accuracy])
        model.summary()

        history = model.fit(X_train_list, self.Y_train,
                            batch_size=self.batch_size,  ##512
                            epochs=self.epochs,  ## 300
                            validation_split=0.25,
                            verbose=1)
        plotImage("FSSNN_LS_AC" + str(epoc), history.history['accuracy'], history.history['loss'],
                  'FSSNN Training Loss and Accuracy', pdf=self.pdf, show=self.show)

        ## show features weights
        weights = model.get_weights()[:len(self.feature_names)]
        plot_feature_image('FSSNN_features_weights' + str(epoc), weights, self.feature_names, self.pdf)

        fpr, tpr, auc = predict_fssnn(epoc, self.feature_testX, self.testY,
                                      self.feature_trainX, self.feature_trainY, model, pdf=self.pdf, show=self.show)
        self.roc_values.append(RocPack(fpr, tpr, auc, "FSSNN"))

        return model

    def _FSESNN(self, epoc):
        print("--------------------FSESNN----------------------\n")

        X_train_0 = data_For_FS_Embedding_Net(self.X_train_list_0, self.target_dims, True)
        X_train_1 = data_For_FS_Embedding_Net(self.X_train_list_1, self.target_dims, True)

        X_train_list = []
        for i in range(len(X_train_0)):
            X_train_list.append(X_train_0[i])
        for i in range(len(X_train_1)):
            X_train_list.append(X_train_1[i])

        model_fs_embedding_siamese = build_FS_Embedding__Siamese_Model(self.identify_or_embedding,
                                                                       self.orgin_dims,
                                                                       self.target_dims, 0.001 / 7)
        rms = RMSprop()
        model_fs_embedding_siamese.compile(loss=contrastive_loss,
                                           optimizer=rms,
                                           metrics=[accuracy])
        model_fs_embedding_siamese.summary()

        history = model_fs_embedding_siamese.fit(X_train_list, self.Y_train,
                                                 batch_size=self.batch_size,  ##512
                                                 epochs=self.epochs,  ##400
                                                 validation_split=0.25,
                                                 verbose=1)
        plotImage("FSESNN_LS_AC" + str(epoc), history.history['accuracy'], history.history['loss'],
                  'FSESNN Training Loss and Accuracy', pdf=self.pdf, show=self.show)

        ## show features weights
        weights = model_fs_embedding_siamese.get_weights()[:len(self.feature_names)]
        plot_feature_image('FSESNN' + str(epoc), weights, self.feature_names, pdf=self.pdf, show=self.show)

        fpr, tpr, auc = predict_fsesnn(epoc, self.feature_testX, self.testY,
                                       self.feature_trainX, self.feature_trainY, model_fs_embedding_siamese,
                                       self.target_dims, pdf=self.pdf, show=self.show)

        self.roc_values.append(RocPack(fpr, tpr, auc, "FSESNN"))

        return model_fs_embedding_siamese

    def showSingleImage(self, args, epoc):

        plt.figure(figsize=(10, 10))
        for rocPack in self.roc_values:
            plt.plot(rocPack.fpr, rocPack.tpr,
                     label=rocPack.name + '(area = {:.3f})'.format(rocPack.auc))
        plt.legend(loc='best')
        plt.title("ROC")
        if self.pdf != True:
            plt.savefig("ROC" + epoc + '.pdf')
        if self.show == True:
            plt.show()

    def showMeanImage(self, all_Roc):
        plt.figure(figsize=(10, 10))
        for name, rocs in all_Roc.items():
            mean = np.mean(rocs, axis=1)
            std = np.std(rocs, axis=1)

    def start(self, epoc=1):

        self.X_train, self.Y_train = self.dataFor()
        self.X_train_list_0 = split_feature(self.X_train[:, 0])
        self.X_train_list_1 = split_feature(self.X_train[:, 1])

        model = self._FSESNN(epoc)
        yb_fsesnn("FSESNN_" + str(epoc), self.X, self.target_dims, model)

        if args.CSDNN == True:
            self._CSDNN(epoc)

        if args.SNN == True:
            model = self._SNN(epoc)
            yb_snn("SNN_" + str(epoc), self.X, model)

        if args.FSSNN == True:
            model = self._FSSNN(epoc)
            yb_fssnn("FSSNN_" + str(epoc), self.X, model)


        if args.FSESNN == True:
            model = self._FSESNN(epoc)
            yb_fsesnn("FSESNN_" + str(epoc), self.X ,self.target_dims, model)

