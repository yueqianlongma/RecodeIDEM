import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Activation, Reshape, Dropout, Lambda
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import sys
import pandas as pd

origin_path = sys.path
sys.path.append("..")
sys.path = origin_path
from utils.utils import *


class FSLayer(keras.layers.Layer):
    def __init__(self, dims, regularizer_rate, **kwargs):
        self.dims = dims
        self.regularizer_rate = regularizer_rate
        super(FSLayer, self).__init__(**kwargs)

    # 构建wfs参数
    def build(self, input_shape):
        shape = tf.TensorShape([1, self.dims])
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      regularizer=regularizers.l1(self.regularizer_rate),
                                      trainable=True
                                      )

        super(FSLayer, self).build(input_shape)

    # 计算获取h0
    def call(self, inputs):
        return inputs * self.kernel

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    def get_config(self):
        base_config = super(FSLayer, self).get_config()
        base_config['output_dim'] = 1

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def FS_layers(target_dims, dims, regularizer_rate):
    input_model = []
    output_model = []
    for i in range(len(target_dims)):
        input_temp = Input(shape=(target_dims[i],))
        input_model.append(input_temp)

        output_temp = FSLayer(dims, regularizer_rate)(input_temp)
        output_temp = Reshape(target_shape=(target_dims[i],))(output_temp)
        output_model.append(output_temp)

    return input_model, output_model


def Embedding_layers(identify_or_embedding, orgin_dims, target_dims):
    input_model = []
    output_model = []
    for i in range(len(identify_or_embedding)):
        input_temp = Input(shape=(1,))
        if identify_or_embedding[i] == True:
            output_temp = Embedding(orgin_dims[i], target_dims[i])(input_temp)
            output_temp = Reshape(target_shape=(target_dims[i],))(output_temp)
        else:
            output_temp = Dense(1)(input_temp)
        output_model.append(output_temp)
        input_model.append(input_temp)

    return input_model, output_model


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    print(shape1)
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


## DNN
def build_Dense_Model(feature_dims):
    input = Input(shape=(feature_dims,))
    output_model = Dense(1000, kernel_initializer="uniform")(input)
    output_model = Activation('relu')(output_model)
    output_model = Dense(500, kernel_initializer="uniform")(output_model)
    output_model = Activation('relu')(output_model)
    output_model = Dense(2)(output_model)
    output_model = Activation('softmax')(output_model)

    model = KerasModel(inputs=input, outputs=output_model)
    # compile model
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    return model


## FSDNN
def build_Feature_Model(feature_dims):
    input = Input(shape=(feature_dims,))

    output_model = FSLayer(feature_dims, 0.1 / feature_dims)(input)
    output_model = Dense(1000, kernel_initializer="uniform")(output_model)
    output_model = Activation('relu')(output_model)
    output_model = Dense(500, kernel_initializer="uniform")(output_model)
    output_model = Activation('relu')(output_model)
    output_model = Dense(2)(output_model)
    output_model = Activation('sigmoid')(output_model)

    model = Model(inputs=input, outputs=output_model)
    # compile model
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    return model


## EDNN
## Embedding
def Embedding_layers(identify_or_embedding, orgin_dims, target_dims):
    input_model = []
    output_model = []
    for i in range(len(identify_or_embedding)):
        input_temp = Input(shape=(1,))
        if identify_or_embedding[i] == True:
            print(input_temp.shape)
            output_temp = Embedding(orgin_dims[i], target_dims[i])(input_temp)
            print(output_temp.shape)
            output_temp = Reshape(target_shape=(target_dims[i],))(output_temp)
        else:
            output_temp = Dense(1)(input_temp)
        output_model.append(output_temp)
        input_model.append(input_temp)

    return input_model, output_model


def build_Embedding_Model(identify_or_embedding, orgin_dims, target_dims):
    input_model, output_temp = Embedding_layers(identify_or_embedding, orgin_dims, target_dims)

    output_model = Concatenate()(output_temp)
    output_model = Dense(1000, kernel_initializer="uniform")(output_model)
    output_model = Activation('relu')(output_model)
    output_model = Dense(500, kernel_initializer="uniform")(output_model)
    output_model = Activation('relu')(output_model)
    output_model = Dense(2)(output_model)
    output_model = Activation('softmax')(output_model)

    model = Model(inputs=input_model, outputs=output_model)
    # compile model
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    return model


## FSEDNN
def build_FS_Embedding_Model(identify_or_embedding, orgin_dims, target_dims):
    input_model1, output_temp1 = FS_layers(target_dims, dims=1, regularizer_rate=0.001 / 15)

    input_model2, output_temp2 = Embedding_layers(identify_or_embedding, orgin_dims, target_dims)

    input_list = []
    output_list = []
    for i in range(len(input_model1)):
        input_list.append(input_model1[i])

    for i in range(len(input_model2)):
        input_list.append(input_model2[i])

    output_model1 = Concatenate()(output_temp1)
    output_model2 = Concatenate()(output_temp2)
    output_model = keras.layers.Multiply()([output_model1, output_model2])

    output_model = Dense(1000, kernel_initializer="uniform")(output_model)
    output_model = Activation('relu')(output_model)
    output_model = Dense(500, kernel_initializer="uniform")(output_model)
    output_model = Activation('relu')(output_model)
    output_model = Dense(2)(output_model)
    output_model = Activation('sigmoid')(output_model)

    model = Model(inputs=input_list, outputs=output_model)
    # compile model
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    return model


## SNN
def create_siamese_network(input_shape):
    input = Input(shape=input_shape)
    x = input
    x = Dense(256,
              activation='relu',
              name='D1')(x)
    # activity_regularizer = regularizers.l2(0.01))(x)
    x = Dropout(0.1)(x)
    x = Dense(256,
              activation='relu',
              name='D2')(x)
    # activity_regularizer = regularizers.l2(0.01))(x)
    x = Dropout(0.1)(x)
    x = Dense(256,
              activation='relu',
              name='Embeddings')(x)
    # activity_regularizer = regularizers.l2(0.01))(x)
    return Model(input, x)


def build_SNN(feature_dims):
    input_shape = (feature_dims,)
    siamese_network = create_siamese_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = siamese_network(input_a)
    processed_b = siamese_network(input_b)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape,
                      name='Distance')([processed_a, processed_b])

    model_siamese = Model([input_a, input_b], distance)

    rms = RMSprop()

    model_siamese.compile(loss=contrastive_loss,
                          optimizer=rms,
                          metrics=[accuracy])
    return model_siamese


## FSSNN
def create_fs_network(target_dims, reg):
    input_list = []
    output_model = []
    for i in range(len(target_dims)):
        input_temp = Input(shape=(1,))
        input_list.append(input_temp)

        output_temp = FSLayer(1, reg)(input_temp)
        output_temp = Reshape(target_shape=(1,))(output_temp)
        output_model.append(output_temp)

    x = Concatenate()(output_model)
    x = Dense(256,
              activation='relu',
              name='D1')(x)
    # activity_regularizer = regularizers.l2(0.01))(x)
    x = Dropout(0.1)(x)
    x = Dense(256,
              activation='relu',
              name='D2')(x)
    # activity_regularizer = regularizers.l2(0.01))(x)
    x = Dropout(0.1)(x)
    x = Dense(256,
              activation='relu',
              name='Embeddings')(x)
    # activity_regularizer = regularizers.l2(0.01))(x)
    return Model(input_list, x)


def build_FS_Siamese_Model(target_dims, reg):
    base_network = create_fs_network(target_dims, reg)

    inputa_list = []
    for i in range(len(target_dims)):
        input_temp = Input(shape=(1,))
        inputa_list.append(input_temp)

    inputb_list = []
    for i in range(len(target_dims)):
        input_temp = Input(shape=(1,))
        inputb_list.append(input_temp)

    processed_a = base_network(inputa_list)
    processed_b = base_network(inputb_list)

    input_list = []
    for i in range(len(inputa_list)):
        input_list.append(inputa_list[i])
    for i in range(len(inputb_list)):
        input_list.append(inputb_list[i])

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape,
                      name='Distance')([processed_a, processed_b])

    model = Model(input_list, distance)

    return model


## ESNN
def create_embedding_siamese_network(identify_or_embedding, orgin_dims, target_dims):
    input_model, output_temp = Embedding_layers(identify_or_embedding, orgin_dims, target_dims)

    input_list = []
    for i in range(len(input_model)):
        input_list.append(input_model[i])

    x = Concatenate()(output_temp)
    x = Dense(256,
              activation='relu',
              name='D1')(x)
    # activity_regularizer = regularizers.l2(0.01))(x)
    x = Dropout(0.1)(x)
    x = Dense(256,
              activation='relu',
              name='D2')(x)
    # activity_regularizer = regularizers.l2(0.01))(x)
    x = Dropout(0.1)(x)
    x = Dense(256,
              activation='relu',
              name='Embeddings')(x)
    # activity_regularizer = regularizers.l2(0.01))(x)
    return Model(input_list, x)


def build_Embedding_Siamese_Model(identify_or_embedding, orgin_dims, target_dims):
    base_network = create_embedding_siamese_network()

    inputa_list = []
    for i in range(len(target_dims)):
        input_temp = Input(shape=(1,))
        inputa_list.append(input_temp)

    inputb_list = []
    for i in range(len(target_dims)):
        input_temp = Input(shape=(1,))
        inputb_list.append(input_temp)

    processed_a = base_network(inputa_list)
    processed_b = base_network(inputb_list)

    input_list = []
    for i in range(len(inputa_list)):
        input_list.append(inputa_list[i])
    for i in range(len(inputb_list)):
        input_list.append(inputb_list[i])

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape,
                      name='Distance')([processed_a, processed_b])

    model = Model(input_list, distance)

    return model


## FSESNN

def create_fs_embedding_siamese_network(identify_or_embedding, orgin_dims, target_dims, regularizer_rate=0.001 / 15):
    input_model1, output_temp1 = FS_layers(target_dims, dims=1, regularizer_rate=regularizer_rate)

    input_model2, output_temp2 = Embedding_layers(identify_or_embedding, orgin_dims, target_dims)

    input_list = []
    for i in range(len(input_model1)):
        input_list.append(input_model1[i])

    for i in range(len(input_model2)):
        input_list.append(input_model2[i])

    output_model1 = Concatenate()(output_temp1)
    output_model2 = Concatenate()(output_temp2)
    x = keras.layers.Multiply()([output_model1, output_model2])

    x = Dense(256,
              activation='relu',
              name='D1')(x)
    # activity_regularizer = regularizers.l2(0.01))(x)
    x = Dropout(0.1)(x)
    x = Dense(256,
              activation='relu',
              name='D2')(x)
    # activity_regularizer = regularizers.l2(0.01))(x)
    x = Dropout(0.1)(x)
    x = Dense(256,
              activation='relu',
              name='Embeddings')(x)
    # activity_regularizer = regularizers.l2(0.01))(x)
    return Model(input_list, x)


def build_FS_Embedding__Siamese_Model(identify_or_embedding, orgin_dims, target_dims, regularizer_rate=0.001 / 15):
    base_network = create_fs_embedding_siamese_network(identify_or_embedding, orgin_dims, target_dims, regularizer_rate)

    inputa_list = []
    for i in range(len(target_dims)):
        input_temp = Input(shape=(target_dims[i],))
        inputa_list.append(input_temp)
    for i in range(len(target_dims)):
        input_temp = Input(shape=(1,))
        inputa_list.append(input_temp)

    inputb_list = []
    for i in range(len(target_dims)):
        input_temp = Input(shape=(target_dims[i],))
        inputb_list.append(input_temp)
    for i in range(len(target_dims)):
        input_temp = Input(shape=(1,))
        inputb_list.append(input_temp)

    processed_a = base_network(inputa_list)
    processed_b = base_network(inputb_list)

    input_list = []
    for i in range(len(inputa_list)):
        input_list.append(inputa_list[i])
    for i in range(len(inputb_list)):
        input_list.append(inputb_list[i])

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape,
                      name='Distance')([processed_a, processed_b])

    model = Model(input_list, distance)

    return model


## CSDNN
def custom_loss(y_true, y_pred):
    c_FP = 2.0
    c_TP = 1.0
    c_FN = 2.0
    c_TN = 1.0

    cost = y_true * K.log(y_pred) * c_FN + y_true * K.log(1 - y_pred) * c_TP + (1 - y_true) * K.log(1 - y_pred) * c_FP + (1 - y_true) * K.log(
        y_pred) * c_TN

    return - K.mean(cost, axis=-1)


def csdnn(indput_dim):
    input = Input(shape=(indput_dim,))
    output_model = Dense(1000, kernel_initializer="uniform")(input)
    output_model = Activation('relu')(output_model)
    output_model = Dense(500, kernel_initializer="uniform")(output_model)
    output_model = Activation('relu')(output_model)
    output_model = Dense(1)(output_model)
    output_model = Activation('sigmoid')(output_model)

    model = KerasModel(inputs=input, outputs=output_model)
    # compile model
    rms = RMSprop()
    model.compile(loss=custom_loss, optimizer=rms, metrics=['accuracy'])
    return model


## predict
def predict_snn(epoc, feature_testX, testY, feature_trainX, feature_trainY, model, pdf=True, show=False):
    print("----------------------------------------------------------predict_snn-----------------------------------------------------------\n")
    y_pre = []
    for i in feature_testX:
        for t in i:
            pp = []
            for j in feature_trainX:
                for c in j:
                    pp += [[t, c]]

            x_test = np.array(pp)
            y_pred = model.predict([x_test[:, 0], x_test[:, 1]])
            pos = np.argmin(y_pred)

            length = 0
            for j in range(len(feature_trainX)):
                length += len(feature_trainX[j])
                if pos < length:
                    y_pre.append(feature_trainY[j][0])
                    break
    label = np.array(y_pre).reshape([-1, 1])
    print("Accuarcy: ", np.mean(testY == label))
    print(classification_report(testY, label))

    # ROC
    fpr_siamese, tpr_siamese, thresholds_siamese = roc_curve(testY, label)
    auc_siamese = auc(fpr_siamese, tpr_siamese)
    plot_confusion_matrix("SNN_matrix" + str(epoc), confusion_matrix(testY, label), pdf=pdf, show=show)

    return fpr_siamese, tpr_siamese, auc_siamese


def predict_fssnn(epoc, feature_testX, testY, feature_trainX, feature_trainY, model, pdf=True, show=False):
    print("----------------------------------------------------------predict_fssnn-----------------------------------------------------------\n")
    y_pre = []
    for i in feature_testX:
        for t in i:
            pp = []
            for j in feature_trainX:
                for c in j:
                    pp += [[t, c]]

            x_test = np.array(pp)
            X_test_list_0 = split_feature(x_test[:, 0])
            X_test_list_1 = split_feature(x_test[:, 1])

            X_test_list = []
            for i in range(len(X_test_list_0)):
                X_test_list.append(X_test_list_0[i])
            for i in range(len(X_test_list_1)):
                X_test_list.append(X_test_list_1[i])

            y_pred = model.predict(X_test_list)
            pos = np.argmin(y_pred)

            length = 0
            for pos in range(len(feature_trainX)):
                length += len(feature_trainX[pos])
                if pos < length:
                    y_pre.append(feature_trainY[pos][0])
                    break

    label = np.array(y_pre).reshape([-1, 1])
    print("Accuarcy: ", np.mean(testY == label))
    print(classification_report(testY, label))

    # ROC
    fpr_fs_siamese, tpr_fs_siamese, thresholds_fs_siamese = roc_curve(testY, label)
    auc_fs_siamese = auc(fpr_fs_siamese, tpr_fs_siamese)
    plot_confusion_matrix("FSSNN_matrix" + str(epoc), confusion_matrix(testY, label), pdf=pdf, show=show)

    return fpr_fs_siamese, tpr_fs_siamese, auc_fs_siamese


def predict_fsesnn(epoc, feature_testX, testY, feature_trainX, feature_trainY, model, target_dims, pdf=True, show=False):
    print("----------------------------------------------------------predict_fsesnn-----------------------------------------------------------\n")
    y_pre = []
    for i in feature_testX:
        for t in i:
            pp = []
            for j in feature_trainX:
                for c in j:
                    pp += [[t, c]]

            x_test = np.array(pp)
            X_test_list_0 = split_feature(x_test[:, 0])
            X_test_list_1 = split_feature(x_test[:, 1])

            X_test_0 = data_For_FS_Embedding_Net(X_test_list_0, target_dims, True)
            X_test_1 = data_For_FS_Embedding_Net(X_test_list_1, target_dims, True)

            X_test_list = []
            for i in range(len(X_test_0)):
                X_test_list.append(X_test_0[i])
            for i in range(len(X_test_1)):
                X_test_list.append(X_test_1[i])
            y_pred = model.predict(X_test_list)
            pos = np.argmin(y_pred)

            length = 0
            for j in range(len(feature_trainX)):
                length += len(feature_trainX[j])
                if pos < length:
                    y_pre.append(feature_trainY[j][0])
                    break

    label = np.array(y_pre).reshape([-1, 1])
    print("Accuarcy: ", np.mean(testY == label))
    print(classification_report(testY, label))
    # ROC
    fpr_embedding_feature_siamese, tpr_embedding_feature_siamese, thresholds_embedding_feature_siamese = roc_curve(testY, label)
    auc_embedding_feature_siamese = auc(fpr_embedding_feature_siamese, tpr_embedding_feature_siamese)

    plot_confusion_matrix("FSESNN_matrix" + str(epoc), confusion_matrix(testY, label), pdf=pdf, show=show)

    return fpr_embedding_feature_siamese, tpr_embedding_feature_siamese, auc_embedding_feature_siamese


def yb_snn(name, X, model):
    print("--------------------------------------------------------yb_snn-----------------------------------------------------------\n")

    y_pre = []
    for i in range(10):
        pp = []
        for j in range(11):
            pp += [[X[i], X[j]]]
        x_test = np.array(pp)
        y_pred = model.predict([x_test[:, 0], x_test[:, 1]])
        y_pre.append(y_pred.reshape([1, -1]))
    lab = np.array(y_pre).reshape([len(y_pre), -1])
    df = pd.DataFrame(lab)
    print(df.shape)
    df.to_csv(name + ".csv", index=False)


def yb_fssnn(name, X, model):
    print("------------------------------------------------------yb_fssnn-----------------------------------------------------------\n")
    y_pre = []
    for i in range(10):
        pp = []
        for j in range(11):
            pp += [[X[i], X[j]]]
        x_test = np.array(pp)

        X_test_list_0 = split_feature(x_test[:, 0])
        X_test_list_1 = split_feature(x_test[:, 1])

        X_test_list = []
        for i in range(len(X_test_list_0)):
            X_test_list.append(X_test_list_0[i])
        for i in range(len(X_test_list_1)):
            X_test_list.append(X_test_list_1[i])

        y_pred = model.predict(X_test_list)
        y_pre.append(y_pred.reshape([1, -1]))

    lab = np.array(y_pre).reshape([len(y_pre), -1])
    df = pd.DataFrame(lab)
    print(df.shape)
    df.to_csv(name + ".csv", index=False)


def yb_fsesnn(name, X, target_dims, model):
    print("----------------------------------------------------------yb_fsesnn-----------------------------------------------------------\n")
    y_pre = []
    for i in range(len(X)):
        pp = []
        for j in range(len(X)):
            pp += [[X[i], X[j]]]
        x_test = np.array(pp)
        X_test_list_0 = split_feature(x_test[:, 0])
        X_test_list_1 = split_feature(x_test[:, 1])

        X_test_0 = data_For_FS_Embedding_Net(X_test_list_0, target_dims, True)
        X_test_1 = data_For_FS_Embedding_Net(X_test_list_1, target_dims, True)

        X_test_list = []
        for i in range(len(X_test_0)):
            X_test_list.append(X_test_0[i])
        for i in range(len(X_test_1)):
            X_test_list.append(X_test_1[i])
        y_pred = model.predict(X_test_list)
        y_pre.append(y_pred.reshape([1, -1]))

    lab = np.array(y_pre).reshape([len(y_pre), -1])
    df = pd.DataFrame(lab)
    print(df.shape)
    df.to_csv(name + ".csv", index=False)
