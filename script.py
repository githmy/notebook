# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# %run script.py

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import jieba
import glob
import copy
# Input data files are available in the "./data/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "./data"]).decode("utf8"))

jieba_userdicts = glob.glob("./jieba/*.txt")
for jieba_userdict in jieba_userdicts:
    jieba.load_userdict(jieba_userdict)

# Any results you write to the current directory are saved as output.

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

max_features = 20000
maxlen = 100

# 文件名字
# 训练数据
train_file = "./data/rasa_train.csv"
# 测试数据
test_file = "./data/rasa_test.csv"
# 结果文件
res_file = "./data/rasa_baseline.csv"
# 模型文件
model_path = "./model/rasa_weights_base.best.hdf5"

# 使用原始文件 分出验证集
train_all = pd.read_csv(train_file)
train_all = train_all.sample(frac=1).reset_index(drop=True)
lenth_train = train_all.shape[0]
spint = int(0.8 * lenth_train)
train = train_all.loc[0:spint, :]
test = train_all.loc[spint:, :]
test.to_csv(test_file, index=False)
# test = pd.read_csv(test_file)

for i1 in train.index:
    train.loc[i1, "comment_text"] = " ".join(jieba.cut(train.loc[i1, "comment_text"]))
for i1 in test.index:
    test.loc[i1, "comment_text"] = " ".join(jieba.cut(test.loc[i1, "comment_text"]))

list_sentences_train = train["comment_text"].fillna("CVxTz").values
list_classes = [i1 for i1 in train.columns]
try:
    list_classes.remove("comment_text")
    list_classes.remove("id")
except Exception as e:
    pass
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("CVxTz").values

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)


def get_model():
    embed_size = 128
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(30, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = get_model()
batch_size = 32
epochs = 100

checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorb = TensorBoard(log_dir='./logs', histogram_freq=10, write_graph=True, write_images=True, embeddings_freq=0,
                      embeddings_layer_names=None, embeddings_metadata=None)
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
callbacks_list = [checkpoint, early, tensorb]  # early

model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=callbacks_list)

model.load_weights(model_path)

y_test = model.predict(X_te)

sample_submission = pd.read_csv(test_file)

sample_submission[list_classes] = y_test
sample_submission["max"] = sample_submission[list_classes].max(axis=1)

for indexs in sample_submission.index:
    for i2 in list_classes:
        if (sample_submission.loc[indexs, i2] == sample_submission.loc[indexs, "max"]):
            sample_submission.loc[indexs, "predict"] = i2
for i1 in list_classes:
    sample_submission.rename(columns={i1: "pred_" + i1}, inplace=True)
sample_submission.to_csv(res_file, index=False)

# 正确率评估
score = model.evaluate(X_t, y, batch_size=batch_size)
print(score)
test_pd = pd.read_csv(test_file)
res_pd = pd.read_csv(res_file)
total_pd = pd.concat([res_pd, test_pd], join='outer', axis=1)
total_right = 0
total_num = 0
for i1 in list_classes:
    tmp_obj = total_pd[total_pd[i1] == 1]
    sum_num = tmp_obj.shape[0]
    right_num = tmp_obj[tmp_obj["predict"] == i1].shape[0]
    total_right += right_num
    total_num += sum_num
    try:
        print("%s, sum_num: %d, right_num: %d, accuracy: %.3f" % (i1, sum_num, right_num, right_num / sum_num))
    except Exception as e:
        print("%s, sum_num: %d, right_num: %d, error: %s" % (i1, sum_num, right_num, str(e)))

print("total data, total_num: %d, total_right: %d, accuracy: %.3f" % (total_num, total_right, total_right / total_num))
