
# coding: utf-8

# ### 1. 模块加载
# 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import jieba
import glob
import copy
import time

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard

# 避免小数据对显存的浪费
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


# ### 2. 目录定义查看
# 

# In[2]:


from subprocess import check_output
path = '../nocode/'
inp = 'input/testproject/'
oup = 'output/testproject/'
logp = 'logs/testproject/'
modp = 'models/testproject/'
    
# print(check_output(["ls", path + inp]).decode("utf8"))

EMBEDDING_FILE = path + 'wordvector/wiki.zh.vec'

# 训练数据总集
data_all_file = path + inp + 'sematic_label_train.csv'

# 训练数据集
train_file = path + inp + 'train.csv'
# 测试数据集
test_file = path + inp + 'test.csv'

TXT_DATA_FILE = path + inp + 'sematic_train.txt'
XLSX_DATA_FILE = path + inp + 'sematic_train.xlsx'
CSV_DATA_FILE = path + inp + 'sematic_train.csv'

# 结果文件
res_file = path + inp + 'baseline.csv'
# 模型文件
model_path = path + modp + 'model_t.h5'
out_path = path + oup + 'baseline.csv'
tensor_path = path + logp + 'baseline.csv'

# model_path = './model/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'


# ### 3. 数据集 标准化

# In[3]:


# # 列的提取
# # my_matrix = np.loadtxt(open(TXT_DATA_FILE, "r"),usecols = （1,4,5）, dtype=np.str, delimiter="\t", skiprows=0)
# data = pd.read_excel(io=XLSX_DATA_FILE, sheet_name='爬虫数据', header=0)
# data=data.loc[:, ["评论内容","评论数","点赞数"]]
# data["分类"] = 1
# data.to_csv(CSV_DATA_FILE,index=False,header=True,  encoding="utf-8")


# ### 4. 数据集 训练转化

# In[4]:


# translate to 1 hot
# 去非nan
data_all = pd.read_csv(data_all_file)
data_all = data_all[~(data_all["评论内容"].isnull())]
# list_sentences_test = test["评论内容"].fillna("CVxTz").values

# 变成one hot
data_all['postive']=0
data_all['neutral']=0
data_all['negative']=0
data_all.loc[data_all['分类'] == 1, 'postive'] = 1
data_all.loc[data_all['分类'] == 0, 'neutral'] = 1
data_all.loc[data_all['分类'] == -1, 'negative'] = 1

# 拆分测试集
train_all = data_all.sample(frac=1).reset_index(drop=True)  
lenth_train = train_all.shape[0]
spint = int(0.8*lenth_train)
train = train_all.loc[0:spint,:]
test = train_all.loc[spint:,:]
train.to_csv(train_file, index=False)
test.to_csv(test_file, index=False)


# ### 5. 开始训练

# In[5]:


max_features = 20000
maxlen = 100

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

for i1 in train.index:
    train.loc[i1, "评论内容"]=" ".join(jieba.cut(train.loc[i1, "评论内容"]))
for i1 in test.index:
    test.loc[i1, "评论内容"]=" ".join(jieba.cut(test.loc[i1, "评论内容"]))
    
list_sentences_train = train["评论内容"].fillna("CVxTz").values
list_sentences_test = test["评论内容"].fillna("CVxTz").values
# list_classes = [i1 for i1 in train.columns]
list_classes = ["negative","neutral","postive"]
try:
    list_classes.remove("评论内容")
    list_classes.remove("id")
except Exception as e:
    pass

y = train[list_classes].values

# list_sentences_test = test["评论内容"].fillna("CVxTz").values


# In[6]:


tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)


# In[7]:


def get_model(memn,dropn):
    embed_size = 128
    # 时间步 = maxlen
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    # memory units = 50
    x = Bidirectional(LSTM(memn, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(dropn)(x)
    x = Dense(memn, activation="relu")(x)
    x = Dropout(dropn)(x)
    x = Dense(3, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# In[8]:


batch_size = 32
epochs = 100

checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorb = TensorBoard(log_dir=tensor_path, histogram_freq=10, write_graph=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
callbacks_list = [checkpoint, early, tensorb] #early

mlist = list(range(1,7))
droplist = list(range(1,7))
for memn in mlist:
    for dropn in droplist:
        model = get_model(memn*50, dropn*0.1)
        start = time.time()
        model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=callbacks_list)
        end = time.time()
        print(end-start)
        print(memn*50, dropn*0.1)
        print("*".center(100,"#"))


# In[ ]:


model.load_weights(model_path)

y_test = model.predict(X_te)
# predict(self, x, batch_size=32, verbose=0)
# predict_classes(self, x, batch_size=32, verbose=1)
# predict_proba(self, x, batch_size=32, verbose=1)
# evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)

sample_submission = pd.read_csv(test_file)

sample_submission[list_classes] = y_test
sample_submission["max"]=sample_submission[list_classes].max(axis=1)

for indexs in sample_submission.index:  
    for  i2 in list_classes:  
        if(sample_submission.loc[indexs,i2] ==sample_submission.loc[indexs,"max"]):
            sample_submission.loc[indexs,"predict"]=i2
for i1 in list_classes:
    sample_submission.rename(columns={i1: "pred_" + i1}, inplace=True)
sample_submission.to_csv(res_file, index=False)


# In[ ]:


# 正确率评估
score = model.evaluate(X_t, y, batch_size=batch_size)
print(score)
test_pd = pd.read_csv(test_file)
res_pd = pd.read_csv(res_file)
total_pd=pd.concat([res_pd,test_pd], join='outer', axis=1)
total_right=0
total_num=0
for i1 in list_classes:
    tmp_obj=total_pd[total_pd[i1] == 1]
    sum_num=tmp_obj.shape[0]
    right_num=tmp_obj[tmp_obj["predict"] == i1].shape[0]
    total_right += right_num
    total_num += sum_num
    try:
        print("%s, sum_num: %d, right_num: %d, accuracy: %.3f" % (i1,sum_num,right_num,right_num/sum_num))
    except Exception as e:
        print("%s, sum_num: %d, right_num: %d, error: %s" % (i1,sum_num,right_num,str(e)))
        
print("total data, total_num: %d, total_right: %d, accuracy: %.3f" % (total_num,total_right,total_right/total_num))

