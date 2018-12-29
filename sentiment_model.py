# coding: utf-8

# ## 情感模型
# ### 基于词典的情感，打分情感分析模型
# - 1.加载情感词词典，否定词词典，程度副词词典，
# - 2.分词，并返回分词后的列表
# - 3.用列表存储，情感词、否定词、程度副词的索引位置
# - 4.对句子进行打分

# In[1]:


from collections import defaultdict
import os
import re
import jieba
import codecs

# In[2]:


# 修改各词库的路径
stopword_path = 'data/stop_words.txt'
degreeword_path = 'data/degreewords.txt'
sentimentword_path = 'data/BosonNLP_sentiment_score_keep.txt'


# In[3]:


# 加载新词库
def reload_dict(stopword_path, degreeword_path, sentimentword_path):
    """
    加载返回基础词典
    Agruments:
    stopword_path --停用词词典目录
    degreeword_path --程度副词的目录
    sentimentword_path --情感词目录
    Returns：
    stopswords --列表形式的停用词
    degree_dict --程度副词词典及其分数
    sentiment_dict --情感词及其强度值字典格式。
    notword --否定词列表
    """
    jieba.load_userdict('data/stock_dict.txt')

    # 停用词列表
    stopword_hand = open(stopword_path, "r", encoding='utf-8')
    stopword_file = stopword_hand.readlines()
    stopwords = [word.replace("\n", "") for word in stopword_file]

    # 否定词表
    notword = [u'不', u'没', u'无', u'非', u'莫', u'弗', u'勿', u'毋', u'未', u'否', u'别', u'無', u'休', u'难道']

    # 程度词表
    degreeword_hand = open(degreeword_path, "r", encoding='utf-8')
    degreeword_file = degreeword_hand.readlines()
    # degreeword_file = open(degreeword_path).readlines()
    degree_dict = {}
    for word in degreeword_file:
        word = word.replace("\n", "").split(" ")
        degree_dict[word[0]] = word[1]

        # 情感词表

    sentimentword_hand = open(sentimentword_path, "r", encoding='utf-8')
    sentimentword_file = sentimentword_hand.readlines()
    sentiment_dict = {}
    for word in sentimentword_file:
        word = word.replace("\n", "").split(",")
        sentiment_dict[word[0]] = word[1]
    return stopwords, degree_dict, sentiment_dict, notword


# In[4]:


def sent2wordloc(sentence, del_stop=False, stopwords=None):
    """
    输入句子进行切分
    Agruments：
    sentence --一段文本
    del_stop --是否去除停用词，默认为FALSE
    stopwords --停用词库，在del_stop为TRUE情况下进行设置
    Returns：
    wordlist --分词后结果列表
    """

    wordlist = []
    if del_stop:
        wordlist = [word for word in jieba.cut(sentence) if word not in stopwords]
    else:
        wordlist = [word for word in jieba.cut(sentence)]

    return wordlist


print("The function sent2word is defined")


# In[5]:


def wordclassify(wordlist, sentiment_dict, notword, degree_dict):
    """
    获得各个词性的位置
    Arguments:
    wordlist --列表形式，是分词后的结果
    sentiment_dict --字典形式，是情感词及其强度值
    notword --列表形式，否定词字典
    degree_dict --字典形式，程度副词及其强度值
    Returns：
    sentimentloc --情感所在wordlist列表的位置
    notloc --否定词所在的位置
    degreeloc --程度副词所在的位置
    othersloc --其他词所在的位置
    """
    sentimentloc, notloc, degreeloc, othersloc = [], [], [], []
    for i in range(len(wordlist)):
        word = wordlist[i]
        if word in sentiment_dict.keys() and word not in notword and word not in degree_dict.keys():
            sentimentloc.append(i)
        elif word in notword and word not in degree_dict.keys():
            notloc.append(i)
        elif word in degree_dict.keys():
            degreeloc.append(i)
        else:
            othersloc.append(i)
    return sentimentloc, notloc, degreeloc, othersloc


print("The function wordclassify is defined")


# ### 打分逻辑
# - 首先定位情感词，从情感词往前进行，查找。如果遇到否定词，情感值为情感强度值乘以-1，情感极性改变。
# - 定位程度副词，如果有程度副词，乘以相应的程度强度。
# - 判断程度副词和否定词在一块情况下的，位置的先后顺序，如果在否定词在前，对原来情感强度有减弱的作用，这里乘以0.75，如果是在程度副词之后
# 具有加强的作用，乘以1.25。

# In[6]:


def sentscore(wordlist, sentimentloc, notloc, degreeloc, othersloc):
    """
    对句子进行评分
    Arguments:
    wordlist --列表形式，分词后的词语列表
    sentimentloc --列表形式，表示情感词在wordlist位置的列表
    notloc --列表形式，表示否定词在wordlist位置的列表
    degreeloc --列表形式，表示程度副词在wordlist位置列表
    othersloc --列表形式，表示其他词在wordlist的位置列表
    """
    score = 0
    for i in range(len(sentimentloc)):
        w = 1
        index = sentimentloc[i]
        j_no = -1
        j_fu = -1
        if i == 0:
            for j in range(0, sentimentloc[i]):
                if j in notloc:
                    j_no = j
                    w *= -1
                    print('now', w)
                elif j in degreeloc:
                    j_fu = j
                    w *= float(degree_dict[wordlist[j]])
        else:
            if index > 0:
                for j in range(sentimentloc[i - 1] + 1, sentimentloc[i]):
                    if j in notloc:
                        j_no = j
                        w *= -1
                    elif j in degreeloc:
                        j_fu = j
                        w *= float(degree_dict[wordlist[j]])
        if j_no > j_fu and j_no != -1 and j_fu != -1:
            score += w * float(sentiment_dict[wordlist[index]]) * 1.25  # 否定词在程度副词后
        elif j_no < j_fu and j_no != -1 and j_fu != -1:
            score += w * float(sentiment_dict[wordlist[index]]) * 0.75 / (
                float(degree_dict[wordlist[j_fu]]))  # 否定词在程度副词之前
        else:
            score += w * float(sentiment_dict[wordlist[index]])
    return score


print("The function sentscore is defined")

# In[7]:


s3 = '重磅！中国对美国128项进口商品加征关税(附清单)'
stopswords, degree_dict, sentiment_dict, notword = reload_dict(stopword_path, degreeword_path, sentimentword_path)
wordlist = sent2wordloc(s3)
sentimentloc, notloc, degreeloc, othersloc = wordclassify(wordlist, sentiment_dict, notword, degree_dict)
a3 = sentscore(wordlist, sentimentloc, notloc, degreeloc, othersloc)
print(a3)
