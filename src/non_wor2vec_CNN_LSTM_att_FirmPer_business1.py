
#ref: https://wikidocs.net/86083
#writer: Kyun Sun Eo, date: 21.05.10
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import gensim

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split



df = pd.read_excel('./data/fi_target_clean.xls', sheet_name='Sheet1')

print(len(df))
mask = df['Stock'].isin([670,680,760,850,910,970,1130,1140,1260,1470,2810,4770,5180,6650,7070,7120,7460,7860,8040,9140,9310,9580,9680,9770,9810,9970,10640,10820,11230,11330,11810,12030,13700,15890,16740,18250,18500,19490,20000,21240,21820,23450,23810,25530,25620,26960,27410,27740,28050,28100,30720,31430,31820,32640,33250,33780,34300,35720,36460,36580,37270,37710,39130,39570,44380,44820,49770,55490,57050,69460,69640,69960,71090,71950,71970,75580,77500,77970,79550,84670,89590,92780,93230,95570,95720,96760,97230,101530,104700,111110,111770,130660,134380,138250,143210,145210,145270,145720,194370,195870,204210,204320,207940,210980,213500,214320,214330,214390,214420,226320,227840,229640,234080,241560,241590,248170,249420,251270,264900,267250,267260,267270,267290,268280,271560,271980,272450,280360,281820,282330,282690,284740])
df = df[~mask]

print(df.head())
print(df.columns)
print(len(df))

# df = df.sample(n=100)
target = '2t_280500/경제적부가가치'
# target = '2t_281100/시장부가가치'
# target = '2t_192070/  자기자본순이익율'
# target = '2t_192020/  총자본순이익율'



text = '사업보고서'

print('총 샘플의 수 :',len(df))
df[target].value_counts().plot(kind='bar');

df[text] = df[text].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

df[text] = df[text].str.replace('^ +', "") # white space 데이터를 empty value로 변경



X_data = df[text]
y_data = df[target]
print('본문의 개수: {}'.format(len(X_data)))
print('레이블의 개수: {}'.format(len(y_data)))

print(X_data.head())
######################################################
######################################################
######################################################


import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
# 형태소 분석기 OKT를 사용한 토큰화 작업 (다소 시간 소요)
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
okt = Okt()
tokenized_data = []

for sentence in X_data:
    temp_X = okt.nouns(sentence) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    tokenized_data.append(temp_X)
    if len(tokenized_data) % 100 == 0:
        print(len(tokenized_data))



print(tokenized_data[:3])
X_data = tokenized_data
######################################################
######################################################
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_data)
print(tokenizer.word_index)

threshold = 5
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
vocab_size = total_cnt - rare_cnt + 1
print('단어 집합의 크기 :',vocab_size)
######################################################
######################################################
tokenizer = Tokenizer(vocab_size)
tokenizer.fit_on_texts(X_data)
X_data = tokenizer.texts_to_sequences(X_data)
print(X_data[:3])
y_data = pd.get_dummies(y_data)
y_data = np.array(y_data)

######################################################


print('리뷰의 최대 길이 : %d' % max(len(l) for l in X_data))
print('리뷰의 평균 길이 : %f' % (sum(map(len, X_data))/len(X_data)))
plt.hist([len(s) for s in X_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

max_len = max(len(l) for l in X_data)
X_data = pad_sequences(X_data, maxlen = max_len)
print("훈련 데이터의 크기(shape): ", X_data.shape)

######################################################
#https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
######################################################
######################################################


X_train, X_test, y_train, y_test = train_test_split(
     X_data, y_data, test_size=0.20, random_state=42)

n_of_val = int(0.1 * X_train.shape[0])
print(n_of_val)

X_train = X_train[:-n_of_val]
y_train = y_train[:-n_of_val]
X_val = X_train[-n_of_val:]
y_val = y_train[-n_of_val:]
X_test = X_test
y_test = y_test

print('훈련 데이터의 크기(shape):', X_train.shape)
# print('검증 데이터의 크기(shape):', X_val.shape)
print('훈련 데이터 레이블의 개수(shape):', y_train.shape)
# print('검증 데이터 레이블의 개수(shape):', y_val.shape)
print('테스트 데이터의 개수 :', len(X_test))
print('테스트 데이터 레이블의 개수 :', len(y_test))

######################################################################################
######################################################################################
######################################################################################

import tensorflow as tf
class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)

  def call(self, values, query): # 단, key와 value는 같음
    # query shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # score 계산을 위해 뒤에서 할 덧셈을 위해서 차원을 변경해줍니다.
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights



###############################
###CNN + LSTM attention########
###############################


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Dropout, MaxPooling1D
from tensorflow.keras.layers import concatenate, MaxPooling1D
from tensorflow.keras import Input, Model
from tensorflow.keras import optimizers
import os
import numpy

sequence_input = Input(shape=(max_len,), dtype='int32')
embedded_sequences = Embedding(vocab_size, 300, input_length=max_len, mask_zero = True)(sequence_input)
conv1 = Conv1D(128, kernel_size=9, strides=1, padding='valid', activation='relu')(embedded_sequences)
pool1 = MaxPooling1D()(conv1)



lstm = Bidirectional(LSTM(64, dropout=0.5, return_sequences = True))(pool1)

lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional \
  (LSTM(64, dropout=0.5, return_sequences=True, return_state=True))(lstm)

print(lstm.shape, forward_h.shape, forward_c.shape, backward_h.shape, backward_c.shape)

state_h = Concatenate()([forward_h, backward_h]) # 은닉 상태
state_c = Concatenate()([forward_c, backward_c]) # 셀 상태
attention = BahdanauAttention(64) # 가중치 크기 정의
context_vector, attention_weights = attention(lstm, state_h)

dense1 = Dense(20, activation="relu")(context_vector)
dropout = Dropout(0.5)(dense1)
output = Dense(2, activation="sigmoid")(dropout)
model = Model(inputs=sequence_input, outputs=output)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
model.summary()
history = model.fit(X_train, y_train,
          batch_size=32,
          epochs=5,
          validation_data = (X_val, y_val))
######################################################################################



###################################################################
###################################################################
###################################################################
epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['acc'])
plt.plot(epochs, history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='lower right')
plt.show()


y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=-1)
y_test = y_test.argmax(axis=-1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
roc_auc_score(y_test, y_pred)
print('accuracy: ', sum(y_pred == y_test) / len(y_test))
print('confusion matrix: \n', confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Precision, Recall and F1-Score:\n\n", classification_report(y_test, y_pred))

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


