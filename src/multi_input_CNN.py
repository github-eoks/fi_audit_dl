
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
import openpyxl


df = pd.read_excel('./data/fi_target_clean.xls', sheet_name='Sheet1')

print(len(df))
mask = df['Stock'].isin([670,680,760,850,910,970,1130,1140,1260,1470,2810,4770,5180,6650,7070,7120,7460,7860,8040,9140,9310,9580,
                         9680,9770,9810,9970,10640,10820,11230,11330,11810,12030,13700,15890,16740,18250,18500,19490,20000,21240,21820,
                         23450,23810,25530,25620,26960,27410,27740,28050,28100,30720,31430,31820,32640,33250,33780,34300,35720,36460,36580,37270,
                         37710,39130,39570,44380,44820,49770,55490,57050,69460,69640,69960,71090,71950,71970,75580,77500,77970,79550,84670,89590,92780,93230,
                         95570,95720,96760,97230,101530,104700,111110,111770,130660,134380,138250,143210,145210,145270,145720,194370,195870,204210,204320,
                         207940,210980,213500,214320,214330,214390,214420,226320,227840,229640,234080,241560,241590,248170,249420,251270,264900,267250,267260,
                         267270,267290,268280,271560,271980,272450,280360,281820,282330,282690,284740])
df = df[~mask]





print(df.columns)
print(len(df))

# df = df.sample(n=100)
# target = '2t_280500/경제적부가가치'
# target = '2t_281100/시장부가가치'
target = '2t_192070/  자기자본순이익율'
# target = '2t_192020/  총자본순이익율'


text = '사업보고서'
# target = '2t_192070/  자기자본순이익율'

print('총 샘플의 수 :',len(df))
df[target].value_counts().plot(kind='bar');

df[text] = df[text].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

df[text] = df[text].str.replace('^ +', "") # white space 데이터를 empty value로 변경
df[text].replace('', np.nan, inplace=True)
print(df[text].isnull().sum())


X_data = df[text]
y_data = df[target]
print('본문의 개수: {}'.format(len(X_data)))
print('레이블의 개수: {}'.format(len(y_data)))

print(X_data.head())
######################################################
######################################################
######################################################
print(X_data.isnull().values.any())
X_data = X_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(X_data.isnull().values.any()) # Null 값이 존재하는지 확인
print('전처리 후 테스트용 샘플의 개수 :',len(X_data))

import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt


# 형태소 분석기 OKT를 사용한 토큰화 작업
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
okt = Okt()
tokenized_data = []

for sentence in X_data:
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
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

threshold = 3
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
######################################################
drop_data = [index for index, sentence in enumerate(X_data) if len(sentence) < 1]
X_data = np.delete(X_data, drop_data, axis=0)
print(len(X_data))


######################################################
######################################################
print('리뷰의 최대 길이 :',max(len(l) for l in X_data))
print('리뷰의 평균 길이 :',sum(map(len, X_data))/len(X_data))
plt.hist([len(s) for s in X_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))
max_len = 4500
below_threshold_len(max_len, X_data)

X_data = pad_sequences(X_data, maxlen = max_len)
######################################################
######################################################


######################################################
######################################################
model = Word2Vec(sentences = tokenized_data, size= 300, window = 5, min_count = 5, workers = 4, sg = 0)

######################################################
print(model.wv.vectors.shape)

print(model.wv.most_similar("매출"))

######################################################
######################################################


fi_va = ['195030/  법인세비용차감전순이익(종업원1인당)',
'191080/  순이익증가율',
'191050/  자기자본증가율',
'192300/  배당성향',
'192030/  기업법인세비용차감전순이익율',
'192040/  기업순이익율',
'192050/  경영자본영업이익율',
'192060/  자기자본법인세비용차감전순이익율']


X_data_fi = df.loc[:,fi_va]

# X_data_fi = df.iloc[:,16:]
print(X_data_fi.columns)
X_data_fi = np.array(X_data_fi, dtype='int')
X_data_fi = np.nan_to_num(X_data_fi, nan=0)

######################################################
######################################################

X_data_conat = np.concatenate((X_data, X_data_fi), axis=1)



X_train, X_test, y_train, y_test = train_test_split(
     X_data_conat, y_data, test_size=0.20, random_state=42)

X_text_train = X_train[:,:4500]
X_fi_train = X_train[:,4500:]
X_text_test = X_test[:,:4500]
X_fi_test = X_test[:,4500:]


print('훈련 데이터의 크기(shape):', X_train.shape)
# print('검증 데이터의 크기(shape):', X_val.shape)
print('훈련 데이터 레이블의 개수(shape):', y_train.shape)
# print('검증 데이터 레이블의 개수(shape):', y_val.shape)
print('테스트 데이터의 개수 :', len(X_test))
print('테스트 데이터 레이블의 개수 :', len(y_test))




# 구글의 사전 훈련된 Word2vec 모델을 로드합니다.
# word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
word2vec_model = model
# print(word2vec_model.wv.vectors.shape) # 모델의 크기 확인

embedding_matrix = np.zeros((vocab_size, 300))
# 단어 집합 크기의 행과 300개의 열을 가지는 행렬 생성. 값은 전부 0으로 채워진다.
np.shape(embedding_matrix)

def get_vector(word):
    if word in word2vec_model:
        return word2vec_model[word]
    else:
        return None

for word, i in tokenizer.word_index.items(): # 훈련 데이터의 단어 집합에서 단어와 정수 인덱스를 1개씩 꺼내온다.
    temp = get_vector(word) # 단어(key) 해당되는 임베딩 벡터의 300개의 값(value)를 임시 변수에 저장
    if temp is not None: # 만약 None이 아니라면 임베딩 벡터의 값을 리턴받은 것이므로
        embedding_matrix[i] = temp # 해당 단어 위치의 행에 벡터의 값을 저장한다.


# print(word2vec_model['business'])
# print('단어 business의 정수 인덱스 :', tokenizer.word_index['business'])
print(embedding_matrix[1])

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



######################################################################################
######################################################################################
######################################################################################


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Dropout, MaxPooling1D
from tensorflow.keras.layers import concatenate, MaxPooling1D
from tensorflow.keras import Input, Model
from keras.layers.merge import concatenate
from tensorflow.keras import optimizers
import os
import numpy



#########################################################################
sequence_input = Input(shape=(max_len,), dtype='int32', name='text_input')
embedded_sequences = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_len, mask_zero = True)(sequence_input)
conv1 = Conv1D(128, kernel_size=9, strides=1, padding='valid', activation='relu')(embedded_sequences)
pool1 = MaxPooling1D()(conv1)
flatten = Flatten()(pool1)

dense1 = Dense(128, activation="relu")(flatten)
dropout1 = Dropout(0.2)(dense1)

output = Dense(4, activation="sigmoid")(dropout1)

########################################################################
########################################################################


fi_input = Input(shape=(X_data_fi.shape[1], ), name='fi_input')

dense2 = Dense(4, activation="relu")(fi_input)
dropout2 = Dropout(0.2)(dense2)
output2 = Dense(4, activation="sigmoid")(dropout2)

concat_model = concatenate([output, output2])
dense3 = Dense(20, activation="relu")(concat_model)
dropout3 = Dropout(0.2)(dense3)
output3 = Dense(2, activation="sigmoid")(dropout3)

model = Model(inputs=[fi_input, sequence_input], outputs=output3)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
model.summary()


history = model.fit({'tex_input':X_text_train, 'fi_input':X_fi_train}, y_train,
          batch_size=32,
          epochs=5,
          validation_data = ({'tex_input':X_text_train, 'fi_input':X_fi_train}, y_train))
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


# y_pred = model.predict(X_test)
y_pred = model.predict({'tex_input':X_text_test, 'fi_input':X_fi_test})
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


