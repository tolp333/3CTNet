import numpy as np

import pandas as pd
import seaborn as sns
import tensorflow as tf

from numba import cuda

import matplotlib.pyplot as plt
from keras import Sequential, Input, Model
from keras.applications.densenet import layers
from keras.layers import Conv1D, BatchNormalization, Dropout, MaxPooling1D, Dense, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.keras.backend.clear_session()

# 清理GPU内存
device = cuda.get_current_device()
device.reset()

pd.set_option('display.max_rows', None)

# 加载数据
data = pd.read_csv(r"data/6分类_ADASYN_原始_3524.csv")
# data = data.iloc[:,200:1025]
# data = shuffle(data)
# print(data.shape)

# 将字符串标签转化为数字标签 ss砂岩  sh页岩   dl白云石  pc纯水泥  cs1-1水泥和沙子
label_mapping = {'cs1_1': 0, 'cs2_1': 1, 'dl': 2, 'pc': 3, 'sh': 4, 'ss': 5}
data['labels'] = data['labels'].replace(label_mapping)

# 获取特征值和标签值
x = data.iloc[:, 0:1024].values
y = data.iloc[:, 1024].values

# 将label数字标签转换为计算机便于识别的one-hot编码
y = to_categorical(y)

# 训练集和测试集的划分
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

# 数据标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
# print(x_test)
# print(x_train.shape[1:])

# 训练集和测试集增加维度
x_train = np.reshape(x_train, (x_train.shape[0], x.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x.shape[1], 1))
# y_train = np.reshape(y_train, (y_train.shape[0], 1, y.shape[1]))
# y_test = np.reshape(y_test, (y_test.shape[0], 1, y.shape[1]))

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# transformer_模块 实现了原理图的左侧内容,即Encoder
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.4):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim, activation="relu")])
        self.ffn = Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim, activation="relu")])
        self.layernorm1 = LayerNormalization(epsilon=1e-4)
        self.layernorm2 = LayerNormalization(epsilon=1e-4)  # 层归一化
        self.layernorm3 = LayerNormalization(epsilon=1e-4)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, inputs, training):  # transformer中使用的是LayerNormalization而不是batchnormalization
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Add残差连接
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out1 + ffn_output)
        return out2+out3


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        # self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = tf.reshape(x, [-1, maxlen, embed_dim])
        out = x + positions
        return out



maxlen = 1024  # Only consider 3 input time points
embed_dim = 1  # Features of each time point
num_heads = 8  # Number of attention heads
ff_dim = 64  # Hidden layer size in feed forward network inside transformer

# Input Time-series
inputs = Input(shape=(maxlen * embed_dim, 1))

'''将输入的时间序列随机编码成embedding向量'''
embedding_layer = TokenAndPositionEmbedding(maxlen, embed_dim)
x = embedding_layer(inputs)

x = Conv1D(64, 64, activation="relu", strides=64, padding="same")(x)
# x = Conv1D(64, 16, activation="relu", strides=1, padding="same")(x)
x = Dropout(0.3)(x)
x = MaxPooling1D()(x)

x = Conv1D(128, 16, activation="relu", strides=1, padding="same")(x)
# x = Conv1D(128, 16, activation="relu", strides=1, padding="same")(x)
x = Dropout(0.3)(x)
x = MaxPooling1D()(x)


x = Conv1D(256, 16, activation="relu", strides=1, padding="same")(x)
# x = Conv1D(256, 16, activation="relu", strides=1, padding="same")(x)
x = Dropout(0.3)(x)
x = MaxPooling1D()(x)

# transformer 模块
transformer_block_1 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
# transformer_block_2 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
x = transformer_block_1(x)
# x = transformer_block_2(x)
x = BatchNormalization()(x)

x = GlobalAveragePooling1D()(x)
x = Dropout(0.3)(x)
x = Dense(60, activation="softmax")(x)
x = Dropout(0.3)(x)
outputs = Dense(6, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

# 绘制模型结构
model.summary()

'''----------------------------------------------------模型训练----------------------------------------------------------'''
'''配置模型训练参数'''
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.00018), metrics=['accuracy'])
'''训练模型'''  # epochs=1000
history = model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_data=(x_test, y_test), shuffle=True,
                    verbose=1)

'''绘制训练曲线'''
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Transformer model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Transformer model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

'''输出训练、测试精确度'''
model_acc = model.evaluate(x_test, y_test, verbose=0)[1]
print("Test Accuracy: {:.3f}%".format(model_acc * 100))

model_acc = model.evaluate(x_train, y_train, verbose=0)[1]
print("Train Accuracy: {:.3f}%".format(model_acc * 100))

# 输出测试准确率
print(history.history['val_accuracy'])
# 输出测试损失
print(history.history['val_loss'])

# 绘制混淆矩阵
prediction = model.predict(x_test)
cm = confusion_matrix(prediction.argmax(axis=1), y_test.argmax(axis=1))

print("Confusion Matrix:\n", cm)

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=True, cmap='Blues')
plt.xticks(np.arange(6) + 0.5, label_mapping.keys())
plt.yticks(np.arange(6) + 0.5, label_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
clr = classification_report(prediction.argmax(axis=1), y_test.argmax(axis=1), target_names=label_mapping.keys())

print("Classification Report:\n----------------------\n", clr)
