import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
# 加载数据
data = pd.read_csv(r'C:\Users\86133\OneDrive - Ulster University\ML_PR大作业\数据集\IMDB Dataset.csv')
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
print(data['sentiment'].unique())
print(data.head())  # 查看数据的前几行

# 假设CSV文件中的两列分别是'review'和'sentiment'
texts = data['review']  # 或者你CSV中对应的列名
labels = data['sentiment']  # 正面为1，负面为0，或者相应的标签

# 文本向量化
tokenizer = Tokenizer(num_words=10000)  # 仅考虑最常见的10000个词
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# 序列填充
data = pad_sequences(sequences, maxlen=500)  # 假设我们取每个序列的最大长度为500

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 清洗训练集中的NaN值
train_mask = ~np.any(np.isnan(X_train), axis=1) & ~np.isnan(y_train)  # 特征和标签都不含NaN
X_train = X_train[train_mask]
y_train = y_train[train_mask]

# 清洗测试集中的NaN值
test_mask = ~np.any(np.isnan(X_test), axis=1) & ~np.isnan(y_test)  # 特征和标签都不含NaN
X_test = X_test[test_mask]
y_test = y_test[test_mask]

# 打印处理后的数据形状以确认
print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing data shape:", X_test.shape)
print("Testing labels shape:", y_test.shape)


# # 将训练集和测试集的数据转换为DataFrame以便更易于查看
# train_df = pd.DataFrame(X_train, columns=[f'Feature_{i}' for i in range(X_train.shape[1])])
# train_df['Label'] = y_train
#
# test_df = pd.DataFrame(X_test, columns=[f'Feature_{i}' for i in range(X_test.shape[1])])
# test_df['Label'] = y_test
#
# # 删除包含NA的行
# train_df.dropna(inplace=True)
# test_df.dropna(inplace=True)
#
# # 重置索引，确保没有间断
# train_df.reset_index(drop=True, inplace=True)
# test_df.reset_index(drop=True, inplace=True)
#
# # 打印结果以确认操作
# print("Training set sample:")
# print(train_df.head())
# print(train_df['Label'].head())
#
# print("Testing set sample:")
# print(test_df.head())
# print(test_df['Label'].head())
#
# # 显示测试集的前几行
# print("Testing set sample:")
# print(test_df.head())

# 设置参数
vocab_size = 10000  # 词汇表大小，与Tokenizer中使用的num_words相同
max_len = 500       # 与pad_sequences使用的maxlen相同
embedding_dim = 128  # 嵌入层的维度

# 构建模型
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))  # 使用sigmoid激活函数，因为这是一个二分类问题

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 训练模型
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
# 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Loss: ", test_loss)
print("Test Accuracy: ", test_acc)

# 保存模型
model.save('LSTMmoxing.h5')  # 保存为HDF5文件



# 绘制训练和验证的准确率
plt.figure()
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# 绘制训练和验证的损失值
plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

model = tf.keras.models.load_model('LSTMmoxing.h5')



    # 预测测试集
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # 由于使用sigmoid，需要转换概率为二进制输出

# 计算精确度、召回率和F1分数
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
plt.yticks(tick_marks, ['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.grid(False)
plt.show()



y_train = y_train.reset_index(drop=True)
# 参数设置
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 准备交叉验证
acc_scores = []
loss_scores = []

for train_index, val_index in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # 训练模型
    history = model.fit(X_train_fold, y_train_fold,
                        validation_data=(X_val_fold, y_val_fold),
                        epochs=1, batch_size=64, verbose=0)  # verbose=0为了减少输出信息

    # 计算验证分数
    val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    acc_scores.append(val_acc)
    loss_scores.append(val_loss)

# 输出平均交叉验证分数
print(f'Average Validation Accuracy: {np.mean(acc_scores):.4f} +/- {np.std(acc_scores):.4f}')
print(f'Average Validation Loss: {np.mean(loss_scores):.4f}')