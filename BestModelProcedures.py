import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# 定义用于预处理文本的函数
def preprocess_text(texts, tokenizer, maxlen=500):
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=maxlen)
    return data

# 加载模型
model = load_model('CNNmoxing.h5')

# 假设你已经有一个训练好的tokenizer
# 如果你没有保存tokenizer，你需要重建它并用相同的方式来初始化
data = pd.read_csv(r'C:\Users\86133\OneDrive - Ulster University\ML_PR大作业\数据集\IMDB Dataset.csv')
texts = data['review']  # 或者你CSV中对应的列名
tokenizer = Tokenizer(num_words=10000)  # 和之前训练时保持一致
tokenizer.fit_on_texts(texts)

# 输入新的文本
new_texts = ["This movie was fantastic! I loved it.", "Not a great movie, it was really boring."]

# 预处理文本
# 注意这里假设我们已经加载了与训练模型同时训练的tokenizer
processed_texts = preprocess_text(new_texts, tokenizer, maxlen=500)

# 使用模型进行预测
predictions = model.predict(processed_texts)
predicted_classes = (predictions > 0.5).astype(int)

# 打印预测结果
for text, prediction in zip(new_texts, predicted_classes):
    print(f"Review: '{text}'\nPredicted sentiment: {'Positive' if prediction[0] == 1 else 'Negative'}\n")

def preprocess_text(text, tokenizer, maxlen=500):
    sequence = tokenizer.texts_to_sequences([text])
    processed_text = pad_sequences(sequence, maxlen=maxlen)
    return processed_text
# 主函数，运行文本情感预测
def main():
    # 用户输入文本
    text = input("Please enter the text you want to analyse: ")
    # 预处理文本
    processed_text = preprocess_text(text, tokenizer, maxlen=500)
    # 使用模型进行预测
    prediction = model.predict(processed_text)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    # 输出预测结果
    print(f"Predicting Mood: {sentiment}")
if __name__ == "__main__":
    main()