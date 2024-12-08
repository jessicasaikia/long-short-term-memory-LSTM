import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, TimeDistributed

data = pd.read_csv('/content/LSTM.csv')

sentences = data['Sentence'].apply(lambda x: x.split())
pos_tags = data['POS_Tags'].apply(lambda x: x.split())

tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(sentences)
X_seq = tokenizer.texts_to_sequences(sentences)
max_length = max(len(seq) for seq in X_seq)
X_padded = pad_sequences(X_seq, padding='post', maxlen=max_length)

pos_tokenizer = Tokenizer(lower=False)
pos_tokenizer.fit_on_texts(pos_tags)
y_seq = pos_tokenizer.texts_to_sequences(pos_tags)
y_padded = pad_sequences(y_seq, padding='post', maxlen=max_length)

X_train, X_test, y_train, y_test = train_test_split(X_padded, y_padded, test_size=0.2, random_state=42)

y_train = np.array([np.array(i) for i in y_train])
y_test = np.array([np.array(i) for i in y_test])

embedding_dim = 100
num_pos_tags = len(pos_tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_length))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.1))
model.add(TimeDistributed(Dense(num_pos_tags, activation='softmax')))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")


def process_input(user_input):
    if user_input.endswith('.csv'):
        user_data = pd.read_csv(user_input)
        sentences_input = user_data['Sentence'].apply(lambda x: x.split()).tolist()
    else:
        sentences_input = [user_input.split()]

    X_input_seq = tokenizer.texts_to_sequences(sentences_input)
    X_input_padded = pad_sequences(X_input_seq, padding='post', maxlen=max_length)

    return X_input_padded, sentences_input


def predict_pos():
    user_input = input("Please enter a sentence or the path to a CSV file: ")

    X_input_padded, sentences_input = process_input(user_input)

    predictions = model.predict(X_input_padded)
    results = []
    for i, (sentence, pred_tags) in enumerate(zip(sentences_input, predictions)):
        tokens = sentence
        pred_tag_indices = np.argmax(pred_tags, axis=-1)
        predicted_tag_names = pos_tokenizer.sequences_to_texts([pred_tag_indices])[0].split()

        results.append({
            "Sentence": ' '.join(tokens),
            "Predicted POS Tags": ' '.join(predicted_tag_names)
        })

        print(f"\nSentence {i+1}:")
        print("Tokens:", tokens)
        print("Predicted POS Tags:", predicted_tag_names)
        print()

    results_df = pd.DataFrame(results)
    results_df.to_csv('/content/LSTMoutput.csv', index=False)
    print(results_df.head())

predict_pos()
