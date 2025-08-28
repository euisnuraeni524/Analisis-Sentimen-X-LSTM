import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import joblib
import os


SEED = 42

def prepare_sequences(texts, labels, num_words=20000, max_len=100, test_size=0.2):
    np.random.seed(SEED)


    le = LabelEncoder()
    y = le.fit_transform(labels) # 0..C-1


    tok = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    seqs = tok.texts_to_sequences(texts)
    X = pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )


    num_classes = len(np.unique(y))
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)


    return tok, le, X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, num_classes


def build_lstm_model(num_words, max_len, num_classes, embed_dim=128, lstm_units=128, dropout=0.2):
    model = Sequential([
        Embedding(input_dim=num_words, output_dim=embed_dim, input_length=max_len),
        LSTM(lstm_units),
        Dropout(dropout),
        Dense(num_classes, activation='softmax')
        ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history):
    # figure kecil + dpi tinggi supaya rapat & enak di-SS
    fig1 = plt.figure(figsize=(4,3))
    plt.plot(history.history['loss'], label='Train Loss', linewidth=1)
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss', linewidth=1)
    plt.title('Loss', fontsize=9)
    plt.xlabel('Epoch', fontsize=8); plt.ylabel('Loss', fontsize=8)
    plt.legend(fontsize=7, loc='best')
    plt.tight_layout()

    fig2 = plt.figure(figsize=(4,3))
    plt.plot(history.history['accuracy'], label='Train Acc', linewidth=1)
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val Acc', linewidth=1)
    plt.title('Accuracy', fontsize=9)
    plt.xlabel('Epoch', fontsize=8); plt.ylabel('Accuracy', fontsize=8)
    plt.legend(fontsize=7, loc='best')
    plt.tight_layout()

    return fig1, fig2

def train_lstm(texts, labels, num_words=20000, max_len=100, epochs=5, batch_size=32, model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)

    tok, le, X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, num_classes = prepare_sequences(
        texts, labels, num_words=num_words, max_len=max_len
    )

    model = build_lstm_model(num_words=min(num_words, len(tok.word_index)+2), max_len=max_len, num_classes=num_classes)


    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=epochs, batch_size=batch_size, verbose=1
    )

    # Simpan artefak
    model_path = os.path.join(model_dir, 'lstm_model.h5')
    tok_path = os.path.join(model_dir, 'tokenizer.pkl')
    le_path = os.path.join(model_dir, 'label_encoder.pkl')
    model.save(model_path)
    joblib.dump(tok, tok_path)
    joblib.dump(le, le_path)


    # Evaluasi
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)


    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)


    return {
        'model': model,
        'tokenizer': tok,
        'label_encoder': le,
        'history': history,
        'train_test': (X_train, X_test, y_train, y_test),
        'metrics': {'confusion_matrix': cm, 'accuracy': acc, 'report': report},
        'paths': {'model': model_path, 'tokenizer': tok_path, 'label_encoder': le_path}
    }


def plot_confusion_matrix(cm, class_names):
    fig = plt.figure(figsize=(4,3))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix', fontsize=9)
    # colorbar kecil & rapat
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=7)
    plt.yticks(tick_marks, class_names, fontsize=7)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                fontsize=7,
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel('True', fontsize=8)
    plt.xlabel('Pred', fontsize=8)
    plt.tight_layout()
    return fig