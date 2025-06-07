# -*- coding: utf-8 -*-
"""
نموذج ML-T1 المتقدم - إصدار مصحح
"""

import os
import re
import pickle
import numpy as np
from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, 
    Dropout, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, 
    ReduceLROnPlateau
)
from sklearn.model_selection import train_test_split
import arabic_reshaper
from bidi.algorithm import get_display

# تفعيل وضع الأداء العالي
tf.config.optimizer.set_jit(True)
tf.config.run_functions_eagerly(True)

# 1. إعداد المعلمات الأساسية
MODEL_NAME = "ML-T1-Fixed"
MODEL_FILE = f"{MODEL_NAME}.h5"
TOKENIZER_FILE = f"{MODEL_NAME}_tokenizer.pkl"

# معلمات النموذج (حسب طلبك)
MAX_SEQUENCE_LEN = 15  # طول السياق
EMBEDDING_DIM = 128    # أبعاد التضمين
LSTM_UNITS = 128       # وحدات LSTM
DENSE_UNITS = 256      # وحدات الطبقة الكثيفة
BATCH_SIZE = 128       # حجم الدُفعة
EPOCHS = 10            # عدد العصور
VOCAB_LIMIT = 30000    # الحد الأقصى للمفردات
SAMPLE_SIZE = 1000     # عدد الجمل للتدريب

# 2. تحميل وتجهيز البيانات
def load_and_preprocess_data():
    """تحميل وتنظيف البيانات"""
    print("🚀 جاري تحميل البيانات...")
    try:
        # استخدام بيانات تجريبية أصغر
        texts = [
            "العلم نور والجهل ظلام",
            "التعليم في الصغر كالنقش على الحجر",
            "اللغة العربية هي لغة الضاد",
            "التكنولوجيا تغير العالم بسرعة",
            "الذكاء الاصطناعي مستقبل البشرية",
            "القراءة توسع آفاق الإنسان",
            "التاريخ يعلمنا دروساً قيمة",
            "الفلسفة تساعدنا على فهم الوجود",
            "الرياضيات لغة الكون",
            "الإبداع هو جوهر التقدم"
        ] * 100  # تكرار لإنشاء 1000 جملة
        print("✅ تم إنشاء بيانات تجريبية (1000 جملة)")
        return texts
    except Exception as e:
        print(f"❌ خطأ في تحميل البيانات: {e}")
        return []

def preprocess_text(text: str) -> str:
    """تنظيف النص العربي"""
    if not isinstance(text, str):
        return ""
    # إزالة التشكيل والحروف غير العربية
    text = re.sub(r'[\u064B-\u065F]', '', text)  # إزالة التشكيل
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)  # إبقاء العربية والمسافات
    text = re.sub(r'\s+', ' ', text).strip()  # إزالة المسافات الزائدة
    return text

# 3. تجهيز Tokenizer (النسخة المصححة)
def prepare_tokenizer(sentences):
    """إنشاء Tokenizer جديد مع ضبط حجم المفردات"""
    print("🔄 جاري إنشاء Tokenizer جديد...")
    tokenizer = Tokenizer(num_words=VOCAB_LIMIT, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    
    # حفظ Tokenizer
    with open(TOKENIZER_FILE, "wb") as f:
        pickle.dump(tokenizer, f)
    
    vocab_size = min(len(tokenizer.word_index) + 1, VOCAB_LIMIT)
    print(f"🔤 حجم القاموس: {vocab_size}")
    return tokenizer, vocab_size

# 4. مولد البيانات المتقدم (النسخة المصححة)
class AdvancedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, sentences, tokenizer, max_seq_len, batch_size, shuffle=True):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.vocab_size = min(len(tokenizer.word_index) + 1, VOCAB_LIMIT)
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.sentences) / self.batch_size))
    
    def __getitem__(self, index):
        batch_sentences = self.sentences[index*self.batch_size:(index+1)*self.batch_size]
        X, y = [], []
        
        for sentence in batch_sentences:
            if not sentence:
                continue
            tokens = self.tokenizer.texts_to_sequences([sentence])[0]
            
            # تأكيد أن جميع الرموز ضمن النطاق الصحيح
            tokens = [t if t < self.vocab_size else 1 for t in tokens]  # 1 = <OOV>
            
            for i in range(1, len(tokens)):
                start_idx = max(0, i - self.max_seq_len + 1)
                seq = tokens[start_idx:i+1]
                if len(seq) < 2:
                    continue
                padded_seq = pad_sequences([seq], maxlen=self.max_seq_len, padding='pre')[0]
                X.append(padded_seq[:-1])
                y.append(padded_seq[-1])
        
        if len(X) == 0:
            return np.zeros((1, self.max_seq_len-1)), np.zeros((1,))
        
        return np.array(X), np.array(y)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.sentences)

# 5. بناء النموذج المتقدم (النسخة المصححة)
def build_advanced_model(vocab_size, max_seq_len):
    model = Sequential(name=MODEL_NAME)
    
    # طبقة التضمين مع تأكيد حجم المفردات
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    ))
    
    model.add(Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)))
    model.add(LayerNormalization())
    model.add(Dropout(0.3))
    
    model.add(Bidirectional(LSTM(LSTM_UNITS // 2)))
    model.add(LayerNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(DENSE_UNITS, activation='relu'))
    model.add(LayerNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(DENSE_UNITS // 2, activation='relu'))
    model.add(LayerNormalization())
    
    model.add(Dense(vocab_size, activation='softmax'))
    
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    model.build((None, max_seq_len-1))
    model.summary()
    return model

# 6. نظام توليد النص المتقدم
def generate_deep_text(seed_text, next_words, model, tokenizer, max_seq_len, temperature=0.7):
    if not seed_text:
        return ""
    
    output = seed_text
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([output])[0]
        
        # تأكيد أن جميع الرموز ضمن النطاق الصحيح
        token_list = [t if t < model.input_shape[-1] else 1 for t in tokenizer.texts_to_sequences([output])[0]]
        
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        
        predictions = model.predict(token_list, verbose=0)[0]
        predictions = np.log(predictions) / max(temperature, 0.1)
        exp_preds = np.exp(predictions)
        probs = exp_preds / np.sum(exp_preds)
        
        predicted_idx = np.random.choice(len(probs), p=probs)
        predicted_word = tokenizer.index_word.get(predicted_idx, "")
        
        if not predicted_word or predicted_word == "<OOV>":
            break
            
        output += " " + predicted_word
        
        if predicted_word in [".", "؟", "!"]:
            break
            
    reshaped_text = arabic_reshaper.reshape(output)
    return get_display(reshaped_text)

# 7. التدريب الرئيسي (النسخة المصححة)
def main():
    print("⚠️ التدريب على CPU (يوصى باستخدام GPU للأداء الأمثل)")
    
    # تحميل وتجهيز البيانات
    texts = load_and_preprocess_data()
    if not texts:
        print("❌ فشل في تحميل البيانات")
        return
    
    sentences = []
    for text in texts:
        cleaned = preprocess_text(text)
        if cleaned:
            sentences.append(cleaned)
    
    # استخدام فقط العدد المطلوب من الجمل
    sentences = sentences[:SAMPLE_SIZE]
    print(f"✅ عدد الجمل المستخدمة: {len(sentences)}")
    
    # إنشاء Tokenizer جديد دائماً
    tokenizer, vocab_size = prepare_tokenizer(sentences)
    
    # بناء النموذج
    model = build_advanced_model(vocab_size, MAX_SEQUENCE_LEN)
    
    # تقسيم البيانات
    train_sents, val_sents = train_test_split(sentences, test_size=0.1, random_state=42)
    
    # إنشاء مولدات البيانات
    train_generator = AdvancedDataGenerator(train_sents, tokenizer, MAX_SEQUENCE_LEN, BATCH_SIZE)
    val_generator = AdvancedDataGenerator(val_sents, tokenizer, MAX_SEQUENCE_LEN, BATCH_SIZE, shuffle=False)
    
    # إعداد نظام المراقبة
    callbacks = [
        ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=1e-6, verbose=1)
    ]
    
    # التدريب
    print("\n🔥 بدء تدريب النموذج")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # اختبار النموذج
    print("\n🎯 اختبار النموذج بعد التدريب:")
    test_seeds = ["العلم هو", "التكنولوجيا", "الذكاء الاصطناعي"]
    for seed in test_seeds:
        generated = generate_deep_text(seed, 8, model, tokenizer, MAX_SEQUENCE_LEN)
        print(f"\nالبذرة: {seed}\nالإكمال: {generated}")
    
    # الواجهة التفاعلية
    print("\n🤖 جاهز للتجربة! أدخل 'خروج' للإنهاء")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ["خروج", "exit", "quit"]:
            break
        generated = generate_deep_text(user_input, 10, model, tokenizer, MAX_SEQUENCE_LEN)
        print(f"الإكمال: {generated}\n")

if __name__ == "__main__":
    main()