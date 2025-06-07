# -*- coding: utf-8 -*-
"""
نموذج ML-T1 المتقدم - ذكاء اصطناعي عالي الأداء
تم تصميمه للتعامل مع كميات كبيرة من البيانات والتفكير العميق
"""

import os
import re
import pickle
import numpy as np
import multiprocessing
from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, 
    Dropout, LayerNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, 
    ReduceLROnPlateau, TensorBoard
)
from sklearn.model_selection import train_test_split
import arabic_reshaper
from bidi.algorithm import get_display

# تفعيل وضع الأداء العالي
tf.config.optimizer.set_jit(True)
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# 1. إعداد المعلمات الأساسية
MODEL_NAME = "ML-T1-Advanced"
MODEL_FILE = f"{MODEL_NAME}.h5"
TOKENIZER_FILE = f"{MODEL_NAME}_tokenizer.pkl"
LOG_DIR = f"{MODEL_NAME}_logs"

# معلمات النموذج
MAX_SEQUENCE_LEN = 15  # طول السياق
EMBEDDING_DIM = 256    # أبعاد التضمين
LSTM_UNITS = 512       # وحدات LSTM
DENSE_UNITS = 1024     # وحدات الطبقة الكثيفة
BATCH_SIZE = 2048      # حجم الدُفعة
EPOCHS = 50            # عدد العصور
VOCAB_LIMIT = 50000    # الحد الأقصى للمفردات

# 2. تحميل وتجهيز البيانات
def load_and_preprocess_data():
    """تحميل وتنظيف البيانات"""
    print("🚀 جاري تحميل بيانات Wiki40B (العربية)...")
    dataset = load_dataset("wiki40b", "ar", split="train")
    texts = dataset["text"][:100000]  # 100,000 مقالة
    
    # معالجة متوازية
    print("🧹 جاري تنظيف النصوص (متوازي)...")
    with multiprocessing.Pool() as pool:
        cleaned_texts = pool.map(preprocess_text, texts)
    
    # تقسيم إلى جمل
    print("🔢 جاري تقسيم النص إلى جمل...")
    sentences = []
    for text in cleaned_texts:
        parts = re.split(r"[\.؟!\n]", text)
        for part in parts:
            if part.strip():
                sentences.append(part.strip())
    
    print(f"✅ تم تحميل وتجهيز {len(sentences)} جملة")
    return sentences

def preprocess_text(text: str) -> str:
    """تنظيف النص العربي"""
    # إزالة التشكيل والحروف غير العربية
    text = re.sub(r'[\u064B-\u065F]', '', text)  # إزالة التشكيل
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)  # إبقاء العربية والمسافات
    text = re.sub(r'\s+', ' ', text).strip()  # إزالة المسافات الزائدة
    return text

# 3. تجهيز Tokenizer
def prepare_tokenizer(sentences):
    """إنشاء أو تحميل Tokenizer"""
    if os.path.exists(TOKENIZER_FILE):
        with open(TOKENIZER_FILE, "rb") as f:
            tokenizer = pickle.load(f)
        print(f"✅ تم تحميل Tokenizer من: {TOKENIZER_FILE}")
    else:
        tokenizer = Tokenizer(num_words=VOCAB_LIMIT, oov_token="<OOV>")
        tokenizer.fit_on_texts(sentences)
        
        # حفظ Tokenizer
        with open(TOKENIZER_FILE, "wb") as f:
            pickle.dump(tokenizer, f)
        print(f"✅ تم إنشاء وحفظ Tokenizer في: {TOKENIZER_FILE}")
    
    vocab_size = min(len(tokenizer.word_index) + 1, VOCAB_LIMIT)
    print(f"🔤 حجم القاموس: {vocab_size}")
    return tokenizer, vocab_size

# 4. مولد البيانات المتقدم
class AdvancedDataGenerator(tf.keras.utils.Sequence):
    """مولد بيانات متقدم للتعامل مع مجموعات البيانات الكبيرة"""
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
            tokens = self.tokenizer.texts_to_sequences([sentence])[0]
            for i in range(1, len(tokens)):
                start_idx = max(0, i - self.max_seq_len + 1)
                seq = tokens[start_idx:i+1]
                padded_seq = pad_sequences([seq], maxlen=self.max_seq_len, padding='pre')[0]
                X.append(padded_seq[:-1])
                y.append(padded_seq[-1])
        
        return np.array(X), to_categorical(y, num_classes=self.vocab_size)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.sentences)

# 5. بناء النموذج المتقدم
def build_advanced_model(vocab_size, max_seq_len):
    """بناء نموذج ML-T1 المتقدم"""
    model = Sequential(name="ML-T1-Advanced")
    
    # طبقة التضمين
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        input_length=max_seq_len - 1,
        mask_zero=True
    ))
    
    # طبقات LSTM ثنائية الاتجاه
    model.add(Bidirectional(LSTM(
        LSTM_UNITS, 
        return_sequences=True,
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ))
    model.add(LayerNormalization())
    model.add(Dropout(0.3))
    
    model.add(Bidirectional(LSTM(
        LSTM_UNITS // 2,
        kernel_regularizer=tf.keras.regularizers.l2(0.001))
    ))
    model.add(LayerNormalization())
    model.add(Dropout(0.3))
    
    # طبقات كثيفة
    model.add(Dense(DENSE_UNITS, activation='relu'))
    model.add(LayerNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(DENSE_UNITS // 2, activation='relu'))
    model.add(LayerNormalization())
    
    # طبقة الإخراج
    model.add(Dense(vocab_size, activation='softmax'))
    
    # المترجم مع جدولة معدل التعلم
    optimizer = AdamW(
        learning_rate=0.001,
        weight_decay=0.0001,
        clipnorm=1.0
    )
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    model.summary()
    return model

# 6. نظام توليد النص المتقدم
def generate_deep_text(seed_text, next_words, model, tokenizer, max_seq_len, temperature=0.7):
    """توليد نص عميق مع التحكم في الإبداع"""
    output = seed_text
    reshaped_output = arabic_reshaper.reshape(seed_text)
    bidi_output = get_display(reshaped_output)
    print(f"🌱 البذرة: {bidi_output}")
    
    for i in range(next_words):
        token_list = tokenizer.texts_to_sequences([output])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        
        # التنبؤ مع تحكم في درجة الحرارة
        predictions = model.predict(token_list, verbose=0)[0]
        predictions = np.log(predictions) / max(temperature, 0.1)
        exp_preds = np.exp(predictions)
        probs = exp_preds / np.sum(exp_preds)
        
        # أخذ عينة مع مرجحة الاحتمالات
        predicted_idx = np.random.choice(len(probs), p=probs)
        predicted_word = tokenizer.index_word.get(predicted_idx, "")
        
        # التوقف عند علامات النهاية
        if not predicted_word or predicted_word == "<OOV>":
            break
            
        output += " " + predicted_word
        
        # إيقاف عند نهاية الجملة
        if predicted_word in [".", "؟", "!"] and i > next_words//2:
            break
            
    # تشكيل النص العربي للعرض
    reshaped_text = arabic_reshaper.reshape(output)
    bidi_text = get_display(reshaped_text)
    return bidi_text

# 7. التدريب الرئيسي
def main():
    # تهيئة بيئة التدريب
    strategy = tf.distribute.MirroredStrategy() if tf.config.list_physical_devices('GPU') else None
    
    if strategy:
        print(f"🚀 تم الكشف عن {strategy.num_replicas_in_sync} GPUs")
        global BATCH_SIZE
        BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
        with strategy.scope():
            sentences = load_and_preprocess_data()
            tokenizer, vocab_size = prepare_tokenizer(sentences)
            model = build_advanced_model(vocab_size, MAX_SEQUENCE_LEN)
    else:
        print("⚠️ التدريب على CPU (يوصى باستخدام GPU للأداء الأمثل)")
        sentences = load_and_preprocess_data()
        tokenizer, vocab_size = prepare_tokenizer(sentences)
        model = build_advanced_model(vocab_size, MAX_SEQUENCE_LEN)
    
    # تقسيم البيانات
    train_sents, val_sents = train_test_split(
        sentences, test_size=0.1, random_state=42
    )
    
    # إنشاء مولدات البيانات
    train_generator = AdvancedDataGenerator(
        train_sents, tokenizer, MAX_SEQUENCE_LEN, BATCH_SIZE
    )
    
    val_generator = AdvancedDataGenerator(
        val_sents, tokenizer, MAX_SEQUENCE_LEN, BATCH_SIZE, shuffle=False
    )
    
    # إعداد نظام المراقبة
    callbacks = [
        ModelCheckpoint(
            filepath=MODEL_FILE,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            save_weights_only=False,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1,
            profile_batch='10,15'
        )
    ]
    
    # التدريب
    print("\n🔥 بدء تدريب ML-T1 المتقدم (قد يستغرق ساعات إلى أيام)")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        workers=multiprocessing.cpu_count(),
        use_multiprocessing=True
    )
    
    # اختبار النموذج بعد التدريب
    print("\n🎯 تم الانتهاء من التدريب! جاري اختبار النموذج...")
    test_seeds = [
        "العلم هو أساس",
        "التكنولوجيا الحديثة تساهم في",
        "الفلسفة تبحث عن",
        "الذكاء الاصطناعي سوف"
    ]
    
    for seed in test_seeds:
        generated = generate_deep_text(
            seed, 20, model, tokenizer, MAX_SEQUENCE_LEN, temperature=0.8
        )
        print(f"\n🌱 البذرة: {get_display(arabic_reshaper.reshape(seed))}")
        print(f"🧠 إبداع ML-T1:\n{generated}")
        print("─" * 50)
    
    # الواجهة التفاعلية
    print("\n🤖 نموذج ML-T1 جاهز للتجربة! أدخل 'خروج' للخروج.")
    while True:
        seed = input("\n> ").strip()
        if seed.lower() in ["خروج", "exit", "quit"]:
            print("✨ انتهى البرنامج. إلى اللقاء!")
            break
        
        seed = preprocess_text(seed)
        if not seed:
            print("❌ الرجاء إدخال نص عربي صالح.")
            continue
        
        generated = generate_deep_text(
            seed, 15, model, tokenizer, MAX_SEQUENCE_LEN, temperature=0.7
        )
        print(f"\n🧠 إبداع ML-T1:\n{generated}")

if __name__ == "__main__":
    main()
