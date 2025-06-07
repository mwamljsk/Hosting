# -*- coding: utf-8 -*-
"""
نموذج ML-T1 الذكي المتقدم - بنية Transformer متقدمة
"""

import os
import re
import pickle
import numpy as np
from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Embedding, LayerNormalization, Dense,
    Dropout, MultiHeadAttention, GlobalAveragePooling1D,
    Bidirectional, LSTM
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, 
    ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import arabic_reshaper
from bidi.algorithm import get_display
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow_text as tf_text

# تفعيل وضع الأداء العالي
tf.config.optimizer.set_jit(True)
tf.config.run_functions_eagerly(True)

# 1. إعداد المعلمات الأساسية
MODEL_NAME = "ML-T1-Transformer"
MODEL_FILE = f"{MODEL_NAME}.h5"
TOKENIZER_FILE = f"{MODEL_NAME}_tokenizer.pkl"
LOG_DIR = f"{MODEL_NAME}_logs"

# معلمات النموذج المتقدم
MAX_SEQUENCE_LEN = 64  # زيادة طول السياق
EMBEDDING_DIM = 512    # زيادة أبعاد التضمين
NUM_HEADS = 8           # عدد رؤوس الاهتمام
FF_DIM = 1024           # وحدات الشبكة العصبية الأمامية
NUM_LAYERS = 8          # عدد طبقات الـ Transformer
BATCH_SIZE = 64         # حجم الدُفعة
EPOCHS = 50             # عدد العصور
VOCAB_LIMIT = 50000     # الحد الأقصى للمفردات
SAMPLE_SIZE = 100000    # زيادة عدد الجمل للتدريب

# 2. تحميل وتجهيز البيانات
def load_and_preprocess_data():
    """تحميل وتنظيف البيانات"""
    print("🚀 جاري تحميل بيانات ويكيبيديا العربية...")
    try:
        dataset = load_dataset("wiki40b", "ar", split="train")
        texts = dataset["text"][:100000]  # 100,000 مقالة
        print("✅ تم تحميل بيانات ويكيبيديا العربية")
        return texts
    except Exception as e:
        print(f"❌ خطأ في تحميل البيانات: {e}")
        print("⚡ جاري تحميل بيانات بديلة...")
        try:
            dataset = load_dataset("arabic_billion_words", split="train")
            texts = dataset["text"][:100000]
            return texts
        except:
            print("⚠️ فشل في تحميل البيانات، جاري استخدام بيانات تجريبية")
            return generate_sample_data(10000)

def generate_sample_data(num_samples):
    """إنشاء بيانات تجريبية غنية"""
    base_texts = [
        "العلم نور والجهل ظلام، فاحرص على طلب العلم دائماً",
        "التعليم في الصغر كالنقش على الحجر، لذلك يجب الاهتمام بالتعليم المبكر",
        "اللغة العربية هي لغة الضاد، وهي من أقدم اللغات السامية",
        "التكنولوجيا الحديثة تساهم في تطور المجتمعات وازدهار الاقتصاد العالمي",
        "الذكاء الاصطناعي يمثل مستقبل البشرية، وسيغير وجه العالم في العقود القادمة",
        "القراءة توسع آفاق الإنسان وتزيد من معرفته وتطور مهاراته الفكرية",
        "التاريخ يعلمنا دروساً قيمة عن صعود وسقوط الحضارات والأمم",
        "الفلسفة تساعدنا على فهم الوجود والإجابة على الأسئلة الوجودية الكبرى",
        "الرياضيات هي لغة الكون، وهي أساس جميع العلوم الطبيعية والتطبيقية",
        "الإبداع والابتكار هما محركات التقدم البشري في جميع المجالات"
    ]
    
    # توليد بيانات متنوعة
    texts = []
    for i in range(num_samples):
        base = base_texts[i % len(base_texts)]
        variations = [
            f"في مجال {base.split()[0]}، {base}",
            f"من المعروف أن {base}",
            f"تعتبر {base}",
            f"بلا شك، {base}",
            f"في العصر الحديث، {base}"
        ]
        texts.append(variations[i % len(variations)])
    return texts

def preprocess_text(text: str) -> str:
    """تنظيف النص العربي المتقدم"""
    if not isinstance(text, str):
        return ""
    
    # إزالة التشكيل والأحرف الخاصة
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)  # إزالة التشكيل
    text = re.sub(r'[^\u0600-\u06FF0-9\s،؛.؟!]', ' ', text)  # إبقاء العربية والأرقام وعلامات الترقيم
    text = re.sub(r'\s+', ' ', text).strip()  # إزالة المسافات الزائدة
    
    # تقطيع الجمل الطويلة
    if len(text.split()) > 50:
        parts = re.split(r'[.،؛؟!]', text)
        text = ' '.join(parts[:3])
    
    return text

# 3. تجهيز Tokenizer
def prepare_tokenizer(sentences):
    """إنشاء Tokenizer متقدم"""
    print("🔄 جاري إنشاء Tokenizer جديد...")
    tokenizer = Tokenizer(
        num_words=VOCAB_LIMIT, 
        oov_token="<OOV>",
        filters=''
    )
    tokenizer.fit_on_texts(sentences)
    
    # حفظ Tokenizer
    with open(TOKENIZER_FILE, "wb") as f:
        pickle.dump(tokenizer, f)
    
    vocab_size = min(len(tokenizer.word_index) + 1, VOCAB_LIMIT)
    print(f"🔤 حجم القاموس: {vocab_size}")
    return tokenizer, vocab_size

# 4. طبقات Transformer المتقدمة
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# 5. بناء نموذج Transformer متقدم
def build_transformer_model(vocab_size, max_len):
    """بناء نموذج Transformer متقدم"""
    inputs = Input(shape=(max_len,))
    embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, EMBEDDING_DIM)
    x = embedding_layer(inputs)
    
    # طبقات Transformer
    for _ in range(NUM_LAYERS):
        x = TransformerBlock(EMBEDDING_DIM, NUM_HEADS, FF_DIM)(x)
    
    # طبقة تجميع نهائية
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    
    # طبقات كثيفة متقدمة
    x = Dense(FF_DIM, activation="gelu", kernel_regularizer=l2(0.001))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(FF_DIM // 2, activation="gelu", kernel_regularizer=l2(0.001))(x)
    x = LayerNormalization()(x)
    
    # طبقة الإخراج
    outputs = Dense(vocab_size, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # المترجم المتقدم
    optimizer = Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9,
        clipnorm=1.0
    )
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    
    model.summary()
    return model

# 6. مولد البيانات المتقدم
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
            tokens = [t if t < self.vocab_size else 1 for t in tokens]  # 1 = <OOV>
            
            # إنشاء عينات متعددة من الجملة
            for i in range(1, len(tokens)):
                start_idx = max(0, i - self.max_seq_len)
                seq = tokens[start_idx:i+1]
                
                if len(seq) < 2:
                    continue
                    
                padded_seq = pad_sequences([seq], maxlen=self.max_seq_len+1, padding='pre')[0]
                X.append(padded_seq[:-1])
                y.append(padded_seq[-1])
        
        if len(X) == 0:
            return np.zeros((1, self.max_seq_len)), np.zeros((1,))
        
        return np.array(X), np.array(y)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.sentences)

# 7. نظام توليد النص المتقدم
def generate_advanced_text(seed_text, next_words, model, tokenizer, max_seq_len, temperature=0.7):
    """توليد نص متقدم باستخدام تقنيات متطورة"""
    if not seed_text:
        return ""
    
    output = seed_text
    reshaped_output = arabic_reshaper.reshape(seed_text)
    bidi_output = get_display(reshaped_output)
    print(f"🌱 البذرة: {bidi_output}")
    
    for _ in range(next_words):
        # تحضير التسلسل
        token_list = tokenizer.texts_to_sequences([output])[0]
        token_list = [t if t < VOCAB_LIMIT else 1 for t in token_list]
        
        if not token_list:
            token_list = [1]  # <OOV>
            
        # اقتصار التسلسل على الطول المسموح
        token_list = token_list[-max_seq_len:]
        token_list = pad_sequences([token_list], maxlen=max_seq_len, padding='pre')
        
        # التنبؤ
        predictions = model.predict(token_list, verbose=0)[0]
        
        # التحكم في الإبداع
        predictions = np.log(predictions) / max(temperature, 0.1)
        exp_preds = np.exp(predictions)
        probs = exp_preds / np.sum(exp_preds)
        
        # أخذ عينة مرجحة
        predicted_idx = np.random.choice(len(probs), p=probs)
        predicted_word = tokenizer.index_word.get(predicted_idx, "")
        
        # التوقف عند الكلمات غير المعروفة أو نهاية الجملة
        if not predicted_word or predicted_word == "<OOV>":
            break
            
        output += " " + predicted_word
        
        # إيقاف ذكي عند نهاية الجملة
        if predicted_word in [".", "؟", "!"] and np.random.random() > 0.3:
            break
            
    # تشكيل النص للعرض
    reshaped_text = arabic_reshaper.reshape(output)
    bidi_text = get_display(reshaped_text)
    return bidi_text

# 8. التدريب الرئيسي
def main():
    # تحميل وتجهيز البيانات
    print("🔥 إعداد أقوى نموذج للغة العربية")
    texts = load_and_preprocess_data()
    
    # تنظيف النصوص
    sentences = []
    for text in texts:
        cleaned = preprocess_text(text)
        if cleaned and len(cleaned.split()) > 2:  # تجاهل الجمل القصيرة جداً
            sentences.append(cleaned)
    
    # استخدام العدد المطلوب من الجمل
    sentences = sentences[:SAMPLE_SIZE]
    print(f"✅ عدد الجمل المستخدمة: {len(sentences)}")
    
    # إنشاء Tokenizer
    tokenizer, vocab_size = prepare_tokenizer(sentences)
    
    # بناء النموذج
    model = build_transformer_model(vocab_size, MAX_SEQUENCE_LEN)
    
    # تقسيم البيانات
    train_sents, val_sents = train_test_split(sentences, test_size=0.1, random_state=42)
    
    # إنشاء مولدات البيانات
    train_generator = AdvancedDataGenerator(train_sents, tokenizer, MAX_SEQUENCE_LEN, BATCH_SIZE)
    val_generator = AdvancedDataGenerator(val_sents, tokenizer, MAX_SEQUENCE_LEN, BATCH_SIZE, shuffle=False)
    
    # إعداد نظام المراقبة
    callbacks = [
        ModelCheckpoint(
            MODEL_FILE, 
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
            patience=2,
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
    print("\n🚀 بدء تدريب النموذج المتقدم (قد يستغرق وقتاً طويلاً)")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # اختبار النموذج
    print("\n🎯 اختبار النموذج بعد التدريب:")
    test_seeds = [
        "العلم هو أساس",
        "التكنولوجيا الحديثة",
        "الذكاء الاصطناعي",
        "مستقبل التعليم",
        "الثورة الصناعية الرابعة"
    ]
    
    for seed in test_seeds:
        generated = generate_advanced_text(seed, 20, model, tokenizer, MAX_SEQUENCE_LEN, temperature=0.8)
        reshaped_seed = arabic_reshaper.reshape(seed)
        print(f"\n🌱 البذرة: {get_display(reshaped_seed)}")
        print(f"🧠 إبداع ML-T1:\n{generated}")
        print("─" * 70)
    
    # الواجهة التفاعلية المتقدمة
    print("\n🤖 نموذج ML-T1 جاهز للتجربة! أدخل 'خروج' للإنهاء")
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ["خروج", "exit", "quit"]:
            print("✨ انتهى البرنامج. إلى اللقاء!")
            break
        
        user_input = preprocess_text(user_input)
        if not user_input or len(user_input.split()) < 2:
            print("❌ الرجاء إدخال نص عربي صالح (كلمتين على الأقل)")
            continue
        
        generated = generate_advanced_text(
            user_input, 
            25, 
            model, 
            tokenizer, 
            MAX_SEQUENCE_LEN,
            temperature=0.7
        )
        print(f"\n🧠 إبداع ML-T1:\n{generated}")

if __name__ == "__main__":
    # إعدادات لتحسين الأداء
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
    
    # تشغيل التدريب
    main()
