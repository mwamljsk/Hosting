# -*- coding: utf-8 -*-
"""
نموذج ML-T1 الذكي المتقدم - ذكاء اصطناعي عربي ذاتي التعلم
"""

import os
import re
import pickle
import numpy as np
import requests
from bs4 import BeautifulSoup
from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Embedding, LayerNormalization, Dense,
    Dropout, MultiHeadAttention, GlobalAveragePooling1D,
    Bidirectional, LSTM, Attention
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
import gc
import datetime
import json

# تفعيل وضع الأداء المتقدم
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ تم تفعيل تسريع GPU: {len(gpus)} وحدة")
else:
    print("⚠️ لم يتم العثور على GPU، جاري التشغيل على CPU")

tf.config.optimizer.set_jit(True)
tf.config.run_functions_eagerly(True)

os.environ['OMP_NUM_THREADS'] = '4'  # لاستغلال جميع الأنوية
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
os.environ['TF_NUM_INTEROP_THREADS'] = '4'

# 1. إعداد المعلمات الأساسية
MODEL_NAME = "ML-T1-Advanced"
MODEL_FILE = f"{MODEL_NAME}.h5"
TOKENIZER_FILE = f"{MODEL_NAME}_tokenizer.pkl"
KNOWLEDGE_FILE = f"{MODEL_NAME}_knowledge.json"
LOG_DIR = f"{MODEL_NAME}_logs"

# معلمات النموذج الذكي
FF_DIM = 384
NUM_LAYERS = 3
VOCAB_LIMIT = 25000
CONTEXT_SIZE = 3  # عدد الجمل للسياق 

MAX_SEQUENCE_LEN = 48  # تقليل من 64
EMBEDDING_DIM = 128    # تقليل من 192
NUM_HEADS = 4          # تقليل من 6
FF_DIM = 256           # تقليل من 384
BATCH_SIZE = 16        # تقليل من 32


# 2. مراحل التدريب الذكية
TRAINING_STAGES = [
    {"name": "المرحلة 1: التأسيس المعرفي", "sample_size": 5000, "epochs": 8},
    {"name": "المرحلة 2: بناء الفهم السياقي", "sample_size": 20000, "epochs": 12},
    {"name": "المرحلة 3: التفكير المنطقي المتقدم", "sample_size": 50000, "epochs": 15},
    {"name": "المرحلة 4: الإبداع والابتكار", "sample_size": 100000, "epochs": 18},
    {"name": "المرحلة 5: التميز والقيادة", "sample_size": 200000, "epochs": 20}
]

class IntelligentArabicAI:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.knowledge_base = self.load_knowledge()
        self.session_context = []
        
    def load_knowledge(self):
        """تحميل قاعدة المعرفة الذاتية"""
        try:
            if os.path.exists(KNOWLEDGE_FILE):
                with open(KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"❌ خطأ في تحميل قاعدة المعرفة: {e}")
        return {
            "facts": {},
            "qa_pairs": {},
            "last_updated": str(datetime.datetime.now())
        }
    
    def save_knowledge(self):
        """حفظ قاعدة المعرفة الذاتية"""
        try:
            self.knowledge_base["last_updated"] = str(datetime.datetime.now())
            with open(KNOWLEDGE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
            print("💾 تم تحديث قاعدة المعرفة بنجاح")
        except Exception as e:
            print(f"❌ خطأ في حفظ قاعدة المعرفة: {e}")
    
    def search_online(self, query):
        """البحث الذكي على الإنترنت"""
        try:
            print(f"🔍 جاري البحث عن: '{query}'")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            url = f"https://www.google.com/search?q={query}&hl=ar"
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # استخراج النتائج
            results = []
            for g in soup.find_all('div', class_='tF2Cxc'):
                title = g.find('h3').text if g.find('h3') else ""
                snippet = g.find('div', class_='VwiC3b').text if g.find('div', class_='VwiC3b') else ""
                if title and snippet:
                    results.append({"title": title, "content": snippet})
            
            # تخزين النتائج في قاعدة المعرفة
            if results:
                self.knowledge_base["facts"][query] = results[:3]  # حفظ أفضل 3 نتائج
                self.save_knowledge()
            
            return results
        except Exception as e:
            print(f"❌ خطأ في البحث: {e}")
            return []

    def update_context(self, text):
        """تحديث السياق المحادثي"""
        self.session_context.append(text)
        if len(self.session_context) > CONTEXT_SIZE:
            self.session_context.pop(0)

    def get_context(self):
        """الحصول على السياق الحالي"""
        return " ".join(self.session_context)
    
    def understand_query(self, query):
        """فهم نية السؤال باستخدام تقنيات متقدمة"""
        intent_keywords = {
            "تعريف": ["ما هو", "من هو", "ما هي", "تعريف", "مفهوم"],
            "مقارنة": ["مقارنة", "الفرق بين", "مقارنة بين", "أيهما أفضل"],
            "تفسير": ["كيف", "لماذا", "شرح", "تفسير", "سبب"],
            "بحث": ["ابحث عن", "معلومات عن", "أرغب في معرفة", "بحث عن"],
            "رأي": ["رأيك", "ما تظن", "ما رأيك", "توصية", "نصيحة"]
        }
        
        # تحليل النية
        intent = "عام"
        for key, words in intent_keywords.items():
            if any(word in query for word in words):
                intent = key
                break
        
        # تحديد الموضوع
        topic = query
        for word in query.split():
            if word in ["عن", "في", "بخصوص", "حول"]:
                idx = query.index(word)
                topic = query[idx+len(word):].strip()
                break
        
        return {"intent": intent, "topic": topic}
    
    def generate_response(self, query, temperature=0.7):
        """توليد رد ذكي مع فهم السياق"""
        # فهم السؤال
        analysis = self.understand_query(query)
        intent = analysis["intent"]
        topic = analysis["topic"]
        
        # تحديث السياق
        self.update_context(query)
        context = self.get_context()
        
        # البحث في المعرفة الذاتية أولاً
        if topic in self.knowledge_base["facts"]:
            knowledge = self.knowledge_base["facts"][topic][0]["content"]
            prompt = f"السؤال: {query}\nالمعرفة: {knowledge}\nالسياق: {context}\nالرد:"
        else:
            # البحث على الإنترنت إذا لزم الأمر
            if intent in ["تعريف", "بحث"]:
                search_results = self.search_online(topic)
                if search_results:
                    knowledge = search_results[0]["content"]
                    prompt = f"السؤال: {query}\nالمعرفة: {knowledge}\nالسياق: {context}\nالرد:"
                else:
                    prompt = f"السؤال: {query}\nالسياق: {context}\nالرد:"
            else:
                prompt = f"السؤال: {query}\nالسياق: {context}\nالرد:"
        
        # توليد الرد باستخدام النموذج
        response = self.generate_text(prompt, 25, temperature)
        
        # تحسين الرد وفقًا لنوع السؤال
        if intent == "رأي":
            response = "بعد التحليل، أرى أن " + response
        elif intent == "مقارنة":
            response = "عند المقارنة نجد أن " + response
        elif intent == "تفسير":
            response = "التفسير العلمي هو أن " + response
        
        # تحديث السياق بالرد
        self.update_context(response)
        
        return response

# 3. تحميل وتجهيز البيانات
def load_and_preprocess_data():
    """تحميل وتنظيف البيانات الذكية"""
    print("🚀 جاري تحميل البيانات المعرفية...")
    sources = [
        "wiki40b/ar",
        "arabic_billion_words",
        "oscar-arabic"
    ]
    
    all_texts = []
    for source in sources:
        try:
            print(f"🔍 جاري تحميل {source}...")
            dataset = load_dataset(source, split="train", streaming=True)
            for item in dataset.take(20000):
                if 'text' in item:
                    all_texts.append(item['text'])
                elif 'content' in item:
                    all_texts.append(item['content'])
            print(f"✅ تم تحميل {len(all_texts)} نصًا من {source}")
        except Exception as e:
            print(f"❌ خطأ في تحميل {source}: {e}")
    
    if not all_texts:
        print("⚠️ فشل في تحميل البيانات، جاري استخدام بيانات تجريبية")
        return generate_sample_data(20000)
    
    return all_texts

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
            f"في العصر الحديث، {base}",
            f"أثبتت الدراسات أن {base}",
            f"من وجهة نظر علمية، {base}",
            f"يعتقد الخبراء أن {base}"
        ]
        texts.append(variations[i % len(variations)])
    return texts

def preprocess_text(text: str) -> str:
    """تنظيف النص العربي المتقدم"""
    if not isinstance(text, str):
        return ""
    
    # إزالة التشكيل والأحرف الخاصة
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[^\u0600-\u06FF0-9\s،؛:.,؟!()\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # تقطيع الجمل الطويلة
    if len(text.split()) > 80:
        parts = re.split(r'[.،؛؟!]', text)
        text = ' '.join(parts[:4])
    
    return text

# 4. تجهيز Tokenizer
def prepare_tokenizer(sentences):
    """إنشاء Tokenizer متقدم"""
    print("🔄 جاري إنشاء Tokenizer ذكي...")
    tokenizer = Tokenizer(
        num_words=VOCAB_LIMIT, 
        oov_token="<OOV>",
        filters='',
        lower=False
    )
    tokenizer.fit_on_texts(sentences)
    
    # حفظ Tokenizer
    with open(TOKENIZER_FILE, "wb") as f:
        pickle.dump(tokenizer, f)
    
    vocab_size = min(len(tokenizer.word_index) + 1, VOCAB_LIMIT)
    print(f"🔤 حجم القاموس الذكي: {vocab_size}")
    return tokenizer, vocab_size

# 5. طبقات Transformer المتقدمة
class ContextAwareTransformer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(ContextAwareTransformer, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.context_aware = Bidirectional(LSTM(embed_dim//2, return_sequences=True))

    def call(self, inputs, training=False):
        # دمج السياق
        contextual = self.context_aware(inputs)
        
        attn_output = self.att(contextual, contextual)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(maxlen, embed_dim)

    def get_angles(self, position, i, embed_dim):
        angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(embed_dim, tf.float32))
        return position * angles

    def positional_encoding(self, maxlen, embed_dim):
        angle_rads = self.get_angles(
            position=tf.range(maxlen, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(embed_dim, dtype=tf.float32)[tf.newaxis, :],
            embed_dim=embed_dim
        )
        
        # تطبيق الجيب على المؤشرات الزوجية وجيب التمام على الفردية
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

# 6. بناء نموذج ذكي
def build_intelligent_model(vocab_size):
    """بناء نموذج ذكي قادر على فهم السياق"""
    inputs = Input(shape=(MAX_SEQUENCE_LEN,))
    
    # التضمين
    embedding = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM)(inputs)
    
    # الترميز الموضعي
    positional_encoding = PositionalEncoding(MAX_SEQUENCE_LEN, EMBEDDING_DIM)(embedding)
    x = Dropout(0.3)(positional_encoding)
    
    # طبقات Transformer مع الوعي السياقي
    for _ in range(NUM_LAYERS):
        transformer_block = ContextAwareTransformer(EMBEDDING_DIM, NUM_HEADS, FF_DIM)
        x = transformer_block(x, training=False)
    
    # آلية الانتباه
    context_vector = Attention()([x, x])
    
    # تجميع السياق
    x = GlobalAveragePooling1D()(context_vector)
    x = Dropout(0.2)(x)
    
    # طبقات الفهم العميق
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

# 7. مولد البيانات الذكي
class IntelligentDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, sentences, tokenizer, batch_size, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.sentences = sentences
        self.tokenizer = tokenizer
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
            tokens = [t if t < self.vocab_size else 1 for t in tokens]
            
            # إنشاء عينات فعالة
            if len(tokens) > 1:
                # أخذ جزء عشوائي من الجملة
                if len(tokens) > MAX_SEQUENCE_LEN + 1:
                    start_idx = np.random.randint(0, len(tokens) - MAX_SEQUENCE_LEN - 1)
                    seq = tokens[start_idx:start_idx + MAX_SEQUENCE_LEN + 1]
                else:
                    seq = tokens[:MAX_SEQUENCE_LEN + 1]
                
                # حشو التسلسل
                padded_seq = pad_sequences([seq], maxlen=MAX_SEQUENCE_LEN + 1, padding='pre')[0]
                X.append(padded_seq[:-1])
                y.append(padded_seq[-1])
        
        if len(X) == 0:
            return np.zeros((1, MAX_SEQUENCE_LEN)), np.zeros((1,))
        
        return np.array(X), np.array(y)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.sentences)

# 8. نظام التوليد الذكي
def generate_intelligent_text(seed_text, next_words, model, tokenizer, temperature=0.7):
    """توليد نص ذكي مع فهم السياق"""
    if not seed_text:
        return ""
    
    output = seed_text
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([output])[0]
        token_list = [t if t < VOCAB_LIMIT else 1 for t in token_list]
        
        if not token_list:
            token_list = [1]
            
        token_list = token_list[-MAX_SEQUENCE_LEN:]
        token_list = pad_sequences([token_list], maxlen=MAX_SEQUENCE_LEN, padding='pre')
        
        predictions = model.predict(token_list, verbose=0)[0]
        
        # تنويع الإخراج
        predictions = np.log(predictions + 1e-10) / max(temperature, 0.1)
        exp_preds = np.exp(predictions)
        probs = exp_preds / np.sum(exp_preds)
        
        predicted_idx = np.random.choice(len(probs), p=probs)
        predicted_word = tokenizer.index_word.get(predicted_idx, "")
        
        # إيقاف ذكي
        if not predicted_word or predicted_word == "<OOV>":
            break
            
        output += " " + predicted_word
        
        # إيقاف عند نهاية الجملة المنطقية
        if predicted_word in [".", "؟", "!"] and np.random.random() > 0.25:
            break
        if len(output.split()) > next_words * 1.5:
            break
            
    # تشكيل النص للعرض
    reshaped_text = arabic_reshaper.reshape(output)
    bidi_text = get_display(reshaped_text)
    return bidi_text

# 9. نظام التدريب الذاتي
def self_learning_loop(ai, texts):
    """نظام التدريب الذاتي المستمر"""
    print("\n🚀 بدء عملية التعلم الذاتي...")
    
    # تجهيز البيانات
    sentences = []
    for text in texts:
        cleaned = preprocess_text(text)
        if cleaned and len(cleaned.split()) > 4:
            sentences.append(cleaned)
    
    print(f"📊 عدد الجمل المدربة: {len(sentences)}")
    
    # إنشاء Tokenizer
    tokenizer, vocab_size = prepare_tokenizer(sentences)
    ai.tokenizer = tokenizer
    
    # بناء النموذج
    ai.model = build_intelligent_model(vocab_size)
    
    # نظام المراقبة
    callbacks = [
        ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        TensorBoard(log_dir=LOG_DIR)
    ]
    
    # التدريب على مراحل متعددة
    for stage in TRAINING_STAGES:
        print(f"\n{'='*70}")
        print(f"🧠 {stage['name']}")
        print(f"📈 حجم العينة: {stage['sample_size']} جملة")
        print(f"⏱️ العصور: {stage['epochs']}")
        print(f"{'='*70}")
        
        # اختيار عينة عشوائية
        sample_size = min(stage['sample_size'], len(sentences))
        stage_sentences = np.random.choice(sentences, size=sample_size, replace=False)
        
        # تقسيم البيانات
        train_sents, val_sents = train_test_split(stage_sentences, test_size=0.1, random_state=42)
        
        # إنشاء مولدات البيانات
        train_gen = IntelligentDataGenerator(train_sents, tokenizer, BATCH_SIZE)
        val_gen = IntelligentDataGenerator(val_sents, tokenizer, BATCH_SIZE, shuffle=False)
        
        # التدريب
        history = ai.model.fit(
            train_gen,
            epochs=stage['epochs'],
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # اختبار النموذج
        print("\n🔍 اختبار الفهم السياقي:")
        test_queries = [
            "ما هو الذكاء الاصطناعي؟",
            "كيف يمكن للتعلم الآلي أن يفيد العالم العربي؟",
            "ما الفرق بين الذكاء الاصطناعي والتعلم العميق؟",
            "كيف أبدأ في تعلم الذكاء الاصطناعي؟"
        ]
        
        for query in test_queries:
            response = ai.generate_response(query)
            reshaped_query = arabic_reshaper.reshape(query)
            print(f"\n❓ السؤال: {get_display(reshaped_query)}")
            print(f"💡 الإجابة الذكية:\n{response}")
            print("─" * 70)
    
    print("✅ اكتمل التدريب بنجاح!")
    return ai

# 10. الواجهة التفاعلية الذكية
def intelligent_interface(ai):
    """واجهة محادثة ذكية"""
    print("\n" + "=" * 70)
    print("🤖 مرحباً! أنا ML-T1، المساعد الذكي المتقدم")
    print("يمكنك سؤالي عن أي موضوع، أو كتابة 'خروج' للإنهاء")
    print("=" * 70)
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ["خروج", "exit", "quit"]:
                print("✨ إلى اللقاء! كان حديثاً ممتعاً.")
                break
                
            if not user_input:
                print("❓ يبدو أنك لم تدخل أي نص، هل يمكنك إعادة صياغة سؤالك؟")
                continue
                
            # معالجة الأوامر الخاصة
            if user_input.startswith("!تعلم"):
                topic = user_input[5:].strip()
                if topic:
                    print(f"🔍 جاري البحث وتعلم المزيد عن: {topic}")
                    ai.search_online(topic)
                    print("✅ تم تحديث المعرفة بنجاح")
                else:
                    print("❌ الرجاء تحديد موضوع للتعلم")
                continue
                
            # توليد الرد الذكي
            response = ai.generate_response(user_input)
            
            # عرض الرد بشكل جميل
            reshaped_response = arabic_reshaper.reshape(response)
            bidi_response = get_display(reshaped_response)
            print(f"\n💡 ML-T1:\n{bidi_response}")
            
        except Exception as e:
            print(f"⚠️ حدث خطأ غير متوقع: {e}")
            print("🔁 جاري إعادة التشغيل...")
            ai.session_context = []

# 11. الدالة الرئيسية
def main():
    # إنشاء الذكاء الاصطناعي
    ai = IntelligentArabicAI()
    
    # تحميل البيانات
    texts = load_and_preprocess_data()
    
    # التدريب الذاتي
    if not os.path.exists(MODEL_FILE):
        ai = self_learning_loop(ai, texts)
    else:
        print("🔍 جاري تحميل النموذج المدرب مسبقاً...")
        ai.model = load_model(MODEL_FILE, custom_objects={
            'ContextAwareTransformer': ContextAwareTransformer,
            'PositionalEncoding': PositionalEncoding
        })
        with open(TOKENIZER_FILE, 'rb') as f:
            ai.tokenizer = pickle.load(f)
        print("✅ تم تحميل النموذج بنجاح")
    
    # الواجهة التفاعلية
    intelligent_interface(ai)

if __name__ == "__main__":
    # إعدادات متقدمة
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() or 4)
    
    # بدء النظام
    main()
