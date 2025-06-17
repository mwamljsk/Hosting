# -*- coding: utf-8 -*-
import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import arabic_reshaper
from bidi.algorithm import get_display
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LayerNormalization, Dense, Dropout, MultiHeadAttention, GlobalAveragePooling1D

# إعدادات الأداء المتقدم
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # تعطيل معظم سجلات TensorFlow
os.environ['OMP_NUM_THREADS'] = '4'  # استخدام جميع الأنوية
tf.get_logger().setLevel('ERROR')  # تعطيل تحذيرات TensorFlow

# ضبط التوازي للاستفادة من الأنوية الأربعة
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# إعدادات النموذج
MODEL_NAME = "ML-T1-Advanced"
MODEL_FILE = f"{MODEL_NAME}.h5"
TOKENIZER_FILE = f"{MODEL_NAME}_tokenizer.pkl"
MAX_SEQUENCE_LEN = 48
EMBEDDING_DIM = 128
VOCAB_SIZE = 30000  # يجب تعديله حسب حجم المفردات الفعلي

# تعريف فئة PositionalEncoding المطلوبة
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, maxlen, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        super().build(input_shape)
        position = np.arange(self.maxlen)[:, np.newaxis]
        i = np.arange(self.embed_dim)[np.newaxis, :]
        
        # حساب الزوايا بشكل صحيح
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(self.embed_dim))
        angle_rads = position * angle_rates
        
        # تطبيق الجيب على المؤشرات الزوجية وجيب التمام على الفردية
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = np.concatenate([sines, cosines], axis=-1)
        self.pos_encoding = self.pos_encoding[np.newaxis, ...]
        self.pos_encoding = tf.constant(self.pos_encoding, dtype=tf.float32)
        
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    
    def compute_output_shape(self, input_shape):
        return input_shape

# بناء نموذج محسن للأداء
def build_optimized_model():
    # طبقة الإدخال
    inputs = Input(shape=(MAX_SEQUENCE_LEN,), dtype='int32')
    
    # طبقة التضمين
    embedding = Embedding(
        input_dim=VOCAB_SIZE, 
        output_dim=EMBEDDING_DIM,
        name='embedding'
    )(inputs)
    
    # الترميز الموضعي
    positional_encoding = PositionalEncoding(MAX_SEQUENCE_LEN, EMBEDDING_DIM, name='pos_encoding')(embedding)
    x = Dropout(0.3)(positional_encoding)
    
    # طبقات الاهتمام المتوازية
    for i in range(3):  # 3 طبقات transformer (خفيفة)
        # الانتباه المتعدد الرؤوس
        attn_output = MultiHeadAttention(
            num_heads=4,
            key_dim=EMBEDDING_DIM // 4,  # تقسيم أبعاد الرأس
            name=f'mha_{i}'
        )(x, x)
        
        # التطبيع والاتصال المتبقي
        x = LayerNormalization(epsilon=1e-6, name=f'ln1_{i}')(x + attn_output)
        
        # شبكة التغذية الأمامية
        ffn = Dense(EMBEDDING_DIM * 2, activation='gelu', name=f'ffn1_{i}')(x)
        ffn = Dense(EMBEDDING_DIM, name=f'ffn2_{i}')(ffn)
        x = LayerNormalization(epsilon=1e-6, name=f'ln2_{i}')(x + ffn)
    
    # طبقة الإخراج
    x = GlobalAveragePooling1D(name='gap')(x)
    x = Dense(256, activation='gelu', name='dense1')(x)
    outputs = Dense(VOCAB_SIZE, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# تحميل النموذج وال Tokenizer
def load_model_and_tokenizer():
    print("🔍 جاري تحميل النموذج...")
    
    # بناء النموذج أولاً
    model = build_optimized_model()
    
    # تحميل الأوزان
    try:
        model.load_weights(MODEL_FILE)
        print("✅ تم تحميل الأوزان بنجاح")
    except Exception as e:
        print(f"❌ خطأ في تحميل الأوزان: {str(e)}")
        sys.exit(1)
    
    print("🔠 جاري تحميل Tokenizer...")
    try:
        with open(TOKENIZER_FILE, 'rb') as f:
            tokenizer = pickle.load(f)
        print("✅ تم تحميل Tokenizer بنجاح")
        return model, tokenizer
    except Exception as e:
        print(f"❌ خطأ في تحميل Tokenizer: {e}")
        sys.exit(1)

# توليد النص بكفاءة
def generate_text(seed_text, next_words, model, tokenizer, temperature=0.7):
    if not seed_text:
        return ""
    
    output = seed_text
    token_sequence = []
    
    for _ in range(next_words):
        # تحويل النص الحالي إلى تسلسل رموز
        tokens = tokenizer.texts_to_sequences([output])[0]
        tokens = [t for t in tokens if t < tokenizer.num_words] or [1]
        token_sequence = tokens[-MAX_SEQUENCE_LEN:]
        
        # حشو التسلسل
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            [token_sequence], maxlen=MAX_SEQUENCE_LEN, padding='pre'
        )
        
        # التنبؤ
        predictions = model.predict(padded_sequence, verbose=0)[0]
        
        # أخذ عينة من التوزيع
        predictions = np.log(predictions + 1e-10) / max(temperature, 0.1)
        exp_preds = np.exp(predictions)
        probs = exp_preds / np.sum(exp_preds)
        predicted_idx = np.random.choice(len(probs), p=probs)
        predicted_word = tokenizer.index_word.get(predicted_idx, "")
        
        # التوقف عند الكلمات غير المعروفة
        if not predicted_word or predicted_word == "<OOV>":
            break
            
        # إضافة الكلمة الجديدة
        output += " " + predicted_word
        
        # إيقاف ذكي عند نهاية الجملة
        if predicted_word in [".", "؟", "!"]:
            break
            
    # تشكيل النص للعرض
    reshaped_text = arabic_reshaper.reshape(output)
    return get_display(reshaped_text)

# الواجهة التفاعلية المحسنة
def interactive_interface(model, tokenizer):
    print("\n" + "=" * 70)
    print("🤖 مرحباً! أنا ML-T1، المساعد الذكي المتقدم")
    print("مواصفات النظام: 4 نوى | 16GB RAM | 100GB تخزين")
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
                
            # توليد الرد
            response = generate_text(user_input, 30, model, tokenizer, temperature=0.7)
            
            # عرض الرد
            print(f"\n💡 ML-T1:\n{response}")
            
        except KeyboardInterrupt:
            print("\n✨ إلى اللقاء! كان حديثاً ممتعاً.")
            break
        except Exception as e:
            print(f"⚠️ حدث خطأ غير متوقع: {e}")
            print("🔁 جاري إعادة المحاولة...")

# الدالة الرئيسية
if __name__ == "__main__":
    # تحميل النموذج
    model, tokenizer = load_model_and_tokenizer()
    
    # بدء الواجهة التفاعلية
    interactive_interface(model, tokenizer)
