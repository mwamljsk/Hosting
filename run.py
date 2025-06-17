# -*- coding: utf-8 -*-
import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import arabic_reshaper
from bidi.algorithm import get_display
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.models import Model

# إعدادات النموذج
MODEL_NAME = "ML-T1-Advanced"
MODEL_FILE = f"{MODEL_NAME}.h5"
TOKENIZER_FILE = f"{MODEL_NAME}_tokenizer.pkl"
MAX_SEQUENCE_LEN = 48

# تعريف فئة PositionalEncoding المطلوبة
class PositionalEncoding(Layer):
    def __init__(self, maxlen, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.pos_encoding = self.positional_encoding(maxlen, embed_dim)

    def positional_encoding(self, maxlen, embed_dim):
        position = tf.range(maxlen, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(embed_dim, dtype=tf.float32)[tf.newaxis, :]
        
        # حساب الزوايا بشكل صحيح
        angle_rads = position * tf.pow(
            10000.0, 
            - (2 * (i // 2)) / tf.cast(embed_dim, tf.float32)
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

# حل مشكلة طبقة الإدخال
def fix_input_layer(config):
    if 'batch_shape' in config:
        config['batch_input_shape'] = config.pop('batch_shape')
    return config

# تحميل النموذج مع إصلاح المشاكل
def load_model_with_fixes(model_path):
    # تسجيل الفئات المخصصة
    custom_objects = {
        'PositionalEncoding': PositionalEncoding,
        'fix_input_layer': fix_input_layer
    }
    
    # محاولة التحميل مع الإصلاحات
    try:
        return load_model(model_path, custom_objects=custom_objects)
    except:
        # حل بديل إذا فشل التحميل المباشر
        return tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )

# تحميل النموذج وال Tokenizer
def load_model_and_tokenizer():
    print("🔍 جاري تحميل النموذج...")
    try:
        model = load_model_with_fixes(MODEL_FILE)
        print("✅ تم تحميل النموذج بنجاح")
    except Exception as e:
        print(f"❌ خطأ في تحميل النموذج: {str(e)}")
        print("⚡ جاري محاولة بديلة...")
        return load_model_alternative()
    
    print("🔠 جاري تحميل Tokenizer...")
    try:
        with open(TOKENIZER_FILE, 'rb') as f:
            tokenizer = pickle.load(f)
        print("✅ تم تحميل Tokenizer بنجاح")
        return model, tokenizer
    except Exception as e:
        print(f"❌ خطأ في تحميل Tokenizer: {e}")
        sys.exit(1)

# حل بديل لتحميل النموذج
def load_model_alternative():
    print("⚙️ جاري بناء نموذج جديد مع تحميل الأوزان...")
    
    # بناء هيكل النموذج يدوياً (يجب تعديله حسب بنية نموذجك)
    vocab_size = 30000  # يجب تعديل هذا الرقم حسب حجم المفردات الفعلي
    
    inputs = Input(shape=(MAX_SEQUENCE_LEN,))
    # ... (أضف هنا طبقات النموذج كما في كود التدريب الأصلي)
    
    # بعد بناء النموذج، تحميل الأوزان
    model = Model(inputs=inputs, outputs=outputs)
    model.load_weights(MODEL_FILE)
    
    return model

# توليد النص
def generate_text(seed_text, next_words, model, tokenizer, temperature=0.7):
    if not seed_text:
        return ""
    
    output = seed_text
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([output])[0]
        token_list = [t for t in token_list if t < tokenizer.num_words] or [1]
        token_list = token_list[-MAX_SEQUENCE_LEN:]
        
        token_list = tf.keras.preprocessing.sequence.pad_sequences(
            [token_list], maxlen=MAX_SEQUENCE_LEN, padding='pre'
        )
        
        predictions = model.predict(token_list, verbose=0)[0]
        
        # تنويع الإخراج
        predictions = np.log(predictions + 1e-10) / max(temperature, 0.1)
        exp_preds = np.exp(predictions)
        probs = exp_preds / np.sum(exp_preds)
        
        predicted_idx = np.random.choice(len(probs), p=probs)
        predicted_word = tokenizer.index_word.get(predicted_idx, "")
        
        if not predicted_word or predicted_word == "<OOV>":
            break
            
        output += " " + predicted_word
        
        if predicted_word in [".", "؟", "!"]:
            break
            
    # تشكيل النص للعرض
    reshaped_text = arabic_reshaper.reshape(output)
    return get_display(reshaped_text)

# الواجهة التفاعلية
def interactive_interface(model, tokenizer):
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
    # إعدادات متقدمة
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # تقليل السجلات
    tf.get_logger().setLevel('ERROR')
    
    # تحميل النموذج
    model, tokenizer = load_model_and_tokenizer()
    
    # بدء الواجهة التفاعلية
    interactive_interface(model, tokenizer)
