# -*- coding: utf-8 -*-
import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import arabic_reshaper
from bidi.algorithm import get_display
from tensorflow.keras.models import load_model

# إعدادات النموذج
MODEL_NAME = "ML-T1-Advanced"
MODEL_FILE = f"{MODEL_NAME}.h5"
TOKENIZER_FILE = f"{MODEL_NAME}_tokenizer.pkl"
MAX_SEQUENCE_LEN = 48

# تحميل النموذج وال Tokenizer
def load_model_and_tokenizer():
    print("🔍 جاري تحميل النموذج...")
    model = load_model(MODEL_FILE)
    print("✅ تم تحميل النموذج بنجاح")
    
    print("🔠 جاري تحميل Tokenizer...")
    with open(TOKENIZER_FILE, 'rb') as f:
        tokenizer = pickle.load(f)
    print("✅ تم تحميل Tokenizer بنجاح")
    
    return model, tokenizer

# توليد النص
def generate_text(seed_text, next_words, model, tokenizer, temperature=0.7):
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
            
        except Exception as e:
            print(f"⚠️ حدث خطأ غير متوقع: {e}")
            print("🔁 جاري إعادة التشغيل...")

# الدالة الرئيسية
if __name__ == "__main__":
    # إعدادات متقدمة
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # تحميل النموذج
    model, tokenizer = load_model_and_tokenizer()
    
    # بدء الواجهة التفاعلية
    interactive_interface(model, tokenizer)
