# -*- coding: utf-8 -*-
"""
نموذج ML-T1-Transformer مع Retriever وتدريب ذاتي
"""

import os
import re
import pickle
import gc
import requests
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras import Model, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Embedding, LayerNormalization, Dense,
    Dropout, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.regularizers import l2
import arabic_reshaper
from bidi.algorithm import get_display

# ——— إعدادات TensorFlow ———
tf.config.optimizer.set_jit(True)
tf.config.run_functions_eagerly(False)

# ——— معلمات عامة ———
MODEL_NAME       = "ML-T1-Transformer"
MODEL_FILE       = MODEL_NAME + ".h5"
TOKENIZER_FILE   = MODEL_NAME + "_tokenizer.pkl"
LOG_DIR          = MODEL_NAME + "_logs"
DATA_DIR         = "dataset"
MAX_SEQUENCE_LEN = 48
EMBEDDING_DIM    = 256
NUM_HEADS        = 4
FF_DIM           = 512
NUM_LAYERS       = 4
BATCH_SIZE       = 32
VOCAB_LIMIT      = 30000

# مراحل التعلم التدريجي
TRAINING_STAGES = [
    {"name": "المرحلة 1: الأساسيات",       "sample": 1000,   "epochs":  5},
    {"name": "المرحلة 2: بناء المفردات",    "sample": 5000,   "epochs": 10},
    {"name": "المرحلة 3: التفكير المنطقي",  "sample":20000,   "epochs": 15},
]

# ——— تحميل وتنظيف البيانات ———
def load_and_preprocess_data():
    print("🚀 تحميل Wikipedia عربي…")
    try:
        ds = load_dataset("wiki40b", "ar", split="train")
        texts = ds["text"]
    except:
        print("⚠️ فشل، استخدام بيانات تجريبية")
        texts = generate_sample_data(10000)
    return [preprocess_text(t) for t in texts if isinstance(t,str)]

def generate_sample_data(n):
    base = [
        "العلم نور والجهل ظلام",
        "التعليم في الصغر كالنقش على الحجر",
        "اللغة العربية هي لغة الضاد",
        "التكنولوجيا تغير العالم سريعاً",
        "الذكاء الاصطناعي مستقبل البشرية",
    ]
    out=[]
    for i in range(n):
        s=base[i % len(base)]
        out.append(s)
    return out

def preprocess_text(text):
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[^\u0600-\u06FF0-9\s،؛.؟!]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    parts = re.split(r'[.،؛؟!]', text)
    return ' '.join(parts[:3]) if len(text.split())>50 else text

# ——— تجهيز Tokenizer ———
def prepare_tokenizer(sentences):
    if os.path.exists(TOKENIZER_FILE):
        with open(TOKENIZER_FILE,"rb") as f: tok=pickle.load(f)
        print("✅ Tokenizer محمّل")
    else:
        tok=Tokenizer(num_words=VOCAB_LIMIT,oov_token="<OOV>",filters='')
        tok.fit_on_texts(sentences)
        with open(TOKENIZER_FILE,"wb") as f: pickle.dump(tok,f)
        print("✅ Tokenizer محفوظ")
    vs=min(len(tok.word_index)+1,VOCAB_LIMIT)
    print("🔤 vocab_size:",vs)
    return tok,vs

# ——— طبقات Transformer ———
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self,embed_dim,heads,ff_dim,rate=0.1):
        super().__init__()
        self.att=MultiHeadAttention(num_heads=heads,key_dim=embed_dim)
        self.ff=tf.keras.Sequential([Dense(ff_dim,activation="gelu"),Dense(embed_dim)])
        self.ln1=LayerNormalization(epsilon=1e-6)
        self.ln2=LayerNormalization(epsilon=1e-6)
        self.do1=Dropout(rate);self.do2=Dropout(rate)
    def call(self,x,training=False):
        a=self.att(x,x)
        a=self.do1(a,training)
        x1=self.ln1(x+a)
        f=self.ff(x1);f=self.do2(f,training)
        return self.ln2(x1+f)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self,maxlen,vocab,dim):
        super().__init__()
        self.tok=Embedding(vocab,dim)
        self.pos=Embedding(maxlen,dim)
    def call(self,x):
        L=tf.shape(x)[-1]
        p=tf.range(L);p=self.pos(p)
        return self.tok(x)+p

# ——— بناء النموذج ———
def build_transformer_model(vocab_size):
    inp=Input((MAX_SEQUENCE_LEN,))
    x=TokenAndPositionEmbedding(MAX_SEQUENCE_LEN,vocab_size,EMBEDDING_DIM)(inp)
    for _ in range(NUM_LAYERS):
        x=TransformerBlock(EMBEDDING_DIM,NUM_HEADS,FF_DIM)(x)
    x=GlobalAveragePooling1D()(x)
    x=Dropout(0.2)(x)
    x=Dense(FF_DIM,activation="gelu",kernel_regularizer=l2(1e-3))(x)
    x=LayerNormalization()(x);x=Dropout(0.3)(x)
    x=Dense(FF_DIM//2,activation="gelu",kernel_regularizer=l2(1e-3))(x)
    x=LayerNormalization()(x)
    out=Dense(vocab_size,activation="softmax")(x)
    model=Model(inp,out)
    opt=Adam(1e-4,beta_1=0.9,beta_2=0.98,epsilon=1e-9,clipnorm=1.0)
    model.compile("sparse_categorical_crossentropy",opt,["accuracy"])
    model.summary()
    return model

# ——— Retriever بسيط (TF-IDF) ———
class Retriever:
    def __init__(self,sentences):
        self.vec=TfidfVectorizer(max_features=VOCAB_LIMIT)
        self.X=self.vec.fit_transform(sentences)
        self.sentences=sentences
    def query(self,q,topk=1,th=0.1):
        v=self.vec.transform([q])
        sims=(self.X @ v.T).toarray().ravel()
        idx=np.argmax(sims)
        return (self.sentences[idx],sims[idx]) if sims[idx]>th else (None,0.)

# ——— جلب صفحة ويكي ومزجها ———
def fetch_wiki(text):
    title=text.replace(" ","_")
    url=f"https://ar.wikipedia.org/wiki/{title}"
    r=requests.get(url,timeout=5)
    if r.status_code==200:
        ps=re.findall(r"<p>(.*?)</p>",r.text,flags=re.S)
        return preprocess_text(" ".join(ps))
    return ""

# ——— توليد النص ———
def generate_advanced_text(seed,nw,model,tokenizer,temp=0.7):
    seed=preprocess_text(seed)
    out=seed
    for _ in range(nw):
        seq=tokenizer.texts_to_sequences([out])[0]
        seq=[t if t<VOCAB_LIMIT else 1 for t in seq]
        seq=pad_sequences([seq],maxlen=MAX_SEQUENCE_LEN,padding="pre")
        p=model.predict(seq,verbose=0)[0]
        p=np.log(p+1e-8)/temp;exp=np.exp(p);probs=exp/exp.sum()
        i=np.random.choice(len(probs),p=probs)
        w=tokenizer.index_word.get(i,"")
        if not w or w=="<OOV>":break
        out+=" "+w
        if w in [".","؟","!"] and np.random.rand()>0.3:break
    return get_display(arabic_reshaper.reshape(out))

# ——— التدريب التدريجي ———
def main():
    # 1) بيانات أولية
    texts=load_and_preprocess_data()
    sentences=[t for t in texts if len(t.split())>2]
    print("✅ جمل متاحة:",len(sentences))
    # 2) Tokenizer أولية
    init_sents=sentences[:5000]
    tokenizer,vocab=prepare_tokenizer(init_sents)
    # 3) نموذج 
    model=build_transformer_model(vocab)
    # 4) Retriever
    retr=Retriever(sentences)
    # 5) Callbacks
    cbs=[
        ModelCheckpoint(MODEL_FILE,save_best_only=True,monitor="val_accuracy",verbose=1),
        EarlyStopping("val_loss",patience=3,restore_best_weights=True,verbose=1),
        ReduceLROnPlateau("val_loss",factor=0.2,patience=2,verbose=1),
        TensorBoard(LOG_DIR)
    ]
    # 6) مراحل
    for st in TRAINING_STAGES:
        print(f"\n🚀 {st['name']} ({st['sample']} جمل، {st['epochs']} عصور)")
        gc.collect()
        samp=sentences[:min(st['sample'],len(sentences))]
        tr,va=train_test_split(samp,test_size=0.1,random_state=42)
        def gen(seq): return AdvancedDataGenerator(seq,tokenizer,BATCH_SIZE)
        history=model.fit(gen(tr),validation_data=gen(va),epochs=st['epochs'],callbacks=cbs,verbose=1)
    # 7) التفاعل والاسترجاع الذكي
    print("\n🤖 ML-T1 جاهز! اكتب 'خروج' للخروج.")
    while True:
        q=input("> ").strip()
        if q.lower() in ["خروج","exit","quit"]: break
        q=preprocess_text(q)
        # أولاً حاول الإجابة من Retriever
        ans,sim=retr.query(q)
        if ans and sim>0.15:
            print("📚 Retrieved:",ans)
        else:
            print("🔍 لم أجد إجابة ملائمة، أجلب من ويكيبيديا...")
            extra=fetch_wiki(q)
            if extra:
                sentences.append(extra)
                retr=Retriever(sentences)
                # تدريب قصير على المثال الجديد
                model.fit(AdvancedDataGenerator([extra],tokenizer,BATCH_SIZE),epochs=1,verbose=0)
                print("✅ تدربت على المعلومة الجديدة.")
            else:
                print("❌ لم أجد شيئاً.")
        # ثم الإكمال الإبداعي
        print("🧠 إكمال:",generate_advanced_text(q,20,model,tokenizer))

if __name__=="__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS']='1'
    os.environ['OMP_NUM_THREADS']=str(os.cpu_count())
    main()
