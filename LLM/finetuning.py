"""finetuning.ipynb

"""

#!pip install -q transformers datasets peft accelerate --upgrade
#!pip install -q bitsandbytes --upgrade

import torch
# torch: PyTorch'un kendisi. Tensörler, GPU/CPU yönetimi ve otomatik türev (autograd) bu kütüphaneden gelir.

import numpy as np
# numpy: Sayısal işlemler ve rastgelelik (seed) ayarları için.

import random
# random: Python'un yerleşik rastgelelik modülü; seed sabitlemek için kullanacağız.

from datasets import Dataset
# datasets.Dataset: Hugging Face'in veri seti sınıfı. Listeden/sözlükten kolayca veri seti oluşturmamızı sağlar
# ve map() fonksiyonu ile toplu (batched) tokenization yapmamıza imkan verir.

from transformers import (
    AutoTokenizer,        # Model adına göre doğru tokenizer'ı otomatik olarak yükler.
    AutoModelForCausalLM, # Model adına göre "causal language modeling" (sıradaki token'ı tahmin etme) mimarisini yükler.
    TrainingArguments,    # Eğitimle ilgili tüm hiperparametreleri (epoch, batch size, learning rate vb.) tutan sınıf.
    Trainer,              # Eğitim döngüsünü (forward, backward, optimizer step, loglama, checkpoint kaydetme) yöneten yüksek seviyeli sınıf.
    DataCollatorForLanguageModeling,  # Bir batch'teki örnekleri aynı uzunluğa getirip (padding) modele uygun formatta birleştiren yardımcı sınıf.
)

from peft import (
    LoraConfig,         # LoRA'nın hiperparametrelerini (r, alpha, hedef katmanlar vb.) tanımlayan konfigürasyon sınıfı.
    get_peft_model,      # Normal bir Hugging Face modelini alıp, üzerine LoRA katmanları ekleyen ("sarmalayan") fonksiyon.
    TaskType,            # PEFT'e bu modelin hangi görev için kullanılacağını bildiren enum (örn. CAUSAL_LM).
    PeftModel,           # Daha sonra kaydedilmiş bir LoRA adaptörünü, temel modelin üzerine tekrar yüklemek için kullanılan sınıf.
)

import matplotlib.pyplot as plt
# matplotlib: Eğitim sırasında kaydedilen loss değerlerini grafikle görselleştirmek için.

SEED = 42
# SEED: Sabit bir sayı seçiyoruz. Aynı seed, aynı rastgele sayı dizisini üretir -> tekrar üretilebilirlik sağlar.

random.seed(SEED)        # Python'un random modülünün rastgeleliğini sabitle.
np.random.seed(SEED)     # NumPy'ın rastgeleliğini sabitle.
torch.manual_seed(SEED)  # PyTorch CPU işlemlerinin rastgeleliğini sabitle.
torch.cuda.manual_seed_all(SEED)  # Eğer GPU varsa, tüm GPU'lardaki rastgeleliği de sabitle.

device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.cuda.is_available(): Sistemde CUDA destekli bir GPU olup olmadığını kontrol eder.
# Varsa "cuda" (GPU), yoksa "cpu" stringini device değişkenine atarız.

print(f"Kullanılacak cihaz: {device}")

import pandas as pd

df = pd.read_csv("Bitext_Customer_Support_Instruction_Tuning.csv")

# Mevcut sütun isimlerini ekrana yazdırır
print("Dosyadaki Sütunlar:", df.columns.tolist())

# Eğer Kaggle'daki güncel versiyonda sütun isimleri 'instruction' ve 'response' ise:
# (Hata almamak için dosyadaki gerçek isimlere göre buraları düzenleyebilirsin)
input_sutunu = (
    "instruction" if "instruction" in df.columns else df.columns[0]
)
output_sutunu = "response" if "response" in df.columns else df.columns[1]

df_temiz = pd.DataFrame(
    {"instruction": df[input_sutunu], "output": df[output_sutunu]}
)

raw_examples = df_temiz.to_dict(orient="records")

print(f"Toplam örnek sayısı: {len(raw_examples)}")



def format_example(example):
    # Bu fonksiyon, her bir (instruction, output) çiftini, modele tek bir metin (prompt) olarak verilecek
    # standart bir şablona dönüştürür. Bu şablona "prompt template" denir.
    text = (
        f"### Talimat:\n{example['instruction']}\n\n"   # Talimat (kullanıcının sorusu/isteği) başlığı altında yazılır.
        f"### Yanıt:\n{example['output']}"                # Modelin üretmesi gereken yanıt, "### Yanıt:" başlığı altında yazılır.
    )
    return {"text": text}
    # Fonksiyon, "text" anahtarına sahip bir sözlük döndürür; bu, Dataset.map() ile uyumlu çalışmasını sağlar.

dataset = Dataset.from_list(raw_examples)
# Dataset.from_list: Python listesindeki sözlükleri alıp bir Hugging Face Dataset nesnesine çevirir.
# Bu nesne, .map(), .shuffle(), .train_test_split() gibi kullanışlı metotlar sunar.

dataset = dataset.map(format_example)
# .map(): Veri setindeki HER örneğe format_example fonksiyonunu uygular ve sonuçları yeni sütun(lar) olarak ekler.
# Sonuçta her örnekte artık bir de "text" alanı bulunacak (orijinal "instruction" ve "output" alanları da kalır).

print(dataset[0]["text"])
# İlk örneğin nihai metin halini ekrana yazdırarak şablonun doğru çalıştığını doğruluyoruz.

MODEL_NAME = "distilgpt2"
# MODEL_NAME: Hugging Face Hub'daki model kimliği. "distilgpt2", GPT-2'nin küçültülmüş halidir (~82M parametre).

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# AutoTokenizer.from_pretrained: Belirtilen model için doğru tokenizer'ı (kelime/alt-kelime bölme mantığını) indirir ve yükler.
# Tokenizer, ham metni modelin anlayabileceği sayısal token ID'lerine çevirir.

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# GPT-2 ailesi modellerde varsayılan olarak bir "pad_token" (doldurma token'ı) tanımlı değildir.
# Batch içindeki metinleri aynı uzunluğa getirmek (padding) için bir pad_token gereklidir.
# Burada, var olan "eos_token" (end-of-sequence, dizinin sonu token'ı) pad_token olarak yeniden kullanılır.

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
# AutoModelForCausalLM.from_pretrained: Önceden eğitilmiş (pretrained) ağırlıklarla modeli indirir ve belleğe yükler.
# "CausalLM" = modelin görevi, önceki token'lara bakarak SIRADAKİ token'ı tahmin etmektir (GPT tarzı modellerin temel görevi).

model.to(device)
# Modeli daha önce belirlediğimiz cihaza (GPU varsa "cuda", yoksa "cpu") taşıyoruz.

print(f"Model parametre sayısı: {model.num_parameters():,}")

MAX_LENGTH = 128
# MAX_LENGTH: Her örneğin maksimum token uzunluğu. Bundan uzun metinler kesilir (truncation),
# kısa metinler ise pad_token ile bu uzunluğa tamamlanır (padding).
# Gerçek projelerde bu değer veri setinizin tipik uzunluğuna göre (örn. 512, 1024, 2048) ayarlanır.

def tokenize_function(examples):
    # examples: dataset.map(batched=True) kullanıldığı için, burada TEK bir örnek değil,
    # bir LISTE halinde birden çok örneğin "text" alanı gelir (örn. examples["text"] bir string listesidir).
    tokenized = tokenizer(
        examples["text"],          # Tokenize edilecek ham metin listesi.
        truncation=True,           # MAX_LENGTH'ten uzun metinleri kes.
        max_length=MAX_LENGTH,     # Maksimum token uzunluğu.
        padding="max_length",      # Tüm örnekleri MAX_LENGTH'e tamamla (sabit uzunluk -> batching için kolaylık).
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    # "labels": Modelin tahmin etmesi GEREKEN doğru token ID'leri.
    # Causal language modeling'de etiketler, girdinin (input_ids) bir kopyasıdır;
    # Trainer/model içeride otomatik olarak "bir token kaydırma" (shift) işlemini yapar,
    # böylece model t. pozisyondaki token'dan, t+1. pozisyondaki token'ı tahmin etmeyi öğrenir.
    return tokenized

tokenized_dataset = dataset.map(
    tokenize_function,    # Az önce tanımladığımız tokenize fonksiyonu.
    batched=True,         # Performans için örnekleri tek tek değil, toplu (batch) halinde işler.
    remove_columns=dataset.column_names,
    # Artık ihtiyacımız olmayan orijinal sütunları (instruction, output, text) kaldırıyoruz;
    # geriye sadece model için gereken input_ids, attention_mask ve labels kalır.
)

print(tokenized_dataset)
# Tokenize edilmiş veri setinin yapısını (sütunlar, örnek sayısı) ekrana yazdırarak kontrol ediyoruz.

#pip install -U torchao

lora_config = LoraConfig(
    r=8,                        # Rank: LoRA matrislerinin iç boyutu. Düşük rank, az parametre demektir.
    lora_alpha=16,               # Ölçeklendirme katsayısı (genelde r'nin 2 katı tercih edilir).
    target_modules=["c_attn"],   # GPT-2 mimarisinde, query/key/value projeksiyonlarını içeren birleşik katmanın adı.
                                  # Farklı mimarilerde (Llama, Mistral) bu isim değişir (örn. ["q_proj", "v_proj"]).
    lora_dropout=0.05,            # LoRA katmanlarına %5 dropout uygulanır -> aşırı öğrenmeye karşı küçük bir önlem.
    bias="none",                  # Bias terimleri eğitilmeyecek (en yaygın ve hafif tercih).
    task_type=TaskType.CAUSAL_LM, # Bu, bir "causal language modeling" (GPT tarzı, sıradaki token'ı tahmin etme) görevidir.
)

model = get_peft_model(model, lora_config)
# get_peft_model: Verilen temel modeli (distilgpt2) alır, target_modules içinde belirtilen katmanların
# yanına LoRA matrislerini (A ve B) ekler ve orijinal ağırlıkları DONDURUR (requires_grad=False).
# Sadece yeni eklenen LoRA matrisleri eğitilebilir (requires_grad=True) hale gelir.

model.print_trainable_parameters()
# PEFT'in sunduğu bu metot, kaç parametrenin eğitilebilir olduğunu ve bunun toplam parametrelere oranını gösterir.
# Beklenen çıktı, eğitilebilir parametre oranının ÇOK küçük (genellikle %1'in altı) olduğunu gösterecektir.

lora_config = LoraConfig(
    r=8,                        # Rank: LoRA matrislerinin iç boyutu. Düşük rank, az parametre demektir.
    lora_alpha=16,               # Ölçeklendirme katsayısı (genelde r'nin 2 katı tercih edilir).
    target_modules=["c_attn"],   # GPT-2 mimarisinde, query/key/value projeksiyonlarını içeren birleşik katmanın adı.
                                  # Farklı mimarilerde (Llama, Mistral) bu isim değişir (örn. ["q_proj", "v_proj"]).
    lora_dropout=0.05,            # LoRA katmanlarına %5 dropout uygulanır -> aşırı öğrenmeye karşı küçük bir önlem.
    bias="none",                  # Bias terimleri eğitilmeyecek (en yaygın ve hafif tercih).
    task_type=TaskType.CAUSAL_LM, # Bu, bir "causal language modeling" (GPT tarzı, sıradaki token'ı tahmin etme) görevidir.
)

model = get_peft_model(model, lora_config)
# get_peft_model: Verilen temel modeli (distilgpt2) alır, target_modules içinde belirtilen katmanların
# yanına LoRA matrislerini (A ve B) ekler ve orijinal ağırlıkları DONDURUR (requires_grad=False).
# Sadece yeni eklenen LoRA matrisleri eğitilebilir (requires_grad=True) hale gelir.

model.print_trainable_parameters()
# PEFT'in sunduğu bu metot, kaç parametrenin eğitilebilir olduğunu ve bunun toplam parametrelere oranını gösterir.
# Beklenen çıktı, eğitilebilir parametre oranının ÇOK küçük (genellikle %1'in altı) olduğunu gösterecektir.

lora_config = LoraConfig(
    r=8,                        # Rank: LoRA matrislerinin iç boyutu. Düşük rank, az parametre demektir.
    lora_alpha=16,               # Ölçeklendirme katsayısı (genelde r'nin 2 katı tercih edilir).
    target_modules=["c_attn"],   # GPT-2 mimarisinde, query/key/value projeksiyonlarını içeren birleşik katmanın adı.
                                  # Farklı mimarilerde (Llama, Mistral) bu isim değişir (örn. ["q_proj", "v_proj"]).
    lora_dropout=0.05,            # LoRA katmanlarına %5 dropout uygulanır -> aşırı öğrenmeye karşı küçük bir önlem.
    bias="none",                  # Bias terimleri eğitilmeyecek (en yaygın ve hafif tercih).
    task_type=TaskType.CAUSAL_LM, # Bu, bir "causal language modeling" (GPT tarzı, sıradaki token'ı tahmin etme) görevidir.
)

model = get_peft_model(model, lora_config)
# get_peft_model: Verilen temel modeli (distilgpt2) alır, target_modules içinde belirtilen katmanların
# yanına LoRA matrislerini (A ve B) ekler ve orijinal ağırlıkları DONDURUR (requires_grad=False).
# Sadece yeni eklenen LoRA matrisleri eğitilebilir (requires_grad=True) hale gelir.

model.print_trainable_parameters()
# PEFT'in sunduğu bu metot, kaç parametrenin eğitilebilir olduğunu ve bunun toplam parametrelere oranını gösterir.
# Beklenen çıktı, eğitilebilir parametre oranının ÇOK küçük (genellikle %1'in altı) olduğunu gösterecektir.