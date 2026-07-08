"""lora_wikipedia.ipynb

"""

#pip install wikipedia transformers datasets torch accelerate

#pip install peft

#!pip install --upgrade torchao

import os
import requests
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

# ==========================================
# ODAKLANILACAK TOPIC (KONU) AYARI
# ==========================================
TOPIC = "Artificial intelligence"

# ==========================================
# 1. ADIM: WIKIPEDIA API İLE VERİ ÇEKME
# ==========================================
def fetch_wikipedia_clean_text(topic_name):
    print(f"'{topic_name}' konusu Wikipedia API üzerinden çekiliyor...")
    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "titles": topic_name,
        "prop": "extracts",
        "explaintext": True,
        "exlimit": "max"
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    response = requests.get(url, params=params, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Bağlantı hatası! Durum kodu: {response.status_code}")

    data = response.json()
    pages = data["query"]["pages"]
    page_id = list(pages.keys())[0]

    if page_id == "-1":
        raise Exception(f"Hata: '{topic_name}' başlığına sahip bir Wikipedia sayfası bulunamadı.")

    full_text = pages[page_id]["extract"]
    paragraphs = [p.strip() for p in full_text.split("\n") if len(p.strip()) > 50]

    print(f"Başarılı! Toplam {len(paragraphs)} paragraf veri kazındı.\n")
    return paragraphs

wiki_paragraphs = fetch_wikipedia_clean_text(TOPIC)

# ==========================================
# 2. ADIM: VERİYİ LLM (TOKENIZER) İÇİN HAZIRLAMA
# ==========================================
print("Veriler LLM formatına dönüştürülüyor...")
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

dataset = Dataset.from_dict({"text": wiki_paragraphs})

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ==========================================
# 3. ADIM: GÜÇLENDİRİLMİŞ LoRA AYARLARI
# ==========================================
print("Model ve Güçlendirilmiş LoRA yükleniyor...")
base_model = GPT2LMHeadModel.from_pretrained(model_name)

# Model istediğiniz bilgileri tam veremiyorsa LoRA'nın kapasitesini (r ve alpha) artırırız.
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,                           # 8 yerine 16 yaparak modelin öğrenme kapasitesini artırdık
    lora_alpha=128,                  # 32 yerine 64 yaparak yeni bilgilerin ağırlığını yükselttik
    lora_dropout=0.05,              # Bilgiyi daha sıkı tutması için düşüm oranını azalttık
    target_modules=["c_attn"]
)

model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

folder_name = TOPIC.lower().replace(" ", "_")
output_directory = f"./lora_model_{folder_name}"

training_args = TrainingArguments(
    output_dir=output_directory,
    num_train_epochs=5,                 # 5 yerine 10 tur döndürerek veriyi iyice ezberlemesini sağlıyoruz
    per_device_train_batch_size=2,
    logging_steps=10,
    learning_rate=5e-4,                  # Öğrenme hızını biraz artırdık
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print(f"\n--- GÜÇLENDİRİLMİŞ LoRA İLE EĞİTİM BAŞLIYOR ---")
trainer.train()

# ==========================================
# 4. ADIM: MODELİ KAYDETME VE TEST
# ==========================================
trainer.model.save_pretrained(output_directory)
tokenizer.save_pretrained(output_directory)
print(f"\nEğitim tamamlandı! Klasör: '{output_directory}'\n")

print("--- Eğitilen Model Test Ediliyor ---")
test_prompt = "Alan Turing was the first person to"
inputs = tokenizer(test_prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_length=60, num_return_sequences=1, do_sample=True, top_k=30, temperature=0.7)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nModelin Cevabı:")
print(generated_text)

# Bu kodu boş bir hücreye yapıştırıp çalıştırın
print(f"Toplam çekilen paragraf sayısı: {len(wiki_paragraphs)}\n")

print("--- İLK 5 PARAGRAF ---")
for i, paragraph in enumerate(wiki_paragraphs[:5]):
    print(f"Paragraf {i+1}:")
    print(paragraph)
    print("-" * 50)