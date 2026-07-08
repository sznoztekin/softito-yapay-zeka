# Büyük Dil Modelleri (LLM) Çalışma ve İnce Ayar (Fine-Tuning) Deposu 🤖

Bu depo, LLM'lerin çalışma prensiplerini anlamak, veri ön işleme süreçlerini yönetmek ve LoRA (Low-Rank Adaptation) gibi modern tekniklerle modelleri özelleştirmek için hazırlanan uygulama kodlarını içerir.

## 📁 Proje İçeriği
- **llm_veri_ön_işleme_çalışma.py**: Ham metin verilerinin temizlenmesi, HTML etiketlerinden arındırılması ve tokenizer yapılarının (BPE vb.) uygulanması süreçlerini içerir.
- **sıcaklık_ve_otoregresyon.py**: LLM'lerin "sıcaklık" (temperature) parametresinin ve otoregresif metin üretimindeki olasılık dağılımlarının nasıl çalıştığını gösteren uygulamalı analizdir.
- **lora_wikipedia.py**: Wikipedia API kullanarak belirli bir konu hakkında veri çekme ve GPT-2 modelini LoRA tekniği ile o konu özelinde eğitme sürecini kapsar.
- **finetuning.py**: Hugging Face `Trainer` API ve PEFT kütüphanesi kullanılarak modellerin verimli bir şekilde eğitilmesi için gereken konfigürasyon yapılarını içerir.

## 🛠 Kullanılan Teknolojiler
* **Hugging Face Stack**: `transformers`, `datasets`, `peft`, `accelerate`.
* **PyTorch**: Model mimarileri ve GPU hesaplamaları.
* **LoRA**: Büyük modelleri sınırlı kaynaklarla eğitmek için verimli parametre adaptasyonu.
* **OpenAI/Wikipedia API**: Veri toplama ve zenginleştirme.

## 📋 Öne Çıkan Özellikler
1. **Verimli Eğitim**: LoRA (Low-Rank Adaptation) sayesinde tam model eğitimi yerine sadece belirli katmanların eğitilmesi ile düşük bellek kullanımı.
2. **Parametre Analizi**: Sıcaklık (temperature) ayarlarının metin üretimindeki "yaratıcılık" vs "tutarlılık" dengesi üzerindeki etkisi.
3. **Veri Hattı (Pipeline)**: Wikipedia'dan veriyi çekip, temizleyip, tokenize ederek eğitime hazır hale getiren uçtan uca iş akışı.

## 🚀 Kullanım
Projedeki analizleri çalıştırmak için gerekli bağımlılıkları yükleyin:
```bash
pip install transformers datasets peft accelerate wikipedia torch