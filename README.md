# Persian Text Summarization using Fine-Tuned ParsT5

An abstractive text summarization system optimized for the Persian language. This project leverages **ParsT5-base**, a specialized Encoder-Decoder Transformer architecture, fine-tuned on a cleaned, multi-domain Persian news dataset to generate concise and contextually coherent summaries.

---

## 📌 Project Overview
Abstractive text summarization in Persian presents unique challenges due to complex syntax, rich morphology, and the lack of standardized preprocessed corpora. This repository showcases an end-to-end NLP pipeline that addresses these challenges. By applying robust Persian text normalization and fine-tuning a pre-trained **ParsT5** model, this system achieves high lexical and structural alignment with human-generated summaries.

---

## ⚙️ Model Architecture & Pre-training
The core engine is based on **ParsT5-base**, an architecture adapted from T5-base specifically for Persian Natural Language Processing. 

### Pre-training Setup
* **Corpus**: Pre-trained on the massive **OSCAR 21.09** dataset containing approximately 35 GB of Persian text.
* **Methodology**: Utilized a Self-Supervised Learning approach with a Span-Masked Language Modeling objective.
* **Framework**: Implemented via JAX/Flax using `FlaxT5ForConditionalGeneration` and trained using parallel execution (`jax.pmap`) for over 725,000 steps.

### Structural Specifications
* **Layers**: 12 Encoder layers and 12 Decoder layers.
* **Dimensions**: Hidden vector dimension ($d_{model}$) = 768; Internal Feed-Forward Network (FFN) dimension = 2048.
* **Attention Mechanism**: Multi-Head Attention with 12 heads and Key/Value dimensions of $d_{kv}$ = 64 per head.
* **Activation**: Gated-GELU activation function in the FFN sub-layers.
* **Positional Bias**: Employs a Relative Position Bias mechanism with 32 Relative Attention Buckets.
* **Vocabulary**: 32,103 tokens generated via the SentencePiece Unigram algorithm.

---

## 📊 Dataset & Preprocessing Pipeline
The model was fine-tuned using the **Persian-News-Dataset** (`archive_v5.csv`), a sequence-to-sequence corpus structured into `body` and `abstract` pairs.

### Preprocessing & Cleansing Steps
1. **Character Standardization**: Converted non-standard Arabic characters to their correct Persian equivalents.
2. **Noise Removal**: Stripped web hyperlinks, HTML tags, and invalid structural characters.
3. **Filtering**: Retained only standard Persian alphanumeric characters and fundamental punctuation, subsequently purging duplicate or empty sequences.
4. **Sampling & Splitting**: To balance computational constraints during initial experimentation, a representative 10% stratified subset was extracted using random sampling (`random_state=42`). The data was split into:
   * **Training Set**: 70% (25,701 samples)
   * **Validation Set**: 15% (5,508 samples)
   * **Test Set**: 15% (5,508 samples)

### Text-to-Text Formatting
To explicitly guide the model's objective, a structural task prompt was prepended to every input:
`"خلاصه کن: " + [News Body]`. 
The target output was mapped directly to the reference summary.

---

## 🚀 Fine-Tuning Setup & Hyperparameters
Fine-tuning was executed using Hugging Face's `Seq2SeqTrainer` ecosystem.

* **Sequence Padding & Truncation**: Maximum input length was constrained to 512 tokens, and maximum target label length to 18 tokens. Dynamic sequence-to-sequence padding was managed via `DataCollatorForSeq2Seq`.
* **Optimization Parameters**:
  * **Epochs**: 3
  * **Learning Rate**: 1e-4
  * **Batch Size**: 4 (optimized for memory stability)
* **Inference Strategy**: During evaluation, text generation was executed autoregressively utilizing **Beam Search** with `num_beams=4` and a maximum output ceiling of 128 tokens to guarantee linguistic fluency and contextual coherence.
* **Metrics Alignment**: Evaluated using ROUGE benchmarks (ROUGE-1, ROUGE-2, ROUGE-L) calculated via the native **Hazm** Persian tokenizer to ensure perfect vocabulary alignment.

---

## 📈 Experimental Results

### Training History
The model exhibited steady convergence across all training epochs without suffering from overfitting, showing a simultaneous decrease in cross-entropy loss and a sharp rise in ROUGE metrics.

| Epoch | Training Loss | Validation Loss | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :---: | :-----------: | :-------------: | :-----: | :-----: | :-----: |
|   1   |     12.19     |      5.93       | 0.2153  | 0.0754  | 0.1850  |
|   2   |     8.17      |      3.79       | 0.3453  | 0.1652  | 0.3052  |
|   3   |     6.37      |      2.87       | 0.4054  | 0.2204  | 0.3602  |

### Final Evaluation on Unseen Test Data
The final assessment on the held-out test set proved the robust generalizability of the model, mirroring the performance observed on the validation split.

| Evaluation Metric | Measured Value |
| :--- | :--- |
| **Test Loss** | 2.8822 |
| **ROUGE-1** | 0.4163 |
| **ROUGE-2** | 0.2332 |
| **ROUGE-L** | 0.3684 |
| **Test Runtime** | 9000.21 seconds |
| **Throughput** | 0.612 samples/second |

---

## 🔬 Qualitative Case Studies & Out-of-Domain Generalization
Beyond raw quantitative metrics, the fine-tuned model was subjected to evaluation across diverse domains (Neuroscience, Descriptive Literature, and Biology) to test zero-shot capabilities:

1. **Scientific Text (Neuroscience)**: Extracted core mechanisms of memory reconstruction accurately, achieving an isolated sample score of **ROUGE-1: 0.6735**.
2. **Literary Narrative (Descriptive Essay)**: Despite being fine-tuned heavily on structured news items, the model successfully condensed abstract emotional themes into single coherent sentences.
3. **Biological Text (Circadian Rhythms)**: Effectively synthesized complex multi-variable interactions (light, temperature, and feeding schedules) into a highly concise summary.

---

## 🛠️ Technologies & Libraries Used
* **Core Modeling**: Hugging Face `transformers`, `Seq2SeqTrainer`
* **Pre-training Framework**: JAX / Flax
* **Data Manipulation**: Pandas, NumPy
* **Persian NLP Preprocessing**: Hazm Tokenizer, SentencePiece Tokenizer