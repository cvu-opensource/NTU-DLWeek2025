# NTU-DLWeek2025 - High-Lie-ter 🖍️

## Overview 📰
In today's digital landscape, misinformation and bias pose significant threats to media integrity. Our goal during the NTU Deep Learning Week 2025 Hackathon was to **develop AI-powered solutions that uphold media integrity and foster trust in digital content.**

To tackle this, Team Perchance built **High-Lie-ter**, a web extension that helps users assess online content for bias and misinformation. 

Our approach to applying **Deep Learning** includes leveraging **Self-Supervised Learning (SSL)** to enable our model to develop an inherent understanding of multiple dimensions in text—allowing for more adaptable and reliable analysis across different contexts. We also adapted **Parameter Efficient Fine Tuning via LoRA Adapters** (Additive fine tuning) to fine tune our pre-trained model without affecting its previous understanding of texts.

## **How It Works ⚙️**  
High-Lie-ter provides users with a way to analyze selected portions of web content through a three-step pipeline:

---

### **1. Bias Classification: 💬**  
Our bias classification system is designed to analyze **linguistic framing** and assign a **bias likelihood score** to text.

#### **Implementation Details:**  
1. **Pretraining with Self-Supervised Learning (SSL)**  
   - We trained our model using **contrastive learning**, where it learns to differentiate between **original and synthetically biased versions** of the same text.
   - To ensure **scale invariance**, we exposed the model to text samples of **varying lengths**, preventing overfitting to a specific level of bias granularity.

2. **Fine-Tuning Using PEFT on a Pseudo-Labeled Dataset**  
   - We used **Parameter Efficient Fine-Tuning (PEFT)**, specifically **LoRA**, to fine-tune the model on a **pseudo-labeled dataset** of biased and neutral samples.
   - These pseudo-labels were generated using **LLM augmentation**, where we prompted an LLM to rewrite text with increased or decreased bias. This method **removes human annotation biases** from the training process.
  - LoRA enables **rapid adaptation to new domains** (e.g., political news, financial reports) without retraining the entire model.

3. **Inference: Assigning a Bias Likelihood Score**  
   - Given an input text, the fine-tuned model **analyzes linguistic patterns** and produces a **bias confidence score**.
   - The higher the score, the more likely the text **exhibits biased framing** rather than neutral reporting.

### **Why This Theoretically Works**
Bias in text manifests through **word choice, sentence structure, and framing** rather than outright factual inaccuracies. Our approach addresses these nuances by:

- **Contrastive Learning for Bias Directionality**  
  - Instead of static labels, our model **learns bias patterns dynamically** by distinguishing between **neutral and synthetically biased versions of various scales** of the same text.

- **Scale Invariance for Multi-Level Bias Understanding**  
  - Training across **different text lengths** ensures that bias is detected at the **word, sentence, and document level**, improving adaptability.

- **Beyond Bias — A Multi-Dimensional Framework?**
  - Our self-supervised pretraining techniques could extend beyond bias.
  - By helping models understand **abstract traits** without relying on labels, they could understand sarcasm, persuasion, or linguistic shifts—unlocking **new ways to analyze subjective language**.

---

### **2. Misinformation Detection: ❗️**  
Unlike bias detection, which focuses on **framing and tone**, misinformation detection evaluates the **factual accuracy** of text.

#### **Pipeline Overview**  
1. **Extract Key Topics & Claims**  
   - The system processes the input text to identify **salient claims**, using **Named Entity Recognition (NER) and sentence parsing**.

2. **Query Google Search for Reputable Sources**  
   - We construct a **filtered Google Search query** to find **top-ranked news articles** discussing the same topic.
   - The top few articles (e.g., **3-5 high-ranking sources**) are retrieved as **fact-checking references**.

3. **Semantic Comparison via FAISS**  
   - Retrieved articles are **vectorized** using a text embedding model and stored in a **FAISS vector database**.
   - We compare the **semantic similarity** between the input text and retrieved articles, measuring factual alignment.

4. **Misinformation Score Assignment**  
   - If the original text **deviates significantly** from the retrieved articles in meaning, the system **flags potential misinformation**.
   - The final score **quantifies factual alignment**, offering an **objective measure of credibility**.

### **Why Querying the Top Few Articles Creates an Objective Baseline**
Rather than relying on **static misinformation databases**, our approach **leverages real-time search results** to establish an **objective factual baseline**:

- **Querying Top-Ranked Articles Ensures Authority**  
  - We source information from a broad spectrum of publishers, reducing over-reliance on any single outlet.
  - High-ranking search results tend to be **more credible**, as Google prioritizes sources with **domain trust, fact-checking, and backlinks**.
  - The retrieved dataset serves as an evolving reference rather than a static misinformation database, preventing it from becoming outdated.

- **FAISS-Based Semantic Matching Provides Robust Verification**  
  - Instead of simple keyword matching, FAISS **compares meaning at a conceptual level**, ensuring the model **detects factual distortions even if phrasing differs**.

---

### 3. **User accessibility** 👤
   - The extension then **highlights** the analyzed text or article in **different colors corresponing** to scores given.
   - Lower level details are also provided (**bias and misinformation scores**) for user reference.

---

### **Conclusion**  
By combining **bias likelihood scoring** with **real-time misinformation verification**, High-Lie-ter provides a **scalable and theoretically sound approach** to evaluating content credibility.

#### To further understand our project implementation and motivation, you can refer to our [google slides](https://docs.google.com/presentation/d/17OnTREfq-5hSUgQMGFLgPFfXZcpf0sJpVGRPEqv61yk/edit?usp=sharing) or  view the UI implementation in our [video](https://www.youtube.com/watch?v=dS-1T7bmxRo).

### **Potential Future Enhancements include** 🚀

- 🔎 **Multi-Modal** system to take in visual data for **deepfake detection**
- 📝 **Fine-tuning with Ray** through hyperparameter tuning
- 📈 **Improved UI extention** to aid accessibility by modelling Grammarly's web extention
- 🏆 **Objective human labels** of biased and non-biased texts


---
---

## Tech Stack 🔧

Section | Stack
------- | -----
Frontend (Web Extension UI) | HTML, JavaScript, CSS
Bias Classification Model | Self-Supervised Pre-training, PEFT Fine-tuning (LoRA adapters)
Misinformation Pipeline | Keyword Extraction, Google Search API, Vector Store, FAISS


## **Deployment Instructions**
To set up and run High-Lie-Ter, follow these steps:

### **1. Clone the Repository**
```bash
git clone https://github.com/cvu-opensource/NTU-DLWeek2025.git
cd NTU-DLWeek2025
```

### **2. Setup backend virtual environments**
Ensure you have Python installed, then:
```bash
cd model_scripts
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
run_bias_server.bat  # Start the FastAPI backend
```
```bash
cd misinformation_scripts
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
run_misinformation_server.bat  # Start the FastAPI backend
```

### **3. Frontend Setup**
Ensure Node.js and npm are installed, then:
```bash
cd extentiontools
uvicorn server:app --reload --port 7002
```

---
---

## Visualisations
<img src="static/overview.png" alt="Architecture" width="500">

### Bias Detection Model
<img src="static/bias_model.png" alt="Bias Model" width="700">

### Misinformation Detection
<img src="static/misinformation_pipeline.png" alt="Misinformation Pipeline" width="800">

### UI implementation
<img src="static/ui_extention.png" alt="Google Extention Dropdown" width="400">

<img src="static/ui_selection.png" alt="User selection" width="700">

<img src="static/ui_highlight.png" alt="Service returns" width="700">

<br>

---
---

### Contributors

| Name            | Tasks                          | GitHub Profile                        | LinkedIn Profile                       |
|-----------------|-------------------------------|---------------------------------------|----------------------------------------|
| **Gerard Lum**   | Fine-tuning, APIs      | [https://github.com/gerardlke](https://github.com/gerardlke) | [https://www.linkedin.com/in/gerardlumkaien/](https://www.linkedin.com/in/gerardlumkaien/) |
| **Benjamin Goh** | Pre-training, ML expert   | [https://github.com/checkpoint214159](https://github.com/checkpoint214159) | [https://www.linkedin.com/in/benjamin-goh-45a0a7307/](https://www.linkedin.com/in/benjamin-goh-45a0a7307/) |
| **Yeo You Ming**   | Web extention, UI, Web scraper  | [https://github.com/Forfeit-15](https://github.com/Forfeit-15) | [https://www.linkedin.com/in/yeo-you-ming-5b10852aa/](https://www.linkedin.com/in/yeo-you-ming-5b10852aa/) |
| **Skyler Lee**    | Misinformation detection | [https://github.com/N1sh0](https://github.com/N1sh0) | [https://www.linkedin.com/in/skyler-lee-zhan-bin/](https://www.linkedin.com/in/skyler-lee-zhan-bin/) |
