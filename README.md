# 🛡️ Content Filtering Pipeline for Generative Video Systems  
**Powered by DeepSeek & Phi-4**

A modular, plug-and-play content filtering system for generative video applications. It leverages state-of-the-art large language models (LLMs) — **DeepSeek** and **Phi-4** — to ensure safety, compliance, and content quality at both input and output stages.

---

## 🔍 Overview

This pipeline is designed to be integrated into any generative video system to filter:

- **📝 User Prompts**: Detect and block NSFW, toxic, or harmful inputs.
- **🎬 Model-Generated Captions / Scripts**: Filter inappropriate or biased outputs before rendering.

The system uses a lightweight, dual-model ensemble of **DeepSeek** and **Phi-4** for high-accuracy, low-latency moderation.

---

## 💡 Use Cases

- Safety layer for text-to-video generation platforms  
- Prompt filtering for creative video tools  
- Caption moderation for automated video narration  
- Compliance and ethical content control for AI media pipelines

---

## 🧠 Models Used

| Model     | Role                              | Highlights                             |
|-----------|-----------------------------------|----------------------------------------|
| DeepSeek  | Primary semantic safety filter     | Large, multilingual, robust reasoning  |
| Phi-4     | Fast secondary validation & scoring| Lightweight, ideal for edge usage      |

---

## 🛠 Features

- ✅ Prompt classification (safe / unsafe / review)
- ✅ Caption filtering and moderation
- ✅ LLM ensemble voting
- ✅ Confidence-based filtering thresholds
- ✅ Easy-to-integrate Python API

