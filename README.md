# 🏥 MedBot: Fine-Tuning DeepSeek R1 for Medical Reasoning with LoRA

#### **MedBot** is a fine-tuned large language model (LLM) based on [`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B), customized for clinical reasoning and diagnostic tasks. This project explores the practical aspects of fine-tuning large models using **LoRA (Low-Rank Adaptation)** and the efficient **Unsloth** library.
---

## 💡 Objective

The primary goal of this project was to gain a comprehensive understanding of the model fine-tuning process, especially in the medical domain and not on achieving state-of-the-art performance. MedBot aims to: 

* Learn from structured clinical reasoning examples.
* Generate chain-of-thought style explanations.
* Provide final diagnostic or therapeutic suggestions.

The focus was not on achieving state-of-the-art performance, but rather understanding the full lifecycle of model fine-tuning prompt formatting to inference and evaluation.

---

## 📦 Dataset

* **Source:** [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
* **Split Used:** First 500 examples from the training set.
* **Fields:**

  * `Question`: Clinical case questions.
  * `Complex_CoT`: Multi-step reasoning or explanation.
  * `Response`: Final medical decision or diagnosis.

This dataset was ideal for testing the model’s ability to reason through real-world medical scenarios.

---

## 📄 Prompt Format

The model was trained using a structured prompt format that encouraged it to:

1. Analyze the medical query.
2. Think through the reasoning process (chain-of-thought).
3. Provide a final answer.

This format was critical in teaching the model how to produce transparent and interpretable responses.

---

## 🧠 What is LoRA?

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique that drastically reduces the computational burden of training large models. Instead of updating all model weights, LoRA introduces a small number of trainable parameters within key layers. This allows us to:

* Fine-tune models using fewer resources.
* Retain the core knowledge of the original model.
* Achieve competitive results with smaller memory and compute footprints.

In MedBot, LoRA was applied to the DeepSeek model to focus the learning on clinical reasoning without having to retrain the entire 8B-parameter base.

⚡ [Unsloth](https://github.com/unslothai/unsloth) is a cutting-edge framework designed to make LLM fine-tuning faster and more memory-efficient.

For MedBot, Unsloth made it feasible to run LoRA fine-tuning on a single GPU within Kaggle's environment.

---
## 🛠 Tools & Technologies

* 🤖 **Model:** DeepSeek R1 Distill Llama 8B
* 🩺 **Dataset:** Medical-O1 Reasoning (SFT)
* ⚙️ **Fine-Tuning:** LoRA via Unsloth
* 🧪 **Libraries:** Hugging Face Transformers, Datasets, PEFT, PyTorch
* 📈 **Experiment Tracking:** Weights & Biases (wandb)

---

## 📚 Learning Outcome

This project provided end-to-end experience in:

* Fine-tuning LLMs.
* Formatting data for clinical QA tasks.
* Understanding and applying LoRA with minimal compute.
* Performing efficient fine-tuning using the Unsloth.
---

## 🚀 Future Directions

* Increase dataset size and train across more epochs.
* Evaluate performance using real-world benchmarks like MedQA or PubMedQA.
* Build a user-friendly web interface for medical professionals.
* Explore zero-shot and few-shot capabilities on unseen tasks.

---

