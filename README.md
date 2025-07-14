# 👋 Hi, I'm Lucas

I'm a machine learning researcher passionate about decoding complex systems—from brainwaves to biology. My work combines deep learning, signal processing, and simulation to understand and model temporal phenomena in neuroscience and the environment.

- 🧠 Currently working at [Beacon Biosignals](https://beacon.bio/), building models on EEG data for neurology.
- 🌍 Previously researched **wildfire dynamics** in the [Fire Management & Advanced Analytics Center](https://www.fire2a.com/), using landscape simulation, spatial statistics and deep learning.
- 🔬 Interested in **self-supervised learning**, **generative models**, and **representation learning** for healthcare and biology.
- 🛠️ Skilled in Python, PyTorch, scikit-learn, and scientific computing.

---

## 🔥 Featured Projects

### 📘 [Deep reinforcement learning for optimal firebreak placement in forest fire prevention](https://github.com/lucasmurray97/Tesis)
> **Paper and Master's Thesis**

In this paper, we propose a **Deep Reinforcement Learning (DRL)** framework to optimize the placement of firebreaks—structures that halt wildfire propagation—across large forested areas. Our approach offers a scalable and interpretable alternative to stochastic optimization or mixed-integer programming techniques, which become computationally intractable at scale.

#### 🧠 Key Features
- Formulated firebreak placement as a **sequential decision-making problem**.
- Trained a **DQN agent** on simulated wildfire data from Cell2Fire.
- Compared results to benchmark heuristic and optimization-based methods.

#### 📈 Results
- Achieved competitive performance against mixed-integer optimization baselines.
- DRL agent learned spatial patterns of fire propagation and placed firebreaks in **high-impact regions**.
- The agent's improvement ranges between 1.59%–1.7% with respect to the heuristic, depending on the size of the instance, and 4.79%–6.81% when compared to a random solution.

#### 🔧 Techniques Used
- Reinforcement Learning (DQN)
- Simulation-based training environment (Cell2Fire)
- Forest grid encoding and episodic decision-making

#### 🏫 Published in:
**Applied Soft Computing, Elsevier** (2025)  
DOI: [10.1016/j.asoc.2025.111234](https://www.sciencedirect.com/science/article/pii/S1568494625003540)

---

### 🔗 [Graph Scenarios: Firebreak Optimization with Graph VAEs](https://github.com/lucasmurray97/graph_scenarios)  
> **Using Generative Models to Improve Wildfire Management**

This project introduces a novel framework for improving forest firebreak placement using **Graph Variational Autoencoders (VAEs)** combined with **two-stage stochastic optimization**. Instead of randomly selecting simulation scenarios for optimization, we identify a compact and representative subset using latent space clustering—boosting both performance and generalization.

- Encoded **50,000 fire spread simulations** (from Cell2Fire) into low-dimensional latent spaces using:
  - [VGAE (Kipf & Welling, 2016)](https://arxiv.org/abs/1611.07308)
  - [GraphVAE (Simonovsky & Komodakis, 2018)](https://arxiv.org/abs/1802.03480)
- Used **K-Medoids clustering** to extract representative graphs in latent space.
- Integrated with existing optimization model (Vilches et al., 2023) for **strategic firebreak allocation**.
- Validated improvements in wildfire mitigation over random sampling.

#### 📈 Results
- **3.8% to 5.6% reduction in burned forest area** across 1,000 evaluation simulations.
- Statistically significant performance gains (p < 0.05 for most experiments).
- Cluster-selected scenarios captured **higher-magnitude fires**, improving optimization robustness.

#### 🧪 Techniques Used
- Graph encoding with PyTorch Geometric
- Variational autoencoders for DAGs
- UMAP + K-Medoids for latent structure discovery
- Stochastic optimization via two-stage MILP


---

### 🔬 [FireEncoder](https://github.com/lucasmurray97/fireEncoder)
> **Latent Representation Meets Genetic Search**

A novel research project combining **latent space encoding** with **genetic algorithms** for search and optimization. This hybrid approach explores how learned neural representations can inform and improve evolutionary strategies.

- Uses VAEs to compress complex data into search-ready representations.
- Mixes convenient properties of those representations to enhance genetic-based search.

---

### 🌲 [Deep Crowns: Interpretability of Wildfire Spread Models](https://github.com/lucasmurray97/deep-crowns-biobio)
> **U-Net + Grad-CAM for Wildfire Prediction**

This project explores the **interpretability of deep learning models** trained to predict wildfire spread. A U-Net segmentation model is used to simulate wildfire front propagation, while **Grad-CAM** helps reveal which features (terrain, vegetation, etc.) drive extreme fire predictions.

- Spatially aware CNN architecture (U-Net).
- Grad-CAM integration for saliency and visual insights.
- Aimed at better understanding wildfire behavior through interpretable ML.

---

### 🤖 [LLM Project: Open QA with DPR + LoRA](https://github.com/lucasmurray97/llm-project)
> **Efficient Open-Domain Question Answering**

An implementation of **Dense Passage Retrieval (DPR)** fine-tuned using **Low-Rank Adaptation (LoRA)** for open-domain question answering.

- Based on Karpukhin et al.'s DPR (2020) and Hu et al.'s LoRA (2021).
- Fine-tuned on **MSMARCO** to boost top-K retrieval accuracy.
- Efficient adaptation of large models for real-world QA tasks.
  
---

## 🛠 Tech & Tools

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=flat)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=flat)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white&style=flat)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=flat)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?logo=matplotlib&logoColor=white&style=flat)
![Git](https://img.shields.io/badge/-Git-F05032?logo=git&logoColor=white&style=flat)

---

## 📫 Contact

- 📧 lucasmurrayh at gmail dot com
