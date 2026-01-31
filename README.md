<div align="center">

<img src="Title image.png" width="100%">

# 🧠 Brain-Decoding-Guide

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

</div>

> 📚 This repo aims to guide researchers who are new to the Brain Decoding field to quickly learn about its techniques, datasets, and applications.

---

## 📖 Table of Contents

- [Introduction](#-introduction)
- [Key Surveys](#-key-surveys)
- [Brain Signal Modalities](#-brain-signal-modalities)
- [Datasets](#-datasets)
- [Core Algorithms](#-core-algorithms)
  - [Traditional ML / Foundational Works](#traditional-ml--foundational-works)
  - [Deep Learning](#deep-learning)
  - [Generative AI & LLMs](#generative-ai--llms) 🌟 *Focus*
- [Applications](#-applications)
- [Clinical Application Cases](#-clinical-application-cases-2024-2025)
- [Learning Resources](#-learning-resources)
- [Metrics & Tools](#-metrics--tools)
- [Contributing](#-contributing)

---

## 📌 Introduction

**Brain decoding** (also referred to as neural decoding) is a computational and neuroscientific technique that extracts meaningful, interpretable information about an individual's **subjective mental states**, **perceptual experiences**, **cognitive processes**, or **behavioral intentions** directly from recorded brain activity (e.g., **fMRI, EEG, MEG, or invasive neural recordings**). It relies on machine learning algorithms, statistical modeling, and neuroscientific insights to map patterns of neural activity to specific mental content, with applications in neuroscience research, brain-computer interfaces (BCIs), and clinical neuroscience.

---

## 📑 Key Surveys

| Year | Title | Venue | Highlights |
|------|-------|-------|------------|
| 2025 | [A Survey on fMRI-based Brain Decoding for Reconstructing Multimodal Stimuli](https://arxiv.org/abs/2503.15978) | TPAMI | Comprehensive dataset/ROI summaries, in-depth model classification (end-to-end, pre-trained, LLM-centric) |
| 2025 | [Deep Neural Networks and Brain Alignment: Brain Encoding and Decoding](https://openreview.net/forum?id=YxKJihRcby) | TMLR | Covers both encoding and decoding, focuses on DL-brain alignment. [[Code]](https://github.com/subbareddy248/Awesome-Brain-Encoding--Decoding) |
| 2025 | [Transformer-based EEG Decoding: A Survey](https://arxiv.org/abs/2507.02320) | ArXiv | 200+ papers (2019-2024) on Transformer architectures for EEG |

---

## 🧬 Brain Signal Modalities

| Modality | Full Name | Spatial Res. | Temporal Res. | Invasiveness | Common Use Cases |
|----------|-----------|--------------|---------------|--------------|------------------|
| **fMRI** | Functional Magnetic Resonance Imaging | ~1-3 mm | ~1-2 s | Non-invasive | Visual/Semantic decoding |
| **EEG** | Electroencephalography | ~10 mm | ~1 ms | Non-invasive | Motor imagery, sleep stages |
| **MEG** | Magnetoencephalography | ~5 mm | ~1 ms | Non-invasive | Language, auditory processing |
| **ECoG** | Electrocorticography | ~1 cm | ~1 ms | Invasive | Speech BCI, epilepsy |
| **sEEG** | Stereoelectroencephalography | ~mm | ~1 ms | Invasive | Deep brain structures |
| **NIRS** | Near-Infrared Spectroscopy | ~10 mm | ~100 ms | Non-invasive | Portable BCI, infants |

> 💡 **Tip**: fMRI excels at *where* (spatial), while EEG/MEG excel at *when* (temporal). Invasive methods (ECoG, sEEG) offer the best of both but require surgery.

---

## 📊 Datasets

### fMRI Datasets

| Year | Dataset | Description | Links |
|------|---------|-------------|-------|
| 2021 | **Natural Scenes Dataset (NSD)** | 7T whole-brain fMRI, 8 subjects, ~70k image trials. The gold standard for visual decoding | [[Website]](https://naturalscenesdataset.org) [[Paper]](https://www.nature.com/articles/s41593-021-00962-x) |
| 2023 | **Algonauts 2023 Challenge** | Predict brain responses to natural scenes (based on NSD) | [[Website]](http://algonauts.csail.mit.edu/) [[Paper]](https://arxiv.org/abs/2301.03198) |

### Multi-modal Datasets

| Year | Dataset | Description | Links |
|------|---------|-------------|-------|
| 2019 | **THINGS Initiative** | 1,854 object concepts, 26,107 images. Includes fMRI, MEG, EEG versions | [[Website]](https://things-initiative.org/) [[Paper]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223792) |

### EEG Datasets

> 🚧 *To be added*

---

## ⚙️ Core Algorithms

### Traditional ML / Foundational Works

| Year | Title | Venue | Highlights | Links |
|------|-------|-------|------------|-------|
| 2016 | Natural speech reveals the semantic maps that tile human cerebral cortex | Nature | Ridge Regression encoding model, semantic atlas mapping | [[Paper]](https://www.nature.com/articles/nature17637) [[Website]](https://gallantlab.org/huth2016/) |

### Deep Learning

| Year | Title | Venue | Highlights | Links |
|------|-------|-------|------------|-------|
| 2019 | Deep Image Reconstruction | PLoS Comp. Bio | End-to-end DL framework for image reconstruction | [[Paper]](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633) |
| 2024 | NeuSpeech | Nature Machine Intelligence | Decodes speech from ECoG using lightweight CNN | [[Paper]](https://www.nature.com/articles/s42256-024-00824-8) |

### Generative AI & LLMs

| Year | Title | Venue | Highlights | Links |
|------|-------|-------|------------|-------|
| 2023 | High-resolution image reconstruction with latent diffusion models from human brain activity | CVPR | Seminal paper using Stable Diffusion for fMRI decoding | [[Paper]](https://www.biorxiv.org/content/10.1101/2022.11.18.517004v3) [[Code]](https://github.com/yu-takagi/StableDiffusionReconstruction) |
| 2023 | MinD-Vis: Seeing Beyond the Brain | CVPR | Sparse masked brain modeling + double-conditioned LDM | [[Paper]](https://arxiv.org/abs/2211.06956) [[Code]](https://github.com/zjc062/mind-vis) |
| 2023 | MinD-Video: Cinematic Mindscapes | NeurIPS | High-quality video reconstruction from fMRI | [[Paper]](https://arxiv.org/abs/2305.11675) [[Code]](https://github.com/jqin4749/MindVideo) [[Website]](https://www.mind-video.com) |
| 2024 | DreamDiffusion | ECCV | Generate images from EEG without text translation | [[Paper]](https://arxiv.org/abs/2306.16934) [[Code]](https://github.com/bbaaii/DreamDiffusion) |
| 2024 | DeWave | NeurIPS | EEG-to-text using discrete encoding + LLMs | [[Paper]](https://arxiv.org/abs/2309.14030) |
| 2024 | MindEye2 | ICML | SOTA fMRI-to-image with extreme data efficiency (1 hour) | [[Paper]](https://arxiv.org/abs/2403.11207) [[Code]](https://github.com/MedARC-AI/MindEyeV2) [[Website]](https://medarc-ai.github.io/mindeye2/) |
| 2024 | MindBridge | CVPR | Cross-subject brain decoding with minimal calibration | [[Paper]](https://arxiv.org/abs/2404.07850) |

---

## 🎯 Applications

| Application | Description | Representative Works |
|-------------|-------------|---------------------|
| **Brain-Computer Interfaces (BCI)** | Decoding motor intentions to control external devices (prosthetics, cursors) | Neuralink, BrainGate |
| **Visual Reconstruction** | Reconstructing seen images or videos from brain activity | MindEye2, MinD-Vis |
| **Speech Decoding** | Decoding imagined or spoken speech for communication in paralysis | NeuSpeech, UC Davis BCI |
| **Clinical Neuroscience** | Understanding cognitive states and diagnosing neurological disorders | - |

---

## 🏥 Clinical Application Cases (2024-2025)

> Sharing recent breakthroughs in clinical brain decoding to provide a more intuitive understanding of the technology.

### Neuralink's First Human Patient: Telepathy (2024)

*In January 2024, Neuralink implanted its "Telepathy" device in Noland Arbaugh, a patient with quadriplegia. He was able to control a computer cursor to play chess and Civilization VI solely through his thoughts.*

- **Key Tech**: High-channel count invasive recording, spike sorting, real-time decoding
- [[News]](https://neuralink.com/blog/) [[Video Demo]](https://www.youtube.com/watch?v=2rXrGH52aoM)

### UC Davis ALS Speech BCI (2024)

*In August 2024, researchers at UC Davis Health successfully used a BCI to restore speech for a man with ALS (Casey Harrell). The system decoded his intended speech with >97% accuracy within minutes of use, maintaining his own voice identity.*

- **Key Tech**: High-density ECoG, deep learning for phoneme-to-speech mapping
- [[Press Release]](https://health.ucdavis.edu/news/headlines/new-brain-computer-interface-allows-man-with-als-to-speak-again/2024/08) [[Paper (NEJM)]](https://www.nejm.org/doi/full/10.1056/NEJMoa2314132)

---

## 📚 Learning Resources

### 📺 Video Tutorials & Talks

| Resource | Description | Link |
|----------|-------------|------|
| Neuromatch Academy (NMA) | World-class open course on computational neuroscience. Highly recommended for encoding/decoding basics | [[Bilibili]](https://search.bilibili.com/all?keyword=Neuromatch) |

### 📖 Community & Articles

> 🚧 *To be added: Zhihu columns, WeChat articles, blog posts*

---

## 📏 Metrics & Tools

### Evaluation Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| **Encoding** | Pearson r, R² | Correlation between predicted and actual brain activity |
| **Low-level Reconstruction** | PixCorr, SSIM, PSNR | Pixel-level similarity |
| **High-level Reconstruction** | CLIP Score, Inception Score | Semantic/perceptual similarity |

### Tools & Libraries

| Tool | Description | Link |
|------|-------------|------|
| **MNE-Python** | Analyze MEG, EEG, sEEG, ECoG, NIRS data | [[Website]](https://mne.tools/stable/index.html) |
| **Nilearn** | Statistical learning on NeuroImaging (fMRI) | [[Website]](https://nilearn.github.io/) |
| **PyCortex** | Visualize fMRI on cortical surface | [[GitHub]](https://github.com/gallantlab/pycortex) |
| **Brain-Score** | Benchmark ANN-brain alignment | [[Website]](https://www.brain-score.org/) |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

<div align="center">

**If you find this guide helpful, please consider giving it a ⭐!**

</div>
