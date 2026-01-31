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
  - [Foundational Works](#-foundational-works-pre-2023)
  - [Recent Advances by Task (2023-2025)](#-recent-advances-by-task-2023-2025)
    - [Visual Reconstruction](#-visual-reconstruction)
    - [Speech & Language Decoding](#-speech--language-decoding)
    - [Motor & Intention Decoding](#-motor--intention-decoding)
  - [By Architecture](#-by-architecture)
  - [By Learning Paradigm](#-by-learning-paradigm)
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

> **Architecture Tags**: `Diffusion` `Transformer` `CNN` `Contrastive` `LLM` `Pre-train` `RNN`

---

### 📜 Foundational Works (Pre-2023)

> Milestone papers that established the field. Not limited to recent years.

| Year | Title | Task | Feature (1-sentence) | Links |
|------|-------|------|----------------------|-------|
| 2011 | Nishimoto et al. (Gallant Lab) | Visual | **First movie reconstruction from fMRI**; motion-energy encoding model, 740+ citations | [[Paper]](https://www.cell.com/current-biology/fulltext/S0960-9822(11)00937-7) |
| 2016 | Huth et al. (Gallant Lab) | Semantic | **Semantic atlas of human cortex**; Ridge Regression maps word embeddings to voxels, 1800+ citations | [[Paper]](https://www.nature.com/articles/nature17637) [[Website]](https://gallantlab.org/huth2016/) |
| 2018 | EEGNet | Motor BCI | **Compact CNN baseline for BCI**; depthwise separable convolutions, 3500+ citations | [[Paper]](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c) [[Code]](https://github.com/vlawhern/arl-eegmodels) |
| 2019 | Shen et al. | Visual | **End-to-end DNN for image reconstruction**; iterative optimization in DNN feature space | [[Paper]](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633) |

---

### 🔬 Recent Advances by Task (2023-2025)

> High-citation papers from the last 3 years, organized by task.

#### 🖼️ Visual Reconstruction

##### fMRI → Image

| Year | Title | Arch | Feature (1-sentence) | Citations | Links |
|------|-------|------|----------------------|-----------|-------|
| 2023 | Takagi & Nishimoto | `Diffusion` | **First to use Stable Diffusion for fMRI**; no fine-tuning, maps fMRI to LDM latent space | 280+ | [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Takagi_High-Resolution_Image_Reconstruction_With_Latent_Diffusion_Models_From_Human_Brain_CVPR_2023_paper.html) [[Code]](https://github.com/yu-takagi/StableDiffusionReconstruction) |
| 2023 | MinD-Vis | `Diffusion` `Pre-train` | **Masked brain modeling + double-conditioned LDM**; self-supervised pre-training | 200+ | [[Paper]](https://arxiv.org/abs/2211.06956) [[Code]](https://github.com/zjc062/mind-vis) |
| 2023 | MindEye | `Diffusion` `Contrastive` | **Dual-pathway (retrieval + reconstruction)**; maps fMRI to CLIP space | 150+ | [[Paper]](https://arxiv.org/abs/2305.18274) [[Code]](https://github.com/MedARC-AI/fmri-reconstruction-nsd) |
| 2024 | MindEye2 | `Diffusion` `Contrastive` | **1-hour fMRI data suffices**; cross-subject pre-training enables extreme efficiency | - | [[Paper]](https://arxiv.org/abs/2403.11207) [[Code]](https://github.com/MedARC-AI/MindEyeV2) [[Website]](https://medarc-ai.github.io/mindeye2/) |
| 2024 | MindBridge | `Diffusion` | **First unified cross-subject model**; cyclic reconstruction for subject-invariant features | - | [[Paper]](https://arxiv.org/abs/2404.07850) [[Code]](https://github.com/littlepure2333/MindBridge) |

##### EEG → Image

| Year | Title | Arch | Feature (1-sentence) | Citations | Links |
|------|-------|------|----------------------|-----------|-------|
| 2024 | DreamDiffusion | `Diffusion` `Pre-train` | **First EEG-to-image without text**; masked signal pre-training + CLIP alignment | 150+ | [[Paper]](https://arxiv.org/abs/2306.16934) [[Code]](https://github.com/bbaaii/DreamDiffusion) |

##### fMRI → Video

| Year | Title | Arch | Feature (1-sentence) | Citations | Links |
|------|-------|------|----------------------|-----------|-------|
| 2023 | MinD-Video | `Diffusion` `Contrastive` | **First high-quality video reconstruction**; progressive learning with temporal inflation | 100+ | [[Paper]](https://arxiv.org/abs/2305.11675) [[Code]](https://github.com/jqin4749/MindVideo) [[Website]](https://www.mind-video.com) |

---

#### 🗣️ Speech & Language Decoding

##### Invasive Speech (ECoG)

| Year | Title | Arch | Feature (1-sentence) | Citations | Links |
|------|-------|------|----------------------|-----------|-------|
| 2023 | Willett et al. | `RNN` | **62 words/min speech-to-text**; highest-performing speech neuroprosthesis (Nature) | 400+ | [[Paper]](https://www.nature.com/articles/s41586-023-06377-x) [[Code]](https://github.com/fwillett/speechBCI) |
| 2023 | Metzger et al. (Chang Lab) | `RNN` `CNN` | **Speech + avatar control**; multimodal output with facial expressions (Nature) | 300+ | [[Paper]](https://www.nature.com/articles/s41586-023-06443-4) |
| 2024 | NeuSpeech | `CNN` | **Differentiable speech synthesizer**; lightweight CNN, natural-sounding output | - | [[Paper]](https://www.nature.com/articles/s42256-024-00824-8) |

##### Non-invasive Semantic (fMRI/EEG)

| Year | Title | Arch | Feature (1-sentence) | Citations | Links |
|------|-------|------|----------------------|-----------|-------|
| 2023 | Tang et al. (Semantic Decoder) | `Transformer` `LLM` | **First continuous language from fMRI**; GPT-based decoder (Nature Neuroscience) | 500+ | [[Paper]](https://www.nature.com/articles/s41593-023-01304-9) |
| 2024 | DeWave | `Transformer` `LLM` | **EEG-to-text via discrete codebook**; vector-quantized encoding + LLM decoder | - | [[Paper]](https://arxiv.org/abs/2309.14030) |

---

#### 🎯 Motor & Intention Decoding

| Year | Title | Arch | Feature (1-sentence) | Citations | Links |
|------|-------|------|----------------------|-----------|-------|
| 2024 | CTNet | `CNN` `Transformer` | **Hybrid CNN-Transformer for motor imagery**; 82.5% on BCI-IV-2a | - | [[Paper]](https://www.nature.com/articles/s41598-024-71118-7) |
| 2025 | AMEEGNet | `CNN` `Attention` | **Multi-scale EEGNet + channel attention**; 81-95% across BCI benchmarks | - | [[Paper]](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1540033/full) |

---

### 🏗️ By Architecture

> Core network architectures used in brain decoding.

| Architecture | Representative Work | Year | Citations | Why It Matters | Links |
|--------------|---------------------|------|-----------|----------------|-------|
| **CNN** | EEGNet | 2018 | 3500+ | Compact depthwise-separable CNN; de facto baseline for EEG-BCI | [[Paper]](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c) |
| **Transformer** | Tang et al. (Semantic Decoder) | 2023 | 500+ | GPT-based decoder for continuous language from fMRI | [[Paper]](https://www.nature.com/articles/s41593-023-01304-9) |
| **RNN/LSTM** | Willett et al. | 2023 | 400+ | RNN for neural-to-text; 62 words/min speech BCI | [[Paper]](https://www.nature.com/articles/s41586-023-06377-x) |
| **Diffusion (U-Net)** | Takagi & Nishimoto | 2023 | 280+ | First Stable Diffusion for fMRI image reconstruction | [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Takagi_High-Resolution_Image_Reconstruction_With_Latent_Diffusion_Models_From_Human_Brain_CVPR_2023_paper.html) |

---

### 📐 By Learning Paradigm

> Training strategies and learning methods in brain decoding.

| Paradigm | Representative Work | Year | Key Idea | Links |
|----------|---------------------|------|----------|-------|
| **Contrastive** | MindEye | 2023 | CLIP-based alignment of fMRI and image embeddings; retrieval + reconstruction | [[Paper]](https://arxiv.org/abs/2305.18274) |
| **Masked Pre-training** | EEGPT | 2024 | Mask-based self-supervised learning on large EEG corpora (NeurIPS 2024) | [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html) |
| **Masked Pre-training** | BrainLM | 2024 | 6,700h fMRI pre-training with masked prediction (ICLR 2024) | [[Paper]](https://openreview.net/forum?id=RwI7ZEfR27) |
| **Masked Pre-training** | MinD-Vis | 2023 | Sparse masked brain modeling for fMRI representation learning | [[Paper]](https://arxiv.org/abs/2211.06956) |
| **Generative (Diffusion)** | DreamDiffusion | 2024 | EEG-to-image via diffusion prior without text intermediate | [[Paper]](https://arxiv.org/abs/2306.16934) |

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
