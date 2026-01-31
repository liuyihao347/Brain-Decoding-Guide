<div align="center">

<img src="Title image.png" width="100%">

# 🧠 Brain-Decoding-Guide

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

</div>

> 📚 This repo aims to guide researchers who are new to the **Brain Decoding** field to quickly learn about its techniques, datasets, and applications.

---

## 📖 Table of Contents

- [Introduction](#-introduction)
- [Brain Signal Modalities](#-brain-signal-modalities)
- [Datasets](#-datasets)
- [Key Surveys](#-key-surveys)
- [Foundational Works](#-foundational-works-pre-2023)
- [Recent Advances & Core Algorithms](#-recent-advances--core-algorithms)
  - [Visual Reconstruction](#-visual-reconstruction)
  - [Speech & Language Decoding](#-speech--language-decoding)
  - [Motor & Intention Decoding](#-motor--intention-decoding)
- [Metrics & Tools](#-metrics--tools)
- [Clinical Application Cases](#-clinical-application-cases)
- [Learning Resources](#-learning-resources)
- [Contributing](#-contributing)

---

## 📌 Introduction

**Brain decoding** (also referred to as neural decoding) is a computational and neuroscientific technique that extracts meaningful, interpretable information about an individual's **subjective mental states**, **perceptual experiences**, **cognitive processes**, or **behavioral intentions** directly from recorded brain activity (e.g., **fMRI, EEG, MEG, or invasive neural recordings**). It relies on machine learning algorithms, statistical modeling, and neuroscientific insights to map patterns of neural activity to specific mental content, with applications in neuroscience research, brain-computer interfaces (BCIs), and clinical neuroscience.

---

## 🧬 Brain Signal Modalities

| Modality | Full Name | Spatial Res. | Temporal Res. | Invasiveness | Common Use Cases |
|----------|-----------|--------------|---------------|--------------|------------------|
| **fMRI** | Functional Magnetic Resonance Imaging | ~1-3 mm | ~1-2 s | Non-invasive | Visual/Semantic decoding |
| **EEG** | Electroencephalography | ~10 mm | ~1 ms | Non-invasive | Motor imagery, emotion, sleep |
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
| 2021 | **Natural Scenes Dataset (NSD)** | 7T whole-brain, 8 subjects, ~70k image trials; gold standard for visual decoding | [[Website]](https://naturalscenesdataset.org) [[Paper]](https://www.nature.com/articles/s41593-021-00962-x) |
| 2023 | **Algonauts 2023 Challenge** | Predict brain responses to natural scenes (based on NSD) | [[Website]](http://algonauts.csail.mit.edu/) [[Paper]](https://arxiv.org/abs/2301.03198) |
| 2021 | **Bold Moments Dataset** | 10 subjects, 1,000+ video clips, naturalistic movie stimuli | [[Website]](https://boldmoments.github.io/) [[Paper]](https://www.nature.com/articles/s41467-021-24899-2) |
| 2017 | **Vim-1 / Vim-2 (Gallant Lab)** | Classic visual decoding datasets; natural images & movies | [[Website]](https://gallantlab.org/data/) |

### EEG Datasets

| Year | Dataset | Description | Links |
|------|---------|-------------|-------|
| 2008 | **BCI Competition IV (2a, 2b)** | Motor imagery benchmark; 9 subjects, 4-class MI | [[Website]](https://www.bbci.de/competition/iv/) |
| 2015 | **SEED** | Emotion recognition; 15 subjects, 3 emotions, film clips | [[Website]](https://bcmi.sjtu.edu.cn/home/seed/) |
| 2012 | **DEAP** | Emotion recognition; 32 subjects, music videos, physiological signals | [[Website]](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) |
| 2019 | **THINGS-EEG** | Object recognition; 50 subjects, 22k images, rapid serial presentation | [[Website]](https://things-initiative.org/) |
| 2024 | **MOABB** | Unified benchmark platform; 30+ pipelines, 36 datasets | [[Website]](https://moabb.neurotechx.com/) [[Paper]](https://arxiv.org/abs/2404.15319) |

### MEG Datasets

| Year | Dataset | Description | Links |
|------|---------|-------------|-------|
| 2017 | **Cam-CAN** | Lifespan study; 700 subjects (18-90 yrs), resting + task MEG/fMRI | [[Website]](https://camcan-archive.mrc-cbu.cam.ac.uk/) [[Paper]](https://www.sciencedirect.com/science/article/pii/S1053811915008150) |
| 2019 | **THINGS-MEG** | Object recognition; same stimuli as THINGS-EEG | [[Website]](https://things-initiative.org/) |

### Multi-modal Datasets

| Year | Dataset | Description | Links |
|------|---------|-------------|-------|
| 2019 | **THINGS Initiative** | 1,854 object concepts, 26,107 images; fMRI + MEG + EEG versions | [[Website]](https://things-initiative.org/) [[Paper]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223792) |

---

## 📑 Key Surveys

| Year | Title | Venue | Highlights |
|------|-------|-------|------------|
| 2025 | [A Survey on fMRI-based Brain Decoding for Reconstructing Multimodal Stimuli](https://arxiv.org/abs/2503.15978) | TPAMI | Dataset/ROI summaries, model taxonomy (end-to-end, pre-trained, LLM-centric) |
| 2025 | [Deep Neural Networks and Brain Alignment: Brain Encoding and Decoding](https://openreview.net/forum?id=YxKJihRcby) | TMLR | Encoding + decoding, DL-brain alignment [[Code]](https://github.com/subbareddy248/Awesome-Brain-Encoding--Decoding) |
| 2025 | [Transformer-based EEG Decoding: A Survey](https://arxiv.org/abs/2507.02320) | ArXiv | 200+ papers on Transformer for EEG (2019-2024) |
| 2025 | [Brain Foundation Models: A Survey](https://arxiv.org/abs/2503.00580) | ArXiv | Foundation models for neural signals, pre-training paradigms |
| 2024 | [Deep Representation Learning for EEG-based BCIs: A Review](https://arxiv.org/abs/2405.19345) | ArXiv | Autoencoders, SSL, foundation models for EEG |
| 2022 | [fMRI Brain Decoding and Its Applications in BCI: A Survey](https://pubmed.ncbi.nlm.nih.gov/35203991/) | Brain | Classic ML to deep learning evolution |

---

## 📜 Foundational Works (Pre-2023)

> Milestone papers that established the field.

| Year | Title | Task | Feature | Links |
|------|-------|------|---------|-------|
| 2011 | [Reconstructing Visual Experiences from Brain Activity Evoked by Natural Movies](https://www.cell.com/current-biology/fulltext/S0960-9822(11)00937-7) | Visual | Motion-energy encoding | [[Website]](https://gallantlab.org/) |
| 2016 | [Natural Speech Reveals the Semantic Maps that Tile Human Cerebral Cortex](https://www.nature.com/articles/nature17637) | Semantic | Cortical semantic atlas | [[Website]](https://gallantlab.org/huth2016/) |
| 2017 | [Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization](https://onlinelibrary.wiley.com/doi/10.1002/hbm.23730) | Motor | Interpretable filters | [[Code]](https://github.com/braindecode/braindecode) |
| 2018 | [EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c) | Motor | BCI baseline | [[Code]](https://github.com/vlawhern/arl-eegmodels) |
| 2019 | [Deep Image Reconstruction from Human Brain Activity](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633) | Visual | Feature optimization | |

---

## ⚙️ Recent Advances & Core Algorithms

> High-impact papers from 2023-2025.

### 🖼️ Visual Reconstruction

#### fMRI → Image

| Year | Title | Arch | Feature | Links |
|------|-------|------|---------|-------|
| 2023 | [High-Resolution Image Reconstruction with Latent Diffusion Models from Human Brain Activity](https://openaccess.thecvf.com/content/CVPR2023/html/Takagi_High-Resolution_Image_Reconstruction_With_Latent_Diffusion_Models_From_Human_Brain_CVPR_2023_paper.html) | `Diffusion` | Zero-shot decoding | [[Code]](https://github.com/yu-takagi/StableDiffusionReconstruction) |
| 2023 | [Seeing Beyond the Brain: MinD-Vis](https://arxiv.org/abs/2211.06956) | `Diffusion` | Masked pre-training | [[Code]](https://github.com/zjc062/mind-vis) |
| 2023 | [Reconstructing the Mind's Eye: MindEye](https://arxiv.org/abs/2305.18274) | `Diffusion` | CLIP alignment | [[Code]](https://github.com/MedARC-AI/fmri-reconstruction-nsd) |
| 2024 | [MindEye2: Shared-Subject Models Enable fMRI-to-Image with 1 Hour of Data](https://arxiv.org/abs/2403.11207) | `Diffusion` | Cross-subject transfer | [[Code]](https://github.com/MedARC-AI/MindEyeV2) [[Website]](https://medarc-ai.github.io/mindeye2/) |
| 2024 | [MindBridge: A Cross-Subject Brain Decoding Framework](https://arxiv.org/abs/2404.07850) | `Diffusion` | Unified cross-subject | [[Code]](https://github.com/littlepure2333/MindBridge) |

#### EEG → Image

| Year | Title | Arch | Feature | Links |
|------|-------|------|---------|-------|
| 2024 | [DreamDiffusion: Generating High-Quality Images from Brain EEG Signals](https://arxiv.org/abs/2306.16934) | `Diffusion` | EEG-to-image | [[Code]](https://github.com/bbaaii/DreamDiffusion) |

#### fMRI → Video

| Year | Title | Arch | Feature | Links |
|------|-------|------|---------|-------|
| 2023 | [Cinematic Mindscapes: High-Quality Video Reconstruction from Brain Activity](https://arxiv.org/abs/2305.11675) | `Diffusion` | Temporal modeling | [[Code]](https://github.com/jqin4749/MindVideo) [[Website]](https://www.mind-video.com) |

---

### 🗣️ Speech & Language Decoding

#### Invasive Speech (ECoG/Intracortical)

| Year | Title | Arch | Feature | Links |
|------|-------|------|---------|-------|
| 2023 | [A High-Performance Speech Neuroprosthesis](https://www.nature.com/articles/s41586-023-06377-x) | `RNN` | 62 words/min | [[Code]](https://github.com/fwillett/speechBCI) |
| 2023 | [A High-Performance Neuroprosthesis for Speech Decoding and Avatar Control](https://www.nature.com/articles/s41586-023-06443-4) | `RNN` | Avatar control | |
| 2025 | [A Streaming Brain-to-Voice Neuroprosthesis](https://www.nature.com/articles/s41593-025-01905-6) | `RNN-T` | 80ms streaming | [[Code]](https://github.com/cheoljun95/streaming.braindecoder) |

#### Non-invasive Semantic (fMRI/EEG)

| Year | Title | Arch | Feature | Links |
|------|-------|------|---------|-------|
| 2023 | [Semantic Reconstruction of Continuous Language from Non-invasive Brain Recordings](https://www.nature.com/articles/s41593-023-01304-9) | `Transformer` | GPT decoder | |
| 2024 | [DeWave: Discrete EEG Waves Encoding for Brain Dynamics to Text Translation](https://arxiv.org/abs/2309.14030) | `Transformer` | VQ codebook | |

---

### 🎯 Motor & Intention Decoding

| Year | Title | Arch | Feature | Links |
|------|-------|------|---------|-------|
| 2024 | [CTNet: A Convolutional Transformer Network for EEG-based Motor Imagery Classification](https://www.nature.com/articles/s41598-024-71118-7) | `CNN-Transformer` | Hybrid architecture | |
| 2023 | [EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization](https://ieeexplore.ieee.org/document/9991178) | `Conformer` | Spatial-temporal | [[Code]](https://github.com/eeyhsong/EEG-Conformer) |
| 2022 | [ATCNet: Attention Temporal Convolutional Network for EEG-based Motor Imagery Classification](https://ieeexplore.ieee.org/document/9852687) | `TCN` | Multi-head attention | [[Code]](https://github.com/Altaheri/EEG-ATCNet) |

---

## 📏 Metrics & Tools

### Evaluation Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| **Encoding** | Pearson r, R² | Correlation between predicted and actual brain activity |
| **Low-level Reconstruction** | PixCorr, SSIM, PSNR | Pixel-level similarity |
| **High-level Reconstruction** | CLIP Score, Inception Score | Semantic/perceptual similarity |
| **Classification** | Accuracy, F1, AUC | Standard classification metrics |
| **Retrieval** | Top-k Accuracy, MRR | Retrieval-based evaluation |

### Software & Libraries

| Tool | Description | Link |
|------|-------------|------|
| **MNE-Python** | MEG, EEG, sEEG, ECoG, NIRS analysis | [[Website]](https://mne.tools/) [[GitHub]](https://github.com/mne-tools/mne-python) |
| **Nilearn** | Statistical learning on fMRI data | [[Website]](https://nilearn.github.io/) [[GitHub]](https://github.com/nilearn/nilearn) |
| **Braindecode** | Deep learning for EEG/ECoG/MEG decoding; EEGNet, ShallowNet, etc. | [[Website]](https://braindecode.org/) [[GitHub]](https://github.com/braindecode/braindecode) |
| **TorchEEG** | PyTorch library for EEG processing & models | [[GitHub]](https://github.com/torcheeg/torcheeg) |
| **Net2Brain** | Compare DNN activations with brain activity (RSA, encoding) | [[GitHub]](https://github.com/cvai-roig-lab/Net2Brain) |
| **Neural_Decoding** | Classic + DL decoders (Kalman, Wiener, LSTM, etc.) | [[GitHub]](https://github.com/KordingLab/Neural_Decoding) |
| **PyCortex** | fMRI visualization on cortical surface | [[GitHub]](https://github.com/gallantlab/pycortex) |
| **RSA Toolbox** | Representational Similarity Analysis | [[GitHub]](https://github.com/rsagroup/rsatoolbox) |

### Benchmark Platforms

| Platform | Description | Link |
|----------|-------------|------|
| **Algonauts Project** | Annual challenge for predicting brain responses to visual stimuli | [[Website]](http://algonauts.csail.mit.edu/) |
| **Brain-Score** | Benchmark for comparing DNNs with primate visual cortex | [[Website]](https://www.brain-score.org/) [[GitHub]](https://github.com/brain-score/brain-score) |
| **MOABB** | Mother of All BCI Benchmarks; 36 EEG datasets, 30 pipelines | [[Website]](https://moabb.neurotechx.com/) [[GitHub]](https://github.com/NeuroTechX/moabb) |

---

## 🏥 Clinical Application Cases

> Recent breakthroughs demonstrating real-world clinical impact.

### UCSF Streaming Speech BCI (2025)

*In March 2025, researchers at UCSF published a streaming brain-to-voice neuroprosthesis in Nature Neuroscience. The system achieved real-time speech synthesis with only 80ms latency, enabling natural conversational flow for a patient with severe paralysis and anarthria.*

- **Key Tech**: ECoG recording, RNN-T decoder, personalized voice synthesis
- [[Paper]](https://www.nature.com/articles/s41593-025-01905-6) [[Code]](https://github.com/cheoljun95/streaming.braindecoder)

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

### 📺 Video Tutorials & Courses

| Resource | Description | Link |
|----------|-------------|------|
| **Neuromatch Academy** | World-class open course on computational neuroscience; encoding/decoding basics | [[Website]](https://neuromatch.io/) [[YouTube]](https://www.youtube.com/@neuaboratory) [[Bilibili]](https://search.bilibili.com/all?keyword=Neuromatch) |
| **INCF: Deep Learning in Neuroscience** | Beginner-level DL for neuroscience applications | [[Website]](https://training.incf.org/lesson/fundamentals-deep-learning-neuroscience) |

### 📖 Textbooks & Reading

| Resource | Description | Link |
|----------|-------------|------|
| **Deep Learning (Goodfellow et al.)** | Deep learning bible; free online | [[Website]](https://www.deeplearningbook.org/) |
| **Awesome-Brain-Encoding-Decoding** | Curated paper list | [[GitHub]](https://github.com/subbareddy248/Awesome-Brain-Encoding--Decoding) |

### 🌐 Communities

| Community | Description | Link |
|-----------|-------------|------|
| **NeuroAI WeChat Group** | Chinese community for brain + AI research | Contact via WeChat number `MobiusAI` |
| **BCI Society** | International BCI research community | [[Website]](https://bcisociety.org/) |
| **OHBM** | Organization for Human Brain Mapping | [[Website]](https://www.humanbrainmapping.org/) |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

<div align="center">

**If you find this guide helpful, please consider giving it a ⭐!**

</div>
