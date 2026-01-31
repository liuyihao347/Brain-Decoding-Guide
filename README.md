<div align="center">

# 🧠 Brain-Decoding-Guide

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

</div>
> 📚This repo aims to guide researchers who are new to the **Brain Decoding** field to quickly learn about its techniques, datasets, and applications.

---

## 📖 Table of Contents

- [Introduction](#introduction)
- [Key Surveys](#key-surveys)
- [Brain Signal Modalities](#brain-signal-modalities)
- [Datasets](#datasets)
- [Core Algorithms](#core-algorithms)
  - [Traditional ML / Foundational works](#traditional-ml) 
  - [Deep Learning](#deep-learning)
  - [Generative AI & LLMs](#generative-ai--llms) 🌟 *Focus*
- [Applications](#applications)
- [Clinical Application Cases](#clinical-application-cases-2024-2025) 🏥 *New*
- [Learning Resources](#learning-resources) 📚 *New*
- [Metrics & tools](#metrics--tools)

  ---

## Introduction

**Brain decoding** (also referred to as neural decoding) is a computational and neuroscientific technique that extracts meaningful, interpretable information about an individual’s **subjective mental states**, **perceptual experiences**, **cognitive processes**, or **behavioral intentions** directly from recorded brain activity (e.g., **fMRI, EEG, MEG, or invasive neural recordings**). It relies on machine learning algorithms, statistical modeling, and neuroscientific insights to map patterns of neural activity to specific mental content, with applications in neuroscience research, brain-computer interfaces (BCIs), and clinical neuroscience.

## Key Surveys

* **[2025] A Survey on fMRI-based Brain Decoding for Reconstructing Multimodal Stimuli** [TPAMI]
    * *A survey featuring comprehensive dataset/ROI summaries, in-depth classification and evaluation of mainstream models (e.g., end-to-end, pre-trained, LLM-centric).*
    * [[Paper]](https://arxiv.org/abs/2503.15978)

* **[2025] Deep Neural Networks and Brain Alignment: Brain Encoding and Decoding** [TMLR]
    * *A comprehensive survey covering both encoding and decoding, focusing on deep learning models and their alignment with brain activity.*
    * [[Paper]](https://openreview.net/forum?id=YxKJihRcby) [[Code]](https://github.com/subbareddy248/Awesome-Brain-Encoding--Decoding)

* **[2025] Transformer-based EEG Decoding: A Survey** - [ArXiv]
    * *Systematic review of 200+ papers (2019-2024) on Transformer and hybrid architectures for EEG decoding across various tasks.*
    * [[Paper]](https://arxiv.org/abs/2507.02320)

## Datasets

* **[2021] Natural Scenes Dataset (NSD)** - Allen et al.
    * *A large-scale fMRI dataset conducted at 7T, consisting of whole-brain, high-resolution fMRI measurements of 8 healthy adult subjects while they viewed thousands of color natural scenes.*
    * [[Website]](https://naturalscenesdataset.org) [[Paper]](https://www.nature.com/articles/s41593-021-00962-x)

* **[2019] THINGS Initiative (fMRI, MEG, EEG)** - Hebart et al.
    * *A global initiative bridging brain and behavior with a shared set of 1,854 object concepts and 26,107 images. Includes THINGS-fMRI, THINGS-MEG, and THINGS-EEG datasets.*
    * [[Website]](https://things-initiative.org/) [[Paper]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223792)

* **[2023] The Algonauts Project 2023 Challenge** - Gifford et al.
    * *A challenge to predict brain responses to complex natural visual scenes, based on the NSD dataset.*
    * [[Website]](http://algonauts.csail.mit.edu/) [[Paper]](https://arxiv.org/abs/2301.03198)

* **[2019] Generic Object Decoding (GOD)** - Horikawa & Kamitani
    * *fMRI dataset used for image reconstruction from human brain activity.*
    * [[Dataset]](https://github.com/kamigaito/mind-vis) (Often used in Mind-Vis and other reconstruction works)

* **[2024] Chisco: A Large-Scale Chinese Imagined Speech Corpus** - Zhang et al.
    * *A high-density EEG dataset for decoding imagined speech, including over 20,000 sentences from healthy adults. The first large-scale Chinese imagined speech dataset.*
    * [[Paper]](https://www.nature.com/articles/s41597-024-04130-3) [[Dataset]](https://figshare.com/articles/dataset/Chisco_A_Large-Scale_Chinese_Imagined_Speech_Corpus/24536644)

* **[2025] ChineseEEG-2: A High-Density EEG Dataset** - [TBD]
    * *A newly released high-density EEG dataset designed for multimodal semantic alignment and neural decoding tasks.*
    * *Note: Details evolving, check recent literature.*

## Core Algorithms

### Traditional ML / Foundational works
* **[2013] Ridge Regression** - Huth et al.
    * *Classic encoding model using ridge regression to map semantic features to brain activity.*
    * *Often used as a baseline.*

### Deep Learning
* **[2019] Deep Image Reconstruction** - Shen et al. [PLoS Comp. Bio]
    * *End-to-end deep learning framework for image reconstruction.*
    * [[Paper]](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633)

* **[2024] NeuSpeech: A Neural Speech Decoding Framework** - Chen et al. [Nature Machine Intelligence]
    * *Decodes human speech from ECoG signals using a lightweight convolutional neural network and speech synthesis. High performance in single-patient decoding.*
    * [[Paper]](https://www.nature.com/articles/s42256-024-00824-8) [[Code]](https://github.com/NeuSpeech/NeuSpeech)

* **[2024] Deep learning in motor imagery EEG signal decoding: A Systematic Review** - [ScienceDirect]
    * *A comprehensive review of deep learning techniques (CNN, RNN, Transformers) applied to Motor Imagery EEG classification.*
    * [[Paper]](https://www.sciencedirect.com/science/article/pii/S174680942400325X)

### Generative AI & LLMs

* **[2023] High-resolution image reconstruction with latent diffusion models from human brain activity (MinD-Vis)** - Takagi & Nishimoto [CVPR]
    * *The seminal paper using Stable Diffusion for fMRI decoding.*
    * [[Paper]](https://arxiv.org/abs/2211.06956) [[Code]](https://github.com/kamigaito/mind-vis)

* **[2024] DreamDiffusion: Generating High-Quality Images from Brain EEG Signals** - Bai et al.
    * *Generating high-quality images directly from EEG signals without translating thoughts into text, using pre-trained text-to-image models.*
    * [[Paper]](https://arxiv.org/abs/2306.16934) [[Code]](https://github.com/bbaaii/DreamDiffusion)

* **[2023] MinD-Video: High-quality Video Reconstruction from Brain Activity** - Chen et al.
    * *Progressively learns spatiotemporal information from continuous fMRI data for high-quality video reconstruction.*
    * [[Paper]](https://arxiv.org/abs/2305.11675) [[Code]](https://github.com/jqin4749/MindVideo) [[Website]](https://www.mind-video.com)

* **[2024] DeWave: Discrete EEG Waves Encoding for Brain Dynamics to Text Translation** - Duan et al.
    * *A framework for translating EEG waves into text using discrete encoding and large language models.*
    * [[Paper]](https://arxiv.org/abs/2309.14030) (Check for code release)

* **[2024] MindEye2: Shared-Subject Models Enable Extreme Data Efficiency in fMRI-to-Image Reconstruction** - Scotti et al. [ICML]
    * *State-of-the-art fMRI-to-image reconstruction model that achieves high performance even with limited data (1 hour) by leveraging shared-subject information.*
    * [[Paper]](https://arxiv.org/abs/2403.11207) [[Code]](https://github.com/MedARC-AI/MindEyeV2) [[Website]](https://medarc-ai.github.io/mindeye2/)

* **[2024] MindBridge: A Cross-Subject Brain Decoding Framework** - Zhao et al. [CVPR]
    * *A novel approach for cross-subject brain decoding that allows transferring decoding capabilities to new subjects with minimal calibration.*
    * [[Paper]](https://arxiv.org/abs/2404.07850) [[Code]](https://github.com/ZhenZHAO/MindBridge)

* **[2024] NeuroCreat: Visual-Semantic Reconstructions to Mental Concept** - Jing et al. [CVPR]
    * *Proposes a multimodal architecture combining visual and textual abilities of LLMs to capture fine-grained semantic information for reconstruction.*
    * [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Jing_Beyond_Brain_Decoding_Visual-Semantic_Reconstructions_to_Mental_Concept_CVPR_2024_paper.html)

## Applications

* **Brain-Computer Interfaces (BCI)**: Decoding motor intentions to control external devices (prosthetics, cursors).
* **Visual Reconstruction**: Reconstructing seen images or videos from brain activity (e.g., "reading dreams").
* **Speech Decoding**: Decoding imagined or spoken speech from neural signals for communication in paralysis.
* **Clinical Neuroscience**: Understanding cognitive states and diagnosing neurological disorders.

## 🏥 Clinical Application Cases (2024-2025)

> Sharing recent breakthroughs in clinical brain decoding to provide a more intuitive understanding of the technology.

* **[2024] Neuralink's First Human Patient: Telepathy**
    * *In January 2024, Neuralink implanted its "Telepathy" device in Noland Arbaugh, a patient with quadriplegia. He was able to control a computer cursor to play chess and Civilization VI solely through his thoughts.*
    * **Key Tech**: High-channel count invasive recording, spike sorting, real-time decoding.
    * [[News]](https://neuralink.com/blog/) [[Video Demo]](https://www.youtube.com/watch?v=2rXrGH52aoM)

* **[2024] UC Davis ALS Speech BCI**
    * *In August 2024, researchers at UC Davis Health successfully used a BCI to restore speech for a man with ALS (Casey Harrell). The system decoded his intended speech with >97% accuracy within minutes of use, maintaining his own voice identity.*
    * **Key Tech**: High-density ECoG, deep learning for phoneme-to-speech mapping.
    * [[Press Release]](https://health.ucdavis.edu/news/headlines/new-brain-computer-interface-allows-man-with-als-to-speak-again/2024/08) [[Paper (NEJM)]](https://www.nejm.org/doi/full/10.1056/NEJMoa2314132)

## 📚 Learning Resources

### 📺 Video Tutorials & Talks
* **[Bilibili] 脑机接口社区 (BCI Community)**
    * *A dedicated channel sharing the latest reports, paper interpretations, and open classes on BCI and brain decoding.*
    * [[Link]](https://space.bilibili.com/393666114)

* **[Bilibili] Neuromatch Academy (NMA) - Computational Neuroscience**
    * *World-class open course on computational neuroscience. Highly recommended for understanding the basics of encoding/decoding models.*
    * [[Link]](https://search.bilibili.com/all?keyword=Neuromatch)

* **[YouTube/Web] The Algonauts Project Talks**
    * *Workshops and talks related to the Algonauts challenge, focusing on visual brain decoding.*
    * [[Link]](http://algonauts.csail.mit.edu/)

### 📖 Community & Articles (Zhihu/Blogs)
* **[Zhihu] Topic: Brain-Computer Interface (脑机接口)**
    * *Follow high-quality contributors discussing fMRI decoding, EEG signal processing, and invasive BCI.*
    * [[Link]](https://www.zhihu.com/topic/19556819/hot)

* **[Zhihu] Topic: Computational Neuroscience (计算神经科学)**
    * *In-depth discussions on neural encoding/decoding theories.*
    * [[Link]](https://www.zhihu.com/topic/19572886/hot)

## Metrics & Tools

### Metrics
* **Encoding Performance**: Pearson Correlation (r), R-squared ($R^2$).
* **Reconstruction Quality**: 
    * Low-level: Pixel Correlation (PixCorr), SSIM, PSNR.
    * High-level (Semantic): CLIP Score (2-way identification accuracy), Inception Score (IS).

### Tools
* **[MNE-Python](https://mne.tools/stable/index.html)**: Open-source Python package for exploring, visualizing, and analyzing human neurophysiological data (MEG, EEG, sEEG, ECoG, NIRS).
* **[Nilearn](https://nilearn.github.io/)**: A Python module for fast and easy statistical learning on NeuroImaging data.
* **[PyCortex](https://github.com/gallantlab/pycortex)**: Python library for visualizing fMRI data on the cortical surface.
* **[Brain-Score](https://www.brain-score.org/)**: A benchmark for comparing the internal representations of artificial neural networks with the brain.

---

## Contributing

Welcome PRs (Pull Requests) for more papers and resources.
