<div align="center">

<img src="Title image.png" width="800">

# 🧠 Brain-Decoding-Guide

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

</div>
> 📚This repo aims to guide researchers who are new to the Brain Decoding field to quickly learn about its techniques, datasets, and applications.

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
- [Clinical Application Cases](#clinical-application-cases-2024-2025) 🏥 
- [Learning Resources](#learning-resources) 📚 
- [Metrics & tools](#metrics--tools)

  ---

## Introduction

**Brain decoding** (also referred to as neural decoding) is a computational and neuroscientific technique that extracts meaningful, interpretable information about an individual’s **subjective mental states**, **perceptual experiences**, **cognitive processes**, or **behavioral intentions** directly from recorded brain activity (e.g., **fMRI, EEG, MEG, or invasive neural recordings**). It relies on machine learning algorithms, statistical modeling, and neuroscientific insights to map patterns of neural activity to specific mental content, with applications in neuroscience research, brain-computer interfaces (BCIs), and clinical neuroscience.

## Key Surveys

* **[2025] A Survey on fMRI-based Brain Decoding for Reconstructing Multimodal Stimuli** [TPAMI] - Liu et al.
    * *A survey featuring comprehensive dataset/ROI summaries, in-depth classification and evaluation of mainstream models (e.g., end-to-end, pre-trained, LLM-centric).*
    * [[Paper]](https://arxiv.org/abs/2503.15978)

* **[2025] Deep Neural Networks aniud Brain Alignment: Brain Encoding and Decoding** [TMLR] - SUBBA et al.
    * *A comprehensive survey covering both encoding and decoding, focusing on deep learning models and their alignment with brain activity.*
    * [[Paper]](https://openreview.net/forum?id=YxKJihRcby) [[Code]](https://github.com/subbareddy248/Awesome-Brain-Encoding--Decoding)

* **[2025] Transformer-based EEG Decoding: A Survey** [ArXiv] - Zhang et al.
    * *Systematic review of 200+ papers (2019-2024) on Transformer and hybrid architectures for EEG decoding across various tasks.*
    * [[Paper]](https://arxiv.org/abs/2507.02320)

## Datasets

* **[2021] Natural Scenes Dataset (NSD)** [Nature Neuroscience] - Allen et al.
    * *A large-scale fMRI dataset conducted at 7T, consisting of whole-brain, high-resolution fMRI measurements of 8 healthy adult subjects while they viewed thousands of color natural scenes.*
    * [[Website]](https://naturalscenesdataset.org) [[Paper]](https://www.nature.com/articles/s41593-021-00962-x)

* **[2019] THINGS Initiative (fMRI, MEG, EEG)** [Plos one] - Hebart et al.
    * *A global initiative bridging brain and behavior with a shared set of 1,854 object concepts and 26,107 images. Includes THINGS-fMRI, THINGS-MEG, and THINGS-EEG datasets.*
    * [[Website]](https://things-initiative.org/) [[Paper]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223792)

* **[2023] The Algonauts Project 2023 Challenge** [ArXiv] - Gifford et al.
    * *A challenge to predict brain responses to complex natural visual scenes, based on the NSD dataset.*
    * [[Website]](http://algonauts.csail.mit.edu/) [[Paper]](https://arxiv.org/abs/2301.03198)

## Core Algorithms

### Traditional ML / Foundational works
* **[2016] Natural speech reveals the semantic maps that tile human cerebral cortex** [Nature] - Huth et al.
    * *A foundational study using Ridge Regression (encoding model) to map semantic features to fMRI voxel responses, creating detailed semantic atlases.*
    * [[Paper]](https://www.nature.com/articles/nature17637) [[Website]](https://gallantlab.org/huth2016/)

### Deep Learning
* **[2019] Deep Image Reconstruction** [PLoS Comp. Bio] - Shen et al. 
    * *End-to-end deep learning framework for image reconstruction.*
    * [[Paper]](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633)

* **[2024] NeuSpeech: A Neural Speech Decoding Framework** [Nature Machine Intelligence] - Chen et al. 
    * *Decodes human speech from ECoG signals using a lightweight convolutional neural network and speech synthesis. High performance in single-patient decoding.*
    * [[Paper]](https://www.nature.com/articles/s42256-024-00824-8)

### Generative AI & LLMs

* **[2023] High-resolution image reconstruction with latent diffusion models from human brain activity (MinD-Vis)** [CVPR] - Takagi & Nishimoto 
    * *The seminal paper using Stable Diffusion for fMRI decoding.*
    * [[Paper]](https://arxiv.org/abs/2211.06956) [[Code]](https://github.com/kamigaito/mind-vis)

* **[2024] DreamDiffusion: Generating High-Quality Images from Brain EEG Signals** [ECCV] - Bai et al.
    * *Generating high-quality images directly from EEG signals without translating thoughts into text, using pre-trained text-to-image models.*
    * [[Paper]](https://arxiv.org/abs/2306.16934) [[Code]](https://github.com/bbaaii/DreamDiffusion)

* **[2023] Seeing Beyond the Brain: Conditional Diffusion Model with Sparse Masked Modeling for Vision Decoding** [CVPR] - Chen et al.
    * *MinD-Vis, a two-stage framework combining sparse masked brain modeling and double-conditioned latent diffusion, reconstructs semantically consistent images from fMRI signals.*
    * [[Paper]](https://arxiv.org/abs/2211.06956) [[Code]](https://github.com/zjc062/mind-vis)

* **[2023] Cinematic Mindscapes: High-quality Video Reconstruction from Brain Activity** [NeurIPS] - Chen et al.
    * *Progressively learns spatiotemporal information from continuous fMRI data for high-quality video reconstruction.*
    * [[Paper]](https://arxiv.org/abs/2305.11675) [[Code]](https://github.com/jqin4749/MindVideo) [[Website]](https://www.mind-video.com)

* **[2024] DeWave: Discrete EEG Waves Encoding for Brain Dynamics to Text Translation** [NeurIPS] - Duan et al.
    * *A framework for translating EEG waves into text using discrete encoding and large language models.*
    * [[Paper]](https://arxiv.org/abs/2309.14030)

* **[2024] MindEye2: Shared-Subject Models Enable Extreme Data Efficiency in fMRI-to-Image Reconstruction** [ICML] - Scotti et al.
    * *State-of-the-art fMRI-to-image reconstruction model that achieves high performance even with limited data (1 hour) by leveraging shared-subject information.*
    * [[Paper]](https://arxiv.org/abs/2403.11207) [[Code]](https://github.com/MedARC-AI/MindEyeV2) [[Website]](https://medarc-ai.github.io/mindeye2/)

* **[2024] MindBridge: A Cross-Subject Brain Decoding Framework** [CVPR] - Zhao et al.
    * *A novel approach for cross-subject brain decoding that allows transferring decoding capabilities to new subjects with minimal calibration.*
    * [[Paper]](https://arxiv.org/abs/2404.07850)

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

* **[Bilibili] Neuromatch Academy (NMA) - Computational Neuroscience**
    * *World-class open course on computational neuroscience. Highly recommended for understanding the basics of encoding/decoding models.*
    * [[Link]](https://search.bilibili.com/all?keyword=Neuromatch)


### 📖 Community & Articles (Zhihu/Blogs)

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
