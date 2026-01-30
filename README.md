# 🧠 Brain-Decoding-Guide
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
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
- [Metrics & tools](#metrics--tools)

  ---

## Introduction

Brief definition of Brain Decoding: 
Brain decoding (also referred to as neural decoding) is a computational and neuroscientific technique that extracts meaningful, interpretable information about an individual’s subjective mental states, perceptual experiences, cognitive processes, or behavioral intentions directly from recorded brain activity (e.g., fMRI, EEG, MEG, or invasive neural recordings). It relies on machine learning algorithms, statistical modeling, and neuroscientific insights to map patterns of neural activity to specific mental content—moving beyond simply detecting brain activation to decoding the representational information encoded in neural signals, with applications in neuroscience research, brain-computer interfaces (BCIs), and clinical neuroscience.

## Key Surveys

* **[2025] Deep Neural Networks and Brain Alignment: Brain Encoding and Decoding** - Oota et al. [TMLR]
    * *A comprehensive survey covering both encoding and decoding, focusing on deep learning models and their alignment with brain activity.*
    * [[Paper]](https://openreview.net/forum?id=YxKJihRcby) [[Code]](https://github.com/subbareddy248/Awesome-Brain-Encoding--Decoding)

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

## Core Algorithms

### Traditional ML / Foundational works
* **[2013] Ridge Regression** - Huth et al.
    * *Classic encoding model using ridge regression to map semantic features to brain activity.*
    * *Often used as a baseline.*

### Deep Learning
* **[2019] Deep Image Reconstruction** - Shen et al. [PLoS Comp. Bio]
    * *End-to-end deep learning framework for image reconstruction.*
    * [[Paper]](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633)

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

## Applications

* **Brain-Computer Interfaces (BCI)**: Decoding motor intentions to control external devices (prosthetics, cursors).
* **Visual Reconstruction**: Reconstructing seen images or videos from brain activity (e.g., "reading dreams").
* **Speech Decoding**: Decoding imagined or spoken speech from neural signals for communication in paralysis.
* **Clinical Neuroscience**: Understanding cognitive states and diagnosing neurological disorders.

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
