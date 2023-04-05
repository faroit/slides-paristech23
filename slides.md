---
# try also 'default' to start simple
theme: default
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://static.wixstatic.com/media/dc75cb_c43d95dcd4c84159a814f52e85bb98e0~mv2.png/v1/fill/w_2215,h_1263,al_c,q_95/dc75cb_c43d95dcd4c84159a814f52e85bb98e0~mv2.webp
# apply any windi css classes to the current slide
class: 'text-center'
# https://sli.dev/custom/highlighters.html
highlighter: shiki
# show line numbers in code blocks
lineNumbers: true
# some information about the slides, markdown enabled
info: |
  ## Music Separation: is it solved, yet?
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
# persist drawings in exports and build
drawings:
  persist: false
# page transition
transition: none
# download: true
# use UnoCSS
css: unocss
fonts:
  sans: 'Fira Sans'
  serif: 'Roboto Slab'
  mono: 'Fira Code'

---

<!-- inria-srp-slides, slides-deeplearning-musicseparation -->
# Music Separation: is it solved, yet?

## Fabian-Robert Stöter, Audioshake

---

# Is it solved, yet?

<center>
  <img width="300" src="/IMG_4123.jpg">
</center>

---
# Research 

- Large-Language Models (LLM) can't solve music separation (not yet)

---

# Introduction to Music Separation 1

<SlideCurrentNo style="position: absolute; top:1em; right: 2em" />

---

# Introduction to Music Separation 2

---

# Introduction to Music Separation 3

---

# Introduction to Music Separation 4

---

# History of models

---

# Music Separation is a good test balloon for AI

- it is a hard problem
- it is a well-defined problem
- it is a problem that is not solved yet
- you can use all architectures
- timeline of separation systems
- today: LLMs tokenize audio

---

# Evaluation

- what does it mean to evaluate a music separation system?
- what are the metrics?
- when is it solved?

---

# Example: "Clean-Source-Separation"

- mixture is a clean source, hence no separation is required
- plot visqol and pesq scores

---
layout: two-cols
---

---

# When is a system solved?

- Does scientific research align with real-world applications?
- Do datasets and metrics match current use cases? 
- Are there songs for which no system estimates a good separation?

---

# Presumed Applications

- Active listening
- Music Education
- Remixing
- Pre-processing for
  - automatic music transcription
  - lyric and music alignment
  - musical instrument detection
  - lyric recognition
  - vocal activity detection
  - fundamental frequency estimation

---

# Actual Commercial applications for Stems

- Social Media
- AR/VR
- Education
- Preprocessing for transcription and alignment
- Karaoke
- Gaming
- Vocal synthesis
- Generative AI

---

# How good would separation need to be  ?

<center>
  <img width="800" src="/applications.png" alt="">
</center>  

---

# Applications

- So if active listening is the most important application, what does it mean to be good enough?

> “A system that achieves human auditory analysis performance in all listening situation” (Wang)

- But for many other applications we might not need to be that good

---

# Arguments why it is solved

- sounds as good ground truth in some cases
- metrics are saturating

---

# Arguments why its not solved

- Perceptible, but not annoying (ODG -1)

---

# Remaining Problems

- what happens with generative models?
- do we still need source separation
- turns out yes

# Stem separation != Source Separation

- Stem separation is a special case of source separation
- For many applications stems are sufficient but they are already mastered
- video one more time

---

# Music Demixing Challenge 2021

---

# Music Demixing Challenge 2023

---

# CV

<img width="1409" src="https://docs.google.com/drawings/d/e/2PACX-1vSi04HMQGxMM-WjT1Ho_2hq4hi-DeqFhXxZ9gfg_9xS752VwzlIhwxVWealZsPwrC4dvv044YtBe2_D/pub?w=1409&h=590">

---
layout: two-cols
---

# Research Interests

<div style="margin-top:2em">

### Machine learning for audio

- Inverse problem: __audio separation__
- Source __count estimation__
- Audio __enhancement__
- Perceptual __losses__ and __evaluation__

</div>

::right::

<p v-click style="margin-top:6em; margin-left: 2em; color: #999999">
  
- $\mathbf{x} = \sum_{j=1}^{k}\mathbf{s}_j$, obtaining $\mathbf{s}_j$ from $\mathbf{x}$
- obtaining $k$ from $\mathbf{x}$
- transfer $\mathbf{x}$ → $\mathbf{y}$
- 👂$(\mathbf{y}, \mathbf{\hat{y}})$

</p>

---

# Challenges for music processing using DL

- Music has __high variance__
- Music has __long-term dependencies__ and have __variable length__
- Evaluation __metrics (perceptual)__ hardly differentiable
- Very __little available data__ (copyright)

<center>
  <img width="650" src="/richter.gif" alt="">
</center>

<div class="caption">Image used courtesy of Jan Van Balen.</div>

---
layout: image
image: '/devons_soft.jpg'
---

# Music Separation

<style>
h1 {
  font-size: 2.5em !important;
  margin-top: 0em;
  margin-left: 0em;
}
</style>

---
layout: image
image: stones_noguitar.jpg
---

---
layout: image
image: stones_noguitar_novocals.jpg
---


---

# Music separation as a machine learning problem

## Generative or discriminative

<img width="800" src="/generative-discriminative.svg" alt="" style="margin-top:2em">

---
layout: image
image: demotrack/mix.jpg
---

<audio controls src="/demotrack/mixture.m4a"></audio>

---
layout: image
image: demotrack/vocals.jpg
---

# Vocals

<audio controls src="/demotrack/vocals.m4a"></audio>

---
layout: image
image: demotrack/drums.jpg
---

# Drums

<audio controls src="/demotrack/drums.m4a"></audio>

---
layout: image
image: demotrack/bass.jpg
---

# Bass

<audio controls src="/demotrack/bass.m4a"></audio>

---

# Basic Separation Architectures

<center>
  <img width="700" src="/encoder_decoder.png" alt="">
</center>

- Encoder here: representation

---

# Basic model: Separating auto-encoder

- __Encoder:__ dense layers the input into a latent vector
- __Decoder:__ dense layers reconstruct the input from the latent vector
- __Complex structures__ can be learned with deeper layers
- Trained to minimize the __reconstruction error__ between input and output

<center>
  <img width="600" src="/autoencoder.png" alt="">
</center>

---

# Basic Encoder: Fixed time frequency representations

<center>
  <img width="700" style="margin-top:5%; float:left" src="/STFT.svg" alt="">
</center>

---

# Training Dataset: MUSDB18

- 100 train / 50 test full tracks
- Mastered with pro. digital audio workstations
- compressed STEMS (`MUSDB18`) and uncompressed WAV `MUSDB18-HQ`
- Parser and Evaluation tools in Python
- Free download on zenodo

<center>
  <img width="500" src="/hero.svg" alt="">
</center>

---

# _Open-unmix_ (UMX) model

<img style="margin-top:2%;" width="800" src="/UMX1.svg" alt="">

---

# _Open-unmix_ (UMX) model

<img style="margin-top:2%;" width="800" src="/UMX2.svg" alt="">

- MIT-licensed Pytorch implementation
- 6.3 dB vocals SDR!

<h6 style="margin-top:4%;color:gray">
  F. Stöter et al, "Open-Unmix - A reference implementation for audio source separation", JOSS 2019.
</h6>

---
layout: iframe
url: https://www.youtube-nocookie.com/embed/9qSkOC7ghgI
---

# Demo: Open-Unmix

---

# <https://open.unmix.app> - edge implementation

<iframe src="https://sigsep.github.io/open-unmix-js/" width="100%" height="680" frameborder="0"></iframe>

---

# Test

<iframe src="https://share.unmix.app/cUIM0xnGx9y92U5onppv/embed" width="100%" height="380" frameborder="0"></iframe>

---

# One more time

<iframe src="https://share.unmix.app/dcYwqyLY9sh3zNlU0HLN/embed" width="100%" height="380" frameborder="0"></iframe>

---
layout: code-right
---

<template v-slot:default>

# Supercharge music separation research

- __MUSDB18__, reference community dataset:
  - 45k downloads
  - top 25 most popular zenodo datasets
- __Open-Unmix__ 806 <uim-star /> on <logos-github />
  - baseline model and training pipline 
- __museval__ reference evaluation toolkit
- __share.unmix.app__ 
  - multitrack player and share platform
- __SiSEC and MDX challenge__ 
  - From _10 to 10k of participants_

</template>
<template v-slot:right>

<div style="padding: 2em; margin-top:0em">

### Data

```python
import musdb
for track in musdb.DB(download=True):
    mix = track.audio
    vocals = track.targets['vocals']
```

### Model

```python
import torch
from torch.hub import load
# load 4-stem model
separator = load('sigsep/open-unmix-pytorch', 'umxl')

estimates = separator(track.audio[None])
# estimates.shape = (1, 4, 2, 100000)
```

### Evaluation

```python
import museval
scores = museval.eval_mus_track(track, estimates)
print(scores)
# ==> SDR:   6.542  SIR:  14.148  ISR:  13.010  SAR:   7.273
```

</div>

</template>

---
layout: image-right
image: /aicrowd.png
---

# Research at Audioshake

- Deploying SOTA Music-ML __respectfully__
- First place in __SONY 2021 MDX challenge__
- Full __AI research life-cycle__: from research to production
  - labeling and dataset design
  - DL architecture design
  - training pipelines
  - perceptual evaluation
  - model optimization for deployment

- We are hiring!

---
layout: code-right
---

# Trends in Music Separation

### Generative Models

- `Diffwave`, `RAVE`, `HIFIGan`
- High-quality waveforms from latent
- 👉 Synthesizers, music enhancement

### Music AI "in-the-wild"

- Training and inference on live audio
- Training on edge devices
- 👉 denoising microphones
- 👉 auto-eq speakers

### transformers

- model long-term relations
- 👉 learn on full tracks

::right::

<img width="700" style="margin-top: 6em" src="/barden.png"><br/>

---

# Outlook - What to work on

- time vs tf models -> tfgridnet
- reducing latency and complexity for live applications
- real stereo models for music
- undo effects
- leaverage long-term relations
- single model vs joint models
- large language models

---

# Sing Song

<video controls>
  <source src="/singsong.mp4" type="video/mp4">
</video>

Donahue, Caillon, Roberts et. al

---

# Movie separation

<video controls>
  <source src="/spiderman.mp4" type="video/mp4">
</video>

Cocktail Fork Problem
Jonathan le Roux, et al.

---
layout: two-cols
---

# Music vs SFX

> - A movie soundtrack typically is a collection of music pieces composed specifically for the film. Its used to complement and enhance the emotional impact of the scenes, create atmosphere and tone, and convey the film's themes and messages.

#

> - Sound effects, are the sounds that are added to a film to create a realistic and immersive environment. Examples of sound effects include the sound of footsteps, a creaking door, or the sound of a car engine.

::right::

<div style="margin-left: 2em; margin-top: 4em">
Music: <audio controls src="/dune.wav"></audio>
Effects: <audio controls src="/drones.wav"></audio>
</div>

<div class="box" style="position: absolute; bottom: 0px; left: 20%">  
  Is this solved yet? NO!
</div>

---

- Soundstream
- AudioLM
- SingSong

---

<style>
table {
  top: -20px;
  font-size: 0.8em;
}
</style>

| Date  | Release                                    | Paper                                            | Code                                                                             | Trained Model                                                                                                                                                       |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 03.04 | [AUDIT: Audio Editing by Following Instructions with Latent Diffusion Models](https://audit-demo.github.io/)                                                                                  | [arXiv](https://arxiv.org/abs/2304.00830)        | -                                                                                | -                                                                                                                                                                   |
| 09.02 | ERNIE-Music: Text-to-Waveform Music Generation with Diffusion Models                                                                                                                           | [arXiv](https://arxiv.org/abs/2302.04456)        | -                                                                                | -                                                                                                                                                                   |
| 08.02 | [Noise2Music: Text-conditioned Music Generation with Diffusion Models](https://google-research.github.io/noise2music/)                                                                         | [arXiv](https://arxiv.org/abs/2302.03917)        | -                                                                                | -                                                                                                                                                                   |
| 04.02 | [Multi-Source Diffusion Models for Simultaneous Music Generation and Separation](https://gladia-research-group.github.io/multi-source-diffusion-models/)                                       | [arXiv](https://arxiv.org/abs/2302.02257)        | [GitHub](https://github.com/gladia-research-group/multi-source-diffusion-models) | -                                                                                                                                                                   |
| 30.01 | [SingSong: Generating musical accompaniments from singing](https://storage.googleapis.com/sing-song/index.html)                                                                                | [arXiv](https://arxiv.org/abs/2301.12662)        | -                                                                                | -                                                                                                                                                                   |
| 30.01 | [AudioLDM: Text-to-Audio Generation with Latent Diffusion Models](https://audioldm.github.io/)                                                                                                 | [arXiv](https://arxiv.org/abs/2301.12503)        | [GitHub](https://github.com/haoheliu/AudioLDM)                                   | [Hugging Face](https://huggingface.co/spaces/haoheliu/audioldm-text-to-audio-generation)                                                                            |
| 30.01 | [Moûsai: Text-to-Music Generation with Long-Context Latent Diffusion](https://anonymous0.notion.site/Mo-sai-Text-to-Audio-with-Long-Context-Latent-Diffusion-b43dbc71caf94b5898f9e8de714ab5dc) | [arXiv](https://arxiv.org/abs/2301.11757)        | [GitHub](https://github.com/archinetai/audio-diffusion-pytorch)                  | -                                                                                                                                                                   |
| 28.01 | [Noise2Music](https://noise2music.github.io/)                                                                                                                                                  | -                                                | -                                                                                | -                                                                                                                                                                   |
| 26.01 | [MusicLM: Generating Music From Text](https://google-research.github.io/seanet/musiclm/examples/)                                                                                              | [arXiv](https://arxiv.org/abs/2301.11325)        | [GitHub (unofficial)](https://github.com/lucidrains/musiclm-pytorch)             | -                                                                                                                                                                   |
| 18.01 | [Msanii: High Fidelity Music Synthesis on a Shoestring Budget](https://kinyugo.github.io/msanii-demo/)                                                                                         | [arXiv](https://arxiv.org/abs/2301.06468)        | [GitHub](https://github.com/Kinyugo/msanii)                                      | [Hugging Face](https://huggingface.co/spaces/kinyugo/msanii) [Colab](https://colab.research.google.com/github/Kinyugo/msanii/blob/main/notebooks/msanii_demo.ipynb) |
| 16.01 | [ArchiSound: Audio Generation with Diffusion](https://flavioschneider.notion.site/Audio-Generation-with-Diffusion-c4f29f39048d4f03a23da13078a44cdb)                                            | [arXiv](https://arxiv.org/abs/2301.13267)        | [GitHub](https://github.com/archinetai/audio-diffusion-pytorch)                  | -                                                                                                                                                                   |
| 05.01 | [VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers](https://valle-demo.github.io/)                                                                                | [arXiv](https://arxiv.org/abs/2301.02111)        | -                                                                                | -                                                                                                                                                                   |
---

# Conclusion

- Music separation is solved for some and some applications
---

- Soundstream
- AudioLM
- SingSong