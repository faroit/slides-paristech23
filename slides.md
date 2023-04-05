---
# try also 'default' to start simple
theme: default
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: /hero.png
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

# About me

<img width="1409" src="https://docs.google.com/drawings/d/e/2PACX-1vSi04HMQGxMM-WjT1Ho_2hq4hi-DeqFhXxZ9gfg_9xS752VwzlIhwxVWealZsPwrC4dvv044YtBe2_D/pub?w=1409&h=590">

---
layout: image
image: '/stones_all.jpg'
---

# What is music separation

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

# What is music separation

---
layout: image
image: stones_noguitar_novocals.jpg
---

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
transition: slide-up
---

# Bass

<audio controls src="/demotrack/bass.m4a"></audio>

---
transition: slide-down
---

<center>
  <img width="300" src="/IMG_4123.jpg">
</center>

<audio controls style="position: absolute; top: 0; right: 0">
  <source src="/billie.mp3" type="audio/mpeg">
</audio>

---
layout: two-cols
---

# Research topics related to music separation

<div style="margin-top:4em">

### Machine learning for audio

- Inverse problem: __audio separation__
- Source __count estimation__
- Audio __enhancement__
- Perceptual __losses__ and __evaluation__

</div>

::right::

<p v-click style="margin-top:10em; margin-left: 4em; color: #999999">
  
- $\mathbf{x} = \sum_{j=1}^{k}\mathbf{s}_j$, obtaining $\mathbf{s}_j$ from $\mathbf{x}$
- obtaining $k$ from $\mathbf{x}$
- transfer $\mathbf{x}$ → $\mathbf{y}$
- 👂$(\mathbf{y}, \mathbf{\hat{y}})$

</p>

---

# Challenges for music processing

- Music has __high variance__
- Music has __long-term dependencies__ and have __variable length__
- Evaluation __metrics (perceptual)__ hardly differentiable
- Very __little available data__ (copyright)

<center>
  <img width="650" src="/richter.gif" alt="">
</center>

<h6 style="margin-top:4%;color:gray">
  Image used courtesy of Jan Van Balen.
</h6>

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

# Evaluation

- **SDR** (signal-to-distortion ratio) or **log-MSE** is the most common evaluation metric
$$\text{SDR} := 10 \log_{10} \left( \frac{\| s_{\text{target}} \|^2}{ \| e_{\text{interf}} + e_{\text{noise}} + e_{\text{artif}} \|^2} \right)$$

- Evaluation, typically between target and references
- Perceptual metrics exist but are rarely used
- When is it solved?

> “A system that achieves human auditory analysis performance in all listening situation” (Wang)

<h6 style="margin-top:4%;color:gray">
  Emmanuel Vincent, Rémi Gribonval, and Cédric Févotte. Performance measurement in blind audio source separation. IEEE transactions on audio, speech, and language processing, 14(4):1462–1469, 2006
</h6>

---

# Baseline: _Open-unmix_ (UMX) 

<img style="margin-top:2%;" width="800" src="/UMX1.svg" alt="">

---

# Baseline: _Open-unmix_ (UMX) model

<img style="margin-top:2%;" width="800" src="/UMX2.svg" alt="">

- MIT-licensed Pytorch implementation
- SOTA in 2019: 6.3 dB SDR!

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

# What is the progress on music separation research?

<center>
  <img width="800" src="/paperswithcode.png" alt="">
</center>

1. Does scientific research align with real-world applications?
2. Do datasets and metrics match current use cases?

---

# Presumed applications of music separation

- Active listening
- Music Education
- Remixing
- Pre-processing for MIR Tasks
  - musical instrument detection
  - vocal activity detection
  - lyric recognition
  - fundamental frequency estimation

---

# Actual Commercial applications

- Social Media
- AR/VR
- Education
- Preprocessing for transcription and alignment
- Karaoke
- Gaming
- Vocal synthesis
- Generative AI
- Sync/Dubbing for Film

---

# How good would separation need to be?

<center>
  <img width="800" src="/applications.png" alt="">
</center>  

---

<iframe src="https://share.unmix.app/cUIM0xnGx9y92U5onppv/embed" width="100%" height="340" frameborder="0"></iframe>

- sounds as good ground truth in some cases
- metrics are saturating: we need new metrics

---

# Proposal: Clean-Source-Separation

<center>
  <img width="800" src="/cleantargetsep.png" alt="">
</center>

---

# Issues with MUSDB18

- too small – are we overfitting on MUSDB18?
- yes:

<center>
  <img src="/loudness.png">
</center>

---

# Stem separation != Source Separation

- Stem separation is a special case of source separation
- For many applications stems are sufficient but they are already mastered

<div v-click style="margin-top:1em">
  <iframe src="https://share.unmix.app/dcYwqyLY9sh3zNlU0HLN/embed" width="100%" height="246" frameborder="0"></iframe>
</div>

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

<img src="/mdx23.png" width="700">

- Signal Separation Evaluation Campaign (SiSEC) 
  - Began in 2007, separating speech and music signals
  - MusDB18 released in 2017 provided an open training dataset for deep learning-based music source separation
  - Four output signals: Vocals, Bass, Drums, Other

- 2021 Music Demixing (MDX) Challenge
  - Direct Predecessor to SDX 2023
  - Evaluation dataset is hidden from participants

---
layout: two-cols
---

# Trends/Outlook

### Generative Models

- High-quality waveforms from latent
  - `Diffwave`, `RAVE`, `HIFIGan`
- time vs tf models :fight:
- Separation with text prompts

::right::

<div style="margin-top: 5.4em"></div>

### Music AI "in-the-wild"

- Training and inference on live audio
- Inference on edge devices
- Real-world metrics
- 👉 undo effects
- 👉 real stereo models for music

### transformers

- model long-term relations
- AudioLMs for larger context
- 👉 learn on full tracks

---
layout: two-cols
---
# Generative Audio models based on LLMs

- Core block: Soundstream
- AudioLM
- SingSong

::right::

<video controls>
  <source src="/singsong.mp4" type="video/mp4">
</video>

Donahue, Caillon, Roberts et. al|

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

- We are hiring and looking for interns!

---
layout: iframe
url: https://www.youtube-nocookie.com/embed/O19u51JSoQI
---

# Conclusion

- Deep learning just got us to the point where we can start to solve the problem
- Music separation is solved for some kind of music and some applications
- Music generation is still a hard problem, and requires source separation for training
- To actually solve the task, we must define success criteria choose the appropriate metrics for the use case, and — if not available—curate suitable datasets.
