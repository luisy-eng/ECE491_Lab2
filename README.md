# ECE 491E Lab 2 — Adversarial Example Generation

This project investigates adversarial attacks against a machine learning classifier using the **Fast Gradient Sign Method (FGSM)**. The experiments use a pretrained PyTorch neural network and the **MNIST handwritten digit dataset**.

The lab consists of three main tasks:

1. Implement and evaluate an untargeted FGSM attack.
2. Modify FGSM to perform a targeted attack toward class `5`.
3. Evaluate preprocessing defenses against the targeted adversarial attack.

The project was implemented using **Python, PyTorch, and Google Colab**. The final report was written in LaTeX using the **NeurIPS 2022 format**.

---

## Project Objectives

The objectives of this lab are to:

- Understand how FGSM generates adversarial examples.
- Evaluate how perturbation strength affects model performance.
- Implement an untargeted adversarial attack.
- Modify FGSM into a targeted attack.
- Force MNIST samples toward a selected target class.
- Evaluate denoising-based defenses.
- Evaluate Gaussian-noise defenses.
- Compare defenses using targeted attack success rate.
- Visualize the effects of attacks and defenses.

---

## Repository Structure

```text
ECE491_Lab2/
│
├── Task1/
│   └── fgsm_tutorial.ipynb
│
├── Task2/
│   └── targeted_fgsm.ipynb
│
├── Task3/
│   └── fgsm_defenses.ipynb
│
│
└── README.md
```

---

# Task 1 — Untargeted FGSM Attack

Task 1 reproduces the PyTorch FGSM adversarial-example demonstration using the MNIST handwritten digit dataset.

The Fast Gradient Sign Method generates an adversarial example using:

```math
x_{\mathrm{adv}}
=
x +
\epsilon
\operatorname{sign}
\left(
\nabla_x J(\theta,x,y)
\right)
```

where:

- `x` is the original input image.
- `y` is the correct class label.
- `θ` represents the neural network parameters.
- `J` is the loss function.
- `ε` controls the magnitude of the perturbation.

FGSM modifies the input in a direction that increases the classification loss.

The following perturbation strengths were evaluated:

```text
ε = 0.00
ε = 0.05
ε = 0.10
ε = 0.15
ε = 0.20
ε = 0.25
ε = 0.30
```

As `ε` increases, the adversarial perturbation becomes stronger and the classifier becomes increasingly likely to produce an incorrect prediction.

### Notebook

[Task 1 — FGSM Tutorial](Task1/fgsm_tutorial.ipynb)

---

# Task 2 — Targeted FGSM Attack

Task 2 modifies the standard FGSM attack into a **targeted adversarial attack**.

Instead of simply causing the classifier to make any incorrect prediction, the objective is to force the classifier toward one specific target class.

For this experiment:

```text
Target Class = 5
```

The targeted FGSM attack is:

```math
x_{\mathrm{adv}}
=
x -
\epsilon
\operatorname{sign}
\left(
\nabla_x J(\theta,x,y_{\mathrm{target}})
\right)
```

The perturbation direction is reversed because the objective is to reduce the loss associated with the selected target class.

Samples whose original class was already `5` were excluded from the targeted attack evaluation.

---

## Targeted FGSM Results

A total of **8,965 correctly classified MNIST test samples** were eligible for the targeted attack.

| Epsilon | Successful Attacks | Targeted Attack Success Rate |
|---:|---:|---:|
| 0.00 | 0 / 8965 | 0.00% |
| 0.05 | 40 / 8965 | 0.45% |
| 0.10 | 184 / 8965 | 2.05% |
| 0.15 | 535 / 8965 | 5.97% |
| 0.20 | 1258 / 8965 | 14.03% |
| 0.25 | 2355 / 8965 | 26.27% |
| 0.30 | 3510 / 8965 | **39.15%** |

The targeted attack became increasingly effective as the perturbation magnitude increased.

At `ε = 0.30`, the attack successfully forced **3,510 of 8,965 samples** into the target class, corresponding to a targeted attack success rate of **39.15%**.

### Notebook

[Task 2 — Targeted FGSM](Task2/targeted_fgsm.ipynb)

---

# Task 3 — Adversarial Defenses

Task 3 investigates whether simple image-preprocessing techniques can reduce the effectiveness of the targeted FGSM attack.

The defense experiments were performed using:

```text
FGSM Epsilon = 0.30
Target Class = 5
```

The undefended targeted attack success rate was:

```text
3510 / 8965 = 39.15%
```

Two categories of defenses were evaluated:

1. Denoising / smoothing defenses
2. Gaussian-noise defense

---

## Denoising Defenses

Three image-processing methods were tested:

- Gaussian blur
- Mean filtering
- Median filtering

Each defense was applied to the adversarial image before it was passed back into the classifier.

### Denoising Results

| Defense Method | Successful Attacks | Attack Success Rate |
|---|---:|---:|
| No Defense | 3510 / 8965 | 39.15% |
| Gaussian Blur | 2420 / 8965 | 26.99% |
| Mean Filter | 2160 / 8965 | **24.09%** |
| Median Filter | 2243 / 8965 | 25.02% |

All three denoising methods reduced the targeted FGSM attack success rate.

The **mean filter produced the best result**, reducing the targeted attack success rate from:

```text
39.15% → 24.09%
```

This corresponds to a reduction of **15.06 percentage points** compared with the undefended targeted attack.

---

## Gaussian Noise Defense

The second defense strategy adds random Gaussian noise to the adversarial image.

The defended image is represented as:

```math
x_{\mathrm{def}}
=
x_{\mathrm{adv}} + n
```

where:

```math
n \sim \mathcal{N}(0,\sigma^2)
```

and `σ` controls the magnitude of the added random noise.

Several noise levels were evaluated.

### Gaussian Noise Results

| Noise Standard Deviation | Successful Attacks | Attack Success Rate |
|---:|---:|---:|
| 0.00 | 3510 / 8965 | 39.15% |
| 0.02 | 3473 / 8965 | 38.74% |
| 0.05 | 3407 / 8965 | 38.00% |
| 0.10 | 3238 / 8965 | 36.12% |
| 0.15 | 3049 / 8965 | 34.01% |
| 0.20 | 2861 / 8965 | 31.91% |

Increasing the amount of Gaussian noise consistently reduced the targeted attack success rate.

However, Gaussian-noise injection was less effective than the tested denoising filters.

---

## Visual Defense Example

A successful targeted FGSM example was also used to visualize how each defense affected the image and model prediction.

For the selected sample:

```text
Original Image
True Class: 0
Prediction: 0

Targeted FGSM Adversarial Image
Prediction: 5

Gaussian Blur
Prediction: 0

Mean Filter
Prediction: 0

Median Filter
Prediction: 6

Gaussian Noise
Prediction: 5
```

For this example:

- **Gaussian blur** defeated the targeted attack and restored the correct class `0`.
- **Mean filtering** defeated the targeted attack and restored the correct class `0`.
- **Median filtering** defeated the targeted attack because the prediction changed away from class `5`, but it did not restore the correct class.
- **Gaussian noise** did not defeat the targeted attack for this sample because the prediction remained class `5`.

This demonstrates an important distinction between **defeating a targeted attack** and **recovering the original correct classification**.

---

# Overall Results

The experiments produced several important observations:

- Neural networks can be vulnerable to small gradient-based input perturbations.
- Increasing FGSM perturbation strength increases attack effectiveness.
- Targeted attacks are more restrictive than untargeted attacks because the adversarial input must be moved toward one specific class.
- The targeted FGSM attack achieved a maximum tested success rate of **39.15%** at `ε = 0.30`.
- All tested preprocessing defenses reduced targeted attack success.
- Mean filtering was the strongest tested defense.
- Gaussian-noise injection also reduced attack success, but less effectively than the denoising filters.
- A defense can defeat the target-class prediction without necessarily restoring the original correct classification.
- Strong preprocessing can improve adversarial robustness while also modifying useful image information.

---

## Defense Comparison

| Method | Targeted Attack Success Rate |
|---|---:|
| No Defense | 39.15% |
| Gaussian Blur | 26.99% |
| **Mean Filter** | **24.09%** |
| Median Filter | 25.02% |
| Gaussian Noise (`σ = 0.20`) | 31.91% |

Among the tested defense configurations, the **mean filter achieved the lowest targeted attack success rate at 24.09%**.

---

# Technologies Used

- Python
- PyTorch
- Torchvision
- MNIST
- NumPy
- Matplotlib
- Pandas
- Google Colab
- LaTeX
- NeurIPS 2022 LaTeX Template
- Git
- GitHub

---

# Running the Notebooks

The experiments were developed using **Google Colab**.

Each task can be viewed and run independently from its corresponding notebook.

### Task 1

[Open Task 1 Notebook](Task1/fgsm_tutorial.ipynb)

### Task 2

[Open Task 2 Notebook](Task2/targeted_fgsm.ipynb)

### Task 3

[Open Task 3 Notebook](Task3/fgsm_defenses.ipynb)

The main Python libraries used are:

```python
torch
torchvision
numpy
matplotlib
pandas
```

---

# Report

The complete report contains:

- Introduction to adversarial examples
- FGSM methodology
- Untargeted FGSM experiments
- Targeted FGSM implementation
- Targeted attack success-rate analysis
- Denoising defense experiments
- Gaussian-noise defense experiments
- Experimental tables
- Adversarial-example visualizations
- Defense visualizations
- Overall discussion
- Conclusion
- References

### LaTeX Source

[View Report LaTeX Source](Report/main.tex)

### References

[View references.bib](Report/references.bib)

The report follows the **NeurIPS 2022 LaTeX format**.

If the compiled PDF is placed in the `Report` folder, add:

```markdown
### Full Report

[View Full Lab Report (PDF)](Report/ECE491_Lab2_Report.pdf)
```

---

# References

The project uses and references the following sources:

1. **Goodfellow, I. J., Shlens, J., and Szegedy, C.**  
   *Explaining and Harnessing Adversarial Examples.*

2. **PyTorch.**  
   *Adversarial Example Generation.*

3. **LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P.**  
   *Gradient-Based Learning Applied to Document Recognition.*

4. **Gonzalez, R. C., and Woods, R. E.**  
   *Digital Image Processing.*

Full BibTeX citation information is available in:

[Report/references.bib](Report/references.bib)

---

# Repository

**ECE 491E Lab 2 — Adversarial Example Generation**

https://github.com/luisy-eng/ECE491_Lab2

---

## Author

**Luis Hernandez**  
Electrical and Computer Engineering  
University of Hawaiʻi at Mānoa
