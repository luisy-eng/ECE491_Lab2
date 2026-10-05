# ECE 491E Lab 2 — Adversarial Example Generation

This project explores adversarial attacks against a neural network classifier using the **Fast Gradient Sign Method (FGSM)**. The experiments use a pretrained PyTorch model and the **MNIST handwritten digit dataset**.

The lab consists of three main tasks:

1. Implement and evaluate an untargeted FGSM attack.
2. Modify FGSM to perform a targeted attack toward class `5`.
3. Evaluate several preprocessing defenses against the targeted attack.

The project was implemented in **Python using PyTorch and Google Colab**. A full report was written in LaTeX using the **NeurIPS 2022 format**.

---

## Project Objectives

The main objectives of this lab are to:

- Understand how FGSM generates adversarial examples.
- Observe how perturbation strength affects neural network predictions.
- Implement a targeted adversarial attack.
- Force MNIST samples toward a selected target class.
- Evaluate denoising-based defenses.
- Evaluate random-noise defenses.
- Compare defense effectiveness using attack success rate.
- Visualize the effects of adversarial attacks and defenses.

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
