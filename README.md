# Health Data Statistical Analysis

## Project Overview
This project demonstrates a Python-based pipeline for generating, processing, and analyzing synthetic medical data. The goal is to explore the statistical relationships between **aging**, **physical activity**, and **cognitive health outcomes**.

This script showcases my ability to:
1. **Handle Data:** Using `pandas` and `numpy` to structure high-dimensional datasets.
2. **Apply Statistics:** Utilizing correlation matrices and Linear Regression to quantify relationships.
3. **Ensure Reproducibility:** Setting random seeds to ensure the analysis produces consistent results, a key requirement in scientific research.

## Methodology
The script generates a synthetic dataset of 200 patients with the following variables:
* **Age:** (50-90 years)
* **Physical Activity Index:** Modeled to have an inverse relationship with age.
* **Cognitive Test Score:** Modeled to be positively correlated with physical activity.

A **Linear Regression Model** (`sklearn`) is then trained to predict Cognitive Scores based on Age and Activity levels.

## Results
The model outputs the **R-squared (R²)** value to evaluate how well the independent variables explain the variance in cognitive scores, demonstrating fundamental predictive modeling techniques applicable to translational research.

## Technologies Used
* Python 3.x
* Pandas / NumPy
* Scikit-Learn (Machine Learning)
