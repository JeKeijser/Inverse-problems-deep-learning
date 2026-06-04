# Inverse Problems Using Deep Learning

This project investigates whether deep learning can serve as an effective tool for solving inverse problems in a simulated microscopy setting. Three neural network architectures are compared: a Multi-Layer Perceptron (MLP), a Convolutional Neural Network (CNN), and a Physics-Informed Neural Network (PINN).

This project was developed as part of the course Advanced Machine Learning with Neural Networks (TIF360) at Chalmers University of Technology, 2026. The simulation pipeline was provided by the project supervisor, Daniel Midtvedt.

## Project Structure

- simulate.py — Simulation pipeline for generating the dataset
- models.py — MLP, CNN and PINN architectures
- dataset.py — Data loading and preprocessing
- train.py — Training script for individual models
- evaluate.py — Evaluation script
- main.py — Runs the full pipeline with a single command
- plot_losses.py — Plots training and validation loss curves

## How to Run

The full pipeline can be run with a single command:

    python main.py --all

Or run individual steps:

    python main.py --generate   # Generate the dataset
    python main.py --train      # Train all models
    python main.py --evaluate   # Evaluate all models

You can also set the number of training epochs:

    python main.py --all --epochs 100

## Note on Data

The simulated dataset is not included in this repository due to its size. Running the pipeline with --generate will regenerate it locally.