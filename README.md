# Empathy Prediction using LSTM Networks

This repository contains the code and related resources for the study on empathy prediction using Long Short-Term Memory (LSTM) networks. The study uses eye-tracking data to predict empathy scores, demonstrating the efficacy of LSTM networks in this context.

## Overview

The proposed architecture includes a dropout layer, an LSTM layer, and a pair of linear layers, which together enhance the model's overall performance on both the training and testing data sets. The dropout layer serves as a regularization technique, effectively mitigating the risk of overfitting. The LSTM layer manages the intricate time-dependent associations embedded within the Gaze point X data, derived from eye-tracking. The final two linear layers equip the model with the ability to comprehend high-level abstractions and to generate a precise, continuous empathy score.

## Requirements

The following Python packages are required to run the code in this repository:

- pandas
- numpy
- os
- warnings
- matplotlib
- sklearn
- torch

You can install these packages using pip:

```bash
pip install pandas numpy matplotlib sklearn torch
```

## Usage

The Python script is written in a Jupyter notebook. After cloning this repository and installing the required packages, you can run the Jupyter notebook as follows:

```bash
jupyter notebook main.ipynb
```

The dataset used in this project is located in the `dataset` directory. Please ensure that your data files are in the correct directory and that you have adjusted any file paths in the script as necessary.

## Future Work

Future research could explore the integration of additional data sources, such as facial expressions or physiological signals, to further enhance the model's predictive capabilities. Detailed fine-tuning of the model and investigation of alternative architectures and optimization techniques may also yield more accurate and efficient models for empathy prediction.

## Contributing

Contributions are welcome!

## License

This project is licensed under the terms of the MIT license.
