# Generating Explanations for Nonsensical Statements with BART

## Project Overview
This project focuses on the task of generating explanations for nonsensical statements using the BART (Bidirectional and Auto-Regressive Transformers) model. Given a statement that does not make sense, the goal is to produce a coherent explanation of why the statement is nonsensical. This involves fine-tuning the pre-trained BART model on a dataset specifically curated for this task.

## Installation

Before you begin, ensure you have Python 3.6+ installed on your system. To install the required libraries, run the following command:

pip install -r requirements.txt

The `requirements.txt` file includes all necessary Python packages to run this project, including `transformers`, `datasets`, `torch`, and others.

## Dataset

The dataset used for this project comes from the ComVE (Commonsense Validation and Explanation) task of SemEval-2020. It consists of nonsensical statements paired with three possible explanations. The dataset is split into training, validation, and test sets.

## Model

We use the `facebook/bart-large` model from Hugging Face's Transformers library. BART is particularly suited for tasks that require understanding the context and generating text, making it an excellent choice for explaining nonsensical statements.

## Usage

### Data Preparation

First, prepare your dataset by ensuring it's in the correct format. The script `prepare_dataset.py` helps in preprocessing the raw data.

python prepare_dataset.py --input_dir ./data/raw --output_dir ./data/processed

### Model Fine-tuning

To fine-tune the BART model on the prepared dataset, run the `train_model.py` script. This script will train the model and save the best version based on validation performance.

python train_model.py --data_dir ./data/processed --output_dir ./model


### Generating Explanations

Use the fine-tuned model to generate explanations for new nonsensical statements with the `generate_explanations.py` script.

python generate_explanations.py --model_dir ./model --statements "Your new statements here"


## Evaluation

The model's performance is evaluated using BLEU and ROUGE metrics to measure the quality of generated explanations against reference explanations. Run the `evaluate_model.py` script to evaluate the model on the test set.

python evaluate_model.py --model_dir ./model --data_dir ./data/processed


## Contributing

Contributions to improve the project are welcome. Please follow the standard pull request process to submit your contributions.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.
