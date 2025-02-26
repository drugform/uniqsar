# UniQSAR
A framework to train and apply QSAR models.
<!--
![Static Badge](https://img.shields.io/badge/DrugForm-UniQSAR-UniQSAR)
![GitHub top language](https://img.shields.io/github/languages/top/drugform/uniqsar)
![GitHub](https://img.shields.io/github/license/drugform/uniqsar)
![GitHub Repo stars](https://img.shields.io/github/stars/drugform/uniqsar)
![GitHub issues](https://img.shields.io/github/issues/drugform/uniqsar)
-->

## Introduction
This code supports articles published as part of the DrugForm project. 
The UniQSAR framework itself is a general purpose tool for training and further using QSAR models. You are not required to perform any programming work (but you still can if you want).

To train a model you need to prepare a config, describing parameters of model, dataset, neural network, molecule encoder and training procedure. Check existing models for the config examples.

To use a trained model you need only the model name and a csv file with `smiles` column.

<details><summary><b>Citation</b></summary>

  For DTA model:
```bibtex
@article{...}
```
  
For UniQSAR framework:
```bibtex
@article{...}
```
</details>

## Using a trained model
Use `bin/run_predict.sh` to predict values with a trained model. Run the script without args to get help. Basically, you have to provide model name and input file, but there are some more details.
1. -m, --model-name - name of the trained model, as it appears in `models/` directory
2. -i, --input-file - path to the input file. A csv file is expected, with a column `smiles`, containing  SMILES strings of the models you gonna estimate. The file may have other columns, they are ignored.
3. 



