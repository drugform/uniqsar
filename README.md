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

## Setting Environment
Use the provided Dockerfile to build an environment to run prediction and training: `docker build -t Dockerfile .` 
Of course you can use UniQSAR without docker, in that case you'll need to install the libraries from the Dockerfile manually, and also crop calling the python code from `bin/` scripts. We strongly advice to use our dockerfile, it will save you some time on combiling torch verions. For example, Chemformer do not work with pytorch 2.x, while some other libs like ESM require pytorch 2.x unless you tell them manually not to use it.

## Obtaining the Data
Trained models, external libs like Chemformer, and datasets are large files. Github cannot handle it even with the LFS module installed. Before first use, download the files from here: `url link`. Directory structure inside the archive matches the code structure, so just unpack it in the root dir: `command here`.

## Using a Trained Model
Use `bin/run_predict.sh` to predict values with a trained model. Run the script without args to get help. Basically, you have to provide model name and input file, but there are some more details.

1. `-m, --model-name` — name of the trained model, as it appears in `models/` directory
2. `-i, --input-file` — path to the input file. A csv file is expected, with a column `smiles`, containing  SMILES strings of the models you gonna estimate. The file may have other columns, they are ignored.
3. `-b, --batch-size` — simply the batch size, it depends on how big is your GPU memory. Some models like the DTA model is pretty big, with weak hardware consider batch size = 1.
4. `-o, --output-file` — path where to put the file with predictions. If the file already exists, the script will ask a permission to delete it, so you will not get it overwritten without a notice.
5. `-g, --gpus` — a comma-separated list of NVidia card ids. Set 0 if you have only a single card. If a list of several cards is given, it will share the batch between them with torch.DataParallel mechanism. Optional parameter, it will run on CPU if the parameter not set.
6. `-d, --dev-mode` — optional parameter, just a key with no args. If set, it will rebuild the docker image and run an interactive session. Useful for debugging, in general you do not need this.

An example of running a prediciton with the DTA model:

`bin/run_predict.sh --batch-size 16 --model-name bindingdb --input-file data/bindingdb/bindingdb_test_tiny.csv --output-file /tmp/output_test.csv --gpus 0,1`

It will return an id of a docker container, where the prediction is running. You can check the progress with `docker logs $id$` command. After it finishes, check `/tmp/output_test.csv` for prediction results.

