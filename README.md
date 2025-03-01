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

## Usage
### Setting environment
Use the provided Dockerfile to build an environment to run prediction and training: `docker build -t Dockerfile .` 

Of course you can use UniQSAR without docker, in that case you'll need to install the libraries from the Dockerfile manually, and also crop calling the python code from `bin/` scripts. We strongly advice to use our dockerfile, it will save you some time on combining torch verions. For example, Chemformer do not work with pytorch 2.x, while some other libs like ESM require pytorch 2.x unless you tell them manually not to use it.

### Obtaining the libraries
Curently the project uses Chemformer to encode small molecules and ESM to encode amino acid sequences. Their code is not included in this repository.

The ESM code and model will be downloaded and cached automatically after the first run, you do not need to do a thing about it. If something went wrong or running the code indicates problems with the ESM model, delete the `lib/esm` directory and try again. Finally it must contain esm repository content in `esm/lib/facebookresearch_esm_main` and `esm/lib/checkpoints/esm2_t30_150M_UR50D.pt` file. 

The Chemformer code will also be downloaded and cached at the first run, checkouting the specific commit in the repo, so the future changes in the 3rd-party repository do not affect the compartibility. Unfortunately, Chemformer model files cannot be downloaded automatically, you have to download it manually from the web page, given in their work: https://az.app.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq. Download the `models/pre-trained/combined/step=1000000.ckpt` file from there and put it to `lib/Chemformer/models/combined.ckpt`. If you need the `combined_large` model, download the `models/pre-trained/combined-large/step=1000000.ckpt` file from there and put it to `lib/Chemformer/models/combined_large.ckpt`. The models are converted into their new format automatically, you do not have to do it manually.

So basically you need to put by hands only the `combined.pt` file.

### Obtaining the data
Trained models and datasets are large files. Github cannot handle it even with the LFS module installed. Before first use, download the archive `drugform-dta_data.tar.gz` from here: `https://zenodo.org/uploads/14949570` (doi: 10.5281/zenodo.14949570). Directory structure inside the archive matches the code structure. Full command list to merge the code and the data is the following:

```
$ git clone https://github.com/drugform/uniqsar
### until the article is not puclished, this github repo is private, so use an early access token to clone the repo: 
$ git clone https://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@github.com/drugform/uniqsar
$ tar xvf drugform-dta_data.tar.gz
$ cp -r drugform-dta_data/uniqsar/* uniqsar/
$ rm -r drugform-dta_data
```

### Using a trained model
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

### Training own QSAR model.
You can train own models with `bin/run_train.sh`. Run the script without args to get help. For example, to train a DTA model on the KIBA benchmark, you need to run the command:

`bin/run_train.sh --config-file train_scripts/kiba.py --gpus 0,1`

It will run the training in a docker container, and give you its id, so you can later check the progress with `docker logs`. If `--gpus` argument not given, it will run the training on the CPU. Most likely, you do not want this, unless you need to debug something. Training a real-life model on CPU will take almost a hole life time.

Use `--dev-mode` key argument to run an interactive docker session (probably you do not need that too).

The config file may be at any path in the file system, but you are advised to put the config files to the `train_scripts` directory. The directory contains other train scripts, use them as examples to make your own one.

### Training on own dataset
