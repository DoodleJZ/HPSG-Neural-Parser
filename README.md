# HPSG Neural Parser

This is a Python implementation of the parsers described in "Head-Driven Phrase Structure Grammar Parsing on Penn Treebank" from ACL 2019.

## Contents
1. [Requirements](#Requirements)
2. [Training](#training)
3. [Reproducing Experiments](#reproducing-experiments)
4. [Citation](#citation)
5. [Credits](#credits)

If you are primarily interested in training your own parsing models, skip to the [Training](#training) section of this README.

## Requirements

* Python 3.6 or higher.
* Cython 0.25.2 or any compatible version.
* [PyTorch](http://pytorch.org/) 0.4.1. This code has not been tested with PyTorch 1.0, but it should work.
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run `make` inside the `EVALB/` directory to compile an `evalb` executable. This will be called from Python for evaluation. If training on the SPMRL datasets, you will need to run `make` inside the `EVALB_SPMRL/` directory instead.
* [AllenNLP](http://allennlp.org/) 0.7.0 or any compatible version (only required when using ELMo word representations)
* [pytorch-pretrained-bert](https://github.com/huggingface/pytorch-pretrained-BERT) 0.4.0 or any compatible version (only required when using BERT word representations)

#### Pre-trained Models (PyTorch)

The following pre-trained parser models are available for download:
* [`en_charlstm_dev.93.61.pt`](https://github.com/nikitakit/self-attentive-parser/releases/download/models/en_charlstm_dev.93.61.pt): 
Our best English single-system parser based on Glove.
* [`en_elmo_dev.95.21.pt`](https://github.com/nikitakit/self-attentive-parser/releases/download/models/en_elmo_dev.95.21.pt): The best English single-system parser from our ACL 2018 paper. Using this parser requires ELMo weights, which must be downloaded separately.

To use ELMo embeddings, download the following files into the `data/` folder (preserving their names):

* [`elmo_2x4096_512_2048cnn_2xhighway_options.json`](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json)
* [`elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5`](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5)

There is currently no command-line option for configuring the locations/names of the ELMo files.

Pre-trained BERT weights will be automatically downloaded as needed by the `pytorch-pretrained-bert` package.

## Training

The code used to train our parsing models is currently different from the code used to parse sentences in the release version described above, though both are stored in this repository. The training code uses PyTorch and can be obtained by cloning this repository from GitHub. 

### Training Instructions

Some of the available arguments are:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base to use for saving models | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
` --train-ptb-path` | Path to training constituent parsing | `data/02-21.10way.clean`
`--dev-ptb-path` | Path to development constituent parsing | `data/22.auto.clean`
`--dep-train-ptb-path` | Path to training dependency parsing | `data/ptb_train_3.3.0.sd`
`--dep-dev-ptb-path` | Path to development dependency parsing | `data/ptb_dev_3.3.0.sd`
`--batch-size` | Number of examples per training update | 250
`--checks-per-epoch` | Number of development evaluations per epoch | 4
`--subbatch-max-tokens` | Maximum number of words to process in parallel while training (a full batch may not fit in GPU memory) | 2000
`--eval-batch-size` | Number of examples to process in parallel when evaluating on the development set | 100
`--numpy-seed` | NumPy random seed | Random
`--use-words` | Use learned word embeddings | Do not use word embeddings
`--use-tags` | Use predicted part-of-speech tags as input | Do not use predicted tags
`--use-chars-lstm` | Use learned CharLSTM word representations | Do not use CharLSTM
`--use-elmo` | Use pre-trained ELMo word representations | Do not use ELMo
`--use-bert` | Use pre-trained BERT word representations | Do not use BERT
`--bert-model` | Pre-trained BERT model to use if `--use-bert` is passed | `bert-base-uncased`
`--no-bert-do-lower-case` | Instructs the BERT tokenizer to retain case information (setting should match the BERT model in use) | Perform lowercasing
`--const-lada` | Number of lambda weight | 0.5
`--embedding-type` | Pre-trained embedding type | glove


Additional arguments are available for other hyperparameters; see `make_hparams()` in `src/main.py`. These can be specified on the command line, such as `--num-layers 2` (for numerical parameters), `--use-tags` (for boolean parameters that default to False), or `--no-partitioned` (for boolean parameters that default to True).

For each development evaluation, the best_dev_score is the sum of F-score and LAS on the development set and compared to the previous best. If the current model is better, the previous model will be deleted and the current model will be saved. The new filename will be derived from the provided model path base and the development best_dev_score.

As an example, after setting the paths for data and embeddings,
to train a Joint-Span parser, simply run:
```
sh run_single.sh
```

to train a Joint-Span parser with BERT, simply run:
```
sh run_bert.sh
```
### Evaluation Instructions

A saved model can be evaluated on a test corpus using the command `python src/main.py test ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base of saved model | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--test-path` | Path to test trees | `data/23.auto.clean`
`--test-path-raw` | Alternative path to test trees that is used for evalb only (used to double-check that evaluation against pre-processed trees does not contain any bugs) | Compare to trees from `--test-path`
`--eval-batch-size` | Number of examples to process in parallel when evaluating on the test set | 100

If the parser was trained to have predicted part-of-speech tags as input (via the `--use-tags` flag) the test trees are assumed to have predicted part-of-speech tags. Otherwise, the tags in the test trees are not used as input to the parser.

As an example, after extracting the pre-trained model, you can evaluate it on the test set using the following command:

```
sh test.sh
```

## Citation

If you use this software for research, please cite our paper as follows:

```

```

## Credits

The code in this repository and portions of this README are based on https://github.com/nikitakit/self-attentive-parser