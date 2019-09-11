# HPSG Neural Parser

This is a Python implementation of the parsers described in "Head-Driven Phrase Structure Grammar Parsing on Penn Treebank" from ACL 2019.

## Contents
1. [Requirements](#Requirements)
2. [Training](#Training)
3. [Citation](#Citation)
4. [Credits](#Credits)

## Requirements

* Python 3.6 or higher.
* Cython 0.25.2 or any compatible version.
* [PyTorch](http://pytorch.org/) 0.4.0. This code has not been tested with PyTorch 1.0, but it should work.
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run `make` inside the `EVALB/` directory to compile an `evalb` executable. This will be called from Python for evaluation. 
* [AllenNLP](http://allennlp.org/) 0.7.0 or any compatible version (only required when using ELMo word representations)
* [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) PyTorch 1.0.0+ or any compatible version (only required when using BERT and XLNet, XLNet only for joint span version.)

#### Pre-trained Models (PyTorch)

The following pre-trained parser models are available for download:
* [`joint_cwt_best_dev=93.85_devuas=95.87_devlas=94.47.pt`](https://drive.google.com/open?id=1ZEMaEQDLRR0-XOCAs_qXtgZo4AOLxaMk): 
Our best English single-system parser based on Glove.
* [`joint_bert_dev=95.55_devuas=96.67_devlas=94.86.pt`](https://drive.google.com/open?id=1TNsJeWVp74iuGINStSfa9z25XwzwHBXX):
Our best English single-system parser based on BERT.
* [`joint_xlnet_dev=96.03_devuas=96.96_devlas=95.32.pt`](https://drive.google.com/open?id=1wF0FoAhG3MarzLrQTz8zTSygMzbZSuNA):
Our best English single-system parser based on XLNet.

The pre-trained model with Glove embeddings obtains 93.78 F-scores of constituent parsing and 96.09 UAS, 94.68 LAS of dependency parsing on the test set. 

The pre-trained model with BERT obtains 95.84 F-scores of constituent parsing and 97.00 UAS, 95.43 LAS of dependency parsing on the test set. 

The pre-trained model with XLNet obtains 96.33 F-scores of constituent parsing and 97.20 UAS, 95.72 LAS of dependency parsing on the test set. 

To use ELMo embeddings, download the following files into the `data/` folder (preserving their names):

* [`elmo_2x4096_512_2048cnn_2xhighway_options.json`](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json)
* [`elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5`](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5)

There is currently no command-line option for configuring the locations/names of the ELMo files.

Pre-trained BERT and XLNet weights will be automatically downloaded as needed by the `pytorch-transformers` package.

## Training

Download the 3 PTB data files from https://github.com/nikitakit/self-attentive-parser/tree/master/data, and put them in the data/ folder.
The dependency structures are mainly obtained by converting constituent structure with version 3.3.0 of [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.html) in the `data/` folder:

```
java -cp stanford-parser_3.3.0.jar edu.stanford.nlp.trees.EnglishGrammaticalStructure -basic -keepPunct -conllx -treeFile 02-21.10way.clean > ptb_train_3.3.0.sd
```

For CTB, we use the same datasets and preprocessing from the [Distance Parser](https://github.com/hantek/distance-parser).
For PTB, we use the same datasets and preprocessing from the [self-attentive-parser](https://github.com/hantek/distance-parser).
[GloVe](https://nlp.stanford.edu/projects/glove) embeddings are optional. 

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
`--eval-batch-size` | Number of examples to process in parallel when evaluating on the development set | 30
`--numpy-seed` | NumPy random seed | Random
`--use-words` | Use learned word embeddings | Do not use word embeddings
`--use-tags` | Use predicted part-of-speech tags as input | Do not use predicted tags
`--use-chars-lstm` | Use learned CharLSTM word representations | Do not use CharLSTM
`--use-elmo` | Use pre-trained ELMo word representations | Do not use ELMo
`--use-bert` | Use pre-trained BERT word representations | Do not use BERT
`--use-xlnet` | Use pre-trained XLNet word representations | Do not use XLNet
`--pad-left` | When using pre-trained XLNet padding on left | Do not pad on left
`--bert-model` | Pre-trained BERT model to use if `--use-bert` is passed | `bert-large-uncased`
`--no-bert-do-lower-case` | Instructs the BERT tokenizer to retain case information (setting should match the BERT model in use) | Perform lowercasing
`--xlnet-model` | Pre-trained XLNet model to use if `--use-xlnet` is passed | `xlnet-large-cased`
`--no-xlnet-do-lower-case` | Instructs the XLNet tokenizer to retain case information (setting should match the XLNet model in use) | Perform uppercasing
`--const-lada` | Lambda weight | 0.5
`--model-name` | Name of model | test
`--embedding-path` | Path to pre-trained embedding | N/A
`--embedding-type` | Pre-trained embedding type | glove
`--dataset`     | Dataset type | ptb


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
to train a Joint-Span parser with XLNet, simply run:
```
sh run_xlnet.sh
```
### Evaluation Instructions

A saved model can be evaluated on a test corpus using the command `python src/main.py test ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base of saved model | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--test-ptb-path` | Path to test constituent parsing | `data/23.auto.clean`
`--dep-test-ptb-path` | Path to test dependency parsing | `data/ptb_test_3.3.0.sd`
`--embedding-path` | Path to pre-trained embedding | `data/glove.6B.100d.txt.gz`
`--eval-batch-size` | Number of examples to process in parallel when evaluating on the test set | 100
`--dataset`     | Dataset type | ptb

As an example, after extracting the pre-trained model, you can evaluate it on the test set using the following command:

```
sh test.sh
```
If you want to parse the sentences, after setting the input file and pre-trained model, run following command:
```
sh parse.sh
```
## Citation
If you use this software for research, please cite our paper as follows:
```
@inproceedings{zhou-zhao-2019-head,
    title = "Head-Driven Phrase Structure Grammar Parsing on {P}enn Treebank",
    author = "Zhou, Junru  and Zhao, Hai",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
}
```

## Credits

The code in this repository and portions of this README are based on https://github.com/nikitakit/self-attentive-parser