import argparse
import itertools
import os.path
import time
import uuid

import torch
import torch.optim.lr_scheduler

import numpy as np
import math
import evaluate
import trees
import vocabulary
import makehp
import Zparser
from dep_reader import CoNLLXReader
import dep_eval
import utils
import json
tokens = Zparser

uid = uuid.uuid4().hex[:6]

def torch_load(load_path):
    if Zparser.use_cuda:
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location=lambda storage, location: storage)

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def make_hparams():
    return makehp.HParams(
        max_len_train=0, # no length limit
        max_len_dev=0, # no length limit

        sentence_max_len=300,

        learning_rate=0.0008,
        learning_rate_warmup_steps=160,
        clip_grad_norm=0., #no clipping
        step_decay=True, # note that disabling step decay is not implemented
        step_decay_factor=0.5,
        step_decay_patience=5,

        partitioned=True,

        use_cat = False,
        num_layers=12,
        d_model=1024,
        num_heads=8,
        d_kv=64,
        d_ff=2048,
        d_label_hidden=250,

        attention_dropout=0.2,
        embedding_dropout=0.2,
        relu_dropout=0.2,
        residual_dropout=0.2,

        use_tags=False,
        use_words=False,
        use_elmo = False,
        use_bert=False,
        use_bert_only=False,
        use_chars_lstm=False,

        dataset = 'ptb',

        model_name = "dep+const",
        embedding_type = 'random',
        #['glove','sskip','random']
        embedding_path = "/data/glove.gz",
        punctuation='.' '``' "''" ':' ',',

        d_char_emb = 64, # A larger value may be better for use_chars_lstm

        tag_emb_dropout=0.2,
        word_emb_dropout=0.4,
        morpho_emb_dropout=0.2,
        timing_dropout=0.0,
        char_lstm_input_dropout=0.2,
        elmo_dropout=0.5, # Note that this semi-stacks with morpho_emb_dropout!

        bert_model="bert-large-uncased",
        bert_do_lower_case=True,
        bert_transliterate="",
        )


def run_train(args, hparams):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    # Make sure that pytorch is actually being initialized randomly.
    # On my cluster I was getting highly correlated results from multiple
    # runs, but calling reset_parameters() changed that. A brief look at the
    # pytorch source code revealed that pytorch initializes its RNG by
    # calling std::random_device, which according to the C++ spec is allowed
    # to be deterministic.
    seed_from_numpy = np.random.randint(2147483648)
    print("Manual seed for pytorch:", seed_from_numpy)
    torch.manual_seed(seed_from_numpy)

    hparams.set_from_args(args)
    print("Hyperparameters:")
    hparams.print()

    train_path = args.train_ptb_path
    dev_path = args.dev_ptb_path

    dep_train_path = args.dep_train_ptb_path
    dep_dev_path = args.dep_dev_ptb_path

    if hparams.dataset == 'ctb':
        train_path = args.train_ctb_path
        dev_path = args.dev_ctb_path

        dep_train_path = args.dep_train_ctb_path
        dep_dev_path = args.dep_dev_ctb_path

    dep_reader = CoNLLXReader(dep_train_path)
    print('Reading dependency parsing data from %s' % dep_train_path)

    dep_dev_reader = CoNLLXReader(dep_dev_path)
    print('Reading dependency parsing data from %s' % dep_dev_path)

    counter = 0
    dep_sentences = []
    dep_data = []
    dep_heads = []
    dep_types = []
    inst = dep_reader.getNext()
    while inst is not None:

        inst_size = inst.length()
        if hparams.max_len_train > 0 and inst_size - 1 > hparams.max_len_train:
            inst = dep_reader.getNext()
            continue

        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)
        sent = inst.sentence
        dep_data.append((sent.words, inst.postags, inst.heads, inst.types))
        #dep_sentences.append([(tag, word) for i, (word, tag) in enumerate(zip(sent.words, sent.postags))])
        dep_sentences.append(sent.words)
        dep_heads.append(inst.heads)
        dep_types.append(inst.types)
        inst = dep_reader.getNext()
    dep_reader.close()
    print("Total number of data: %d" % counter)

    dep_dev_data = []
    dev_inst = dep_dev_reader.getNext()
    dep_dev_headid = np.zeros([3000,300],dtype=int)
    dep_dev_type = []
    dep_dev_word = []
    dep_dev_pos = []
    dep_dev_lengs = np.zeros(3000, dtype=int)
    cun = 0
    while dev_inst is not None:
        inst_size = dev_inst.length()
        if hparams.max_len_dev > 0 and inst_size - 1> hparams.max_len_dev:
            dev_inst = dep_dev_reader.getNext()
            continue
        dep_dev_lengs[cun] = inst_size
        sent = dev_inst.sentence
        dep_dev_data.append((sent.words, dev_inst.postags, dev_inst.heads, dev_inst.types))
        for i in range(inst_size):
            dep_dev_headid[cun][i] = dev_inst.heads[i]

        dep_dev_type.append(dev_inst.types)
        dep_dev_word.append(sent.words)
        dep_dev_pos.append(sent.postags)
        #dep_sentences.append([(tag, word) for i, (word, tag) in enumerate(zip(sent.words, sent.postags))])
        dev_inst = dep_dev_reader.getNext()
        cun = cun + 1
    dep_dev_reader.close()

    print("Loading training trees from {}...".format(train_path))
    train_treebank = trees.load_trees(train_path, dep_heads, dep_types)
    if hparams.max_len_train > 0:
        train_treebank = [tree for tree in train_treebank if len(list(tree.leaves())) <= hparams.max_len_train]
    print("Loaded {:,} training examples.".format(len(train_treebank)))

    print("Loading development trees from {}...".format(dev_path))
    dev_treebank = trees.load_trees(dev_path, dep_dev_headid, dep_dev_type)
    if hparams.max_len_dev > 0:
        dev_treebank = [tree for tree in dev_treebank if len(list(tree.leaves())) <= hparams.max_len_dev]
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    print("Processing trees for training...")
    train_parse = [tree.convert() for tree in train_treebank]

    print("Constructing vocabularies...")

    tag_vocab = vocabulary.Vocabulary()
    tag_vocab.index(Zparser.START)
    tag_vocab.index(Zparser.STOP)
    tag_vocab.index(Zparser.TAG_UNK)

    word_vocab = vocabulary.Vocabulary()
    word_vocab.index(Zparser.START)
    word_vocab.index(Zparser.STOP)
    word_vocab.index(Zparser.UNK)

    label_vocab = vocabulary.Vocabulary()
    label_vocab.index(())

    sublabels = [Zparser.Sub_Head]
    label_vocab.index(tuple(sublabels))

    sublabels = [Zparser.Head]
    label_vocab.index(tuple(sublabels))

    sublabels.append(Zparser.Sub_Head)
    label_vocab.index(tuple(sublabels))

    type_vocab = vocabulary.Vocabulary()

    char_set = set()

    for i, tree in enumerate(train_parse):

        const_sentences = [leaf.word for leaf in tree.leaves()]
        assert len(const_sentences)  == len(dep_sentences[i])
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalParseNode):
                if node.label[0] != Zparser.Head:
                    label_vocab.index(node.label)
                if node.type is not Zparser.ROOT:#not include root type
                    type_vocab.index(node.type)
                nodes.extend(reversed(node.children))
            else:
                tag_vocab.index(node.tag)
                type_vocab.index(node.type)
                word_vocab.index(node.word)
                char_set |= set(node.word)

    #label with head
    for i, tree in enumerate(train_parse):

        const_sentences = [leaf.word for leaf in tree.leaves()]
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalParseNode):
                if node.label[0] == Zparser.Head:
                    label_vocab.index(node.label)
                nodes.extend(reversed(node.children))

    char_vocab = vocabulary.Vocabulary()

    # If codepoints are small (e.g. Latin alphabet), index by codepoint directly
    highest_codepoint = max(ord(char) for char in char_set)
    if highest_codepoint < 512:
        if highest_codepoint < 256:
            highest_codepoint = 256
        else:
            highest_codepoint = 512

        # This also takes care of constants like tokens.CHAR_PAD
        for codepoint in range(highest_codepoint):
            char_index = char_vocab.index(chr(codepoint))
            assert char_index == codepoint
    else:
        char_vocab.index(tokens.CHAR_UNK)
        char_vocab.index(tokens.CHAR_START_SENTENCE)
        char_vocab.index(tokens.CHAR_START_WORD)
        char_vocab.index(tokens.CHAR_STOP_WORD)
        char_vocab.index(tokens.CHAR_STOP_SENTENCE)
        for char in sorted(char_set):
            char_vocab.index(char)

    tag_vocab.freeze()
    word_vocab.freeze()
    label_vocab.freeze()
    char_vocab.freeze()
    type_vocab.freeze()

    punctuation = hparams.punctuation
    punct_set = punctuation

    def print_vocabulary(name, vocab):
        special = {tokens.START, tokens.STOP, tokens.UNK}
        print("{} ({:,}): {}".format(
            name, vocab.size,
            sorted(value for value in vocab.values if value in special) +
            [value for value in vocab.values if value not in special]))

    if args.print_vocabs:
        print_vocabulary("Tag", tag_vocab)
        print_vocabulary("Word", word_vocab)
        print_vocabulary("Label", label_vocab)
        print_vocabulary("Char", char_vocab)
        print_vocabulary("Type", type_vocab)


    print("Initializing model...")

    load_path = None
    if load_path is not None:
        print(f"Loading parameters from {load_path}")
        info = torch_load(load_path)
        parser = Zparser.ChartParser.from_spec(info['spec'], info['state_dict'])
    else:
        parser = Zparser.ChartParser(
            tag_vocab,
            word_vocab,
            label_vocab,
            char_vocab,
            type_vocab,
            hparams,
        )

    print("Initializing optimizer...")
    trainable_parameters = [param for param in parser.parameters() if param.requires_grad]
    trainer = torch.optim.Adam(trainable_parameters, lr=1., betas=(0.9, 0.98), eps=1e-9)
    if load_path is not None:
        trainer.load_state_dict(info['trainer'])

    def set_lr(new_lr):
        for param_group in trainer.param_groups:
            param_group['lr'] = new_lr

    assert hparams.step_decay, "Only step_decay schedule is supported"

    warmup_coeff = hparams.learning_rate / hparams.learning_rate_warmup_steps
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer, 'max',
        factor=hparams.step_decay_factor,
        patience=hparams.step_decay_patience,
        verbose=True,
    )
    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= hparams.learning_rate_warmup_steps:
            set_lr(iteration * warmup_coeff)

    clippable_parameters = trainable_parameters
    grad_clip_threshold = np.inf if hparams.clip_grad_norm == 0 else hparams.clip_grad_norm


    print("Training...")
    total_processed = 0
    current_processed = 0
    check_every = len(train_parse) / args.checks_per_epoch
    best_dev_score = -np.inf
    best_model_path = None
    model_name = hparams.model_name

    print("This is ", model_name)
    start_time = time.time()

    def check_dev(epoch_num):
        nonlocal best_dev_score
        nonlocal best_model_path

        dev_start_time = time.time()

        parser.eval()

        dev_predicted = []

        for dev_start_index in range(0, len(dev_treebank), args.eval_batch_size):
            subbatch_trees = dev_treebank[dev_start_index:dev_start_index+args.eval_batch_size]
            subbatch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in subbatch_trees]

            predicted, _, = parser.parse_batch(subbatch_sentences)
            del _
            dev_predicted.extend([p.convert() for p in predicted])

        dev_fscore = evaluate.evalb(args.evalb_dir, dev_treebank, dev_predicted)

        print(
            "dev-fscore {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_fscore,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        dev_pred_head = [[leaf.father for leaf in tree.leaves()] for tree in dev_predicted]
        dev_pred_type = [[leaf.type for leaf in tree.leaves()] for tree in dev_predicted]
        assert len(dev_pred_head) == len(dev_pred_type)
        assert len(dev_pred_type) == len(dep_dev_type)
        stats, stats_nopunc, stats_root, num_inst = dep_eval.eval(len(dev_pred_head), dep_dev_word, dep_dev_pos,
                                                                  dev_pred_head, dev_pred_type,
                                                                  dep_dev_headid, dep_dev_type,
                                                                  dep_dev_lengs, punct_set=punct_set,
                                                                  symbolic_root=False)
        dev_ucorr, dev_lcorr, dev_total, dev_ucomlpete, dev_lcomplete = stats
        dev_ucorr_nopunc, dev_lcorr_nopunc, dev_total_nopunc, dev_ucomlpete_nopunc, dev_lcomplete_nopunc = stats_nopunc
        dev_root_corr, dev_total_root = stats_root
        dev_total_inst = num_inst
        print(
            'W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
                dev_ucorr, dev_lcorr, dev_total, dev_ucorr * 100 / dev_total, dev_lcorr * 100 / dev_total,
                dev_ucomlpete * 100 / dev_total_inst, dev_lcomplete * 100 / dev_total_inst))
        print(
            'Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
                dev_ucorr_nopunc, dev_lcorr_nopunc, dev_total_nopunc,
                dev_ucorr_nopunc * 100 / dev_total_nopunc,
                dev_lcorr_nopunc * 100 / dev_total_nopunc,
                dev_ucomlpete_nopunc * 100 / dev_total_inst, dev_lcomplete_nopunc * 100 / dev_total_inst))
        print('Root: corr: %d, total: %d, acc: %.2f%%' % (
            dev_root_corr, dev_total_root, dev_root_corr * 100 / dev_total_root))

        dev_uas = dev_ucorr_nopunc * 100 / dev_total_nopunc
        dev_las = dev_lcorr_nopunc * 100 / dev_total_nopunc

        if dev_fscore.fscore + dev_las  > best_dev_score :
            if best_model_path is not None:
                extensions = [".pt"]
                for ext in extensions:
                    path = best_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_score = dev_fscore.fscore + dev_las
            best_model_path = "{}_best_dev={:.2f}_devuas={:.2f}_devlas={:.2f}".format(
                args.model_path_base, dev_fscore.fscore, dev_uas,dev_las)
            print("Saving new best model to {}...".format(best_model_path))
            torch.save({
                'spec': parser.spec,
                'state_dict': parser.state_dict(),
                'trainer' : trainer.state_dict(),
                }, best_model_path + ".pt")

    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break
        np.random.shuffle(train_parse)
        epoch_start_time = time.time()

        for start_index in range(0, len(train_parse), args.batch_size):
            trainer.zero_grad()
            schedule_lr(total_processed // args.batch_size)

            parser.train()

            batch_loss_value = 0.0
            batch_trees = train_parse[start_index:start_index + args.batch_size]

            batch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in batch_trees]

            for subbatch_sentences, subbatch_trees in parser.split_batch(batch_sentences, batch_trees, args.subbatch_max_tokens):
                _, loss = parser.parse_batch(subbatch_sentences, subbatch_trees)

                loss = loss / len(batch_trees)
                loss_value = float(loss.data.cpu().numpy())
                batch_loss_value += loss_value
                if loss_value > 0:
                    loss.backward()
                del loss
                total_processed += len(subbatch_trees)
                current_processed += len(subbatch_trees)

            grad_norm = torch.nn.utils.clip_grad_norm_(clippable_parameters, grad_clip_threshold)

            trainer.step()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "grad-norm {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    start_index // args.batch_size + 1,
                    int(np.ceil(len(train_parse) / args.batch_size)),
                    total_processed,
                    batch_loss_value,
                    grad_norm,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

            if current_processed >= check_every:
                current_processed -= check_every
                check_dev(epoch)

        # adjust learning rate at the end of an epoch
        if hparams.step_decay:
            if (total_processed // args.batch_size + 1) > hparams.learning_rate_warmup_steps:
                scheduler.step(best_dev_score)


def run_test(args):

    const_test_path = args.consttest_ptb_path

    dep_test_path = args.deptest_ptb_path

    if args.dataset == 'ctb':
        const_test_path = args.consttest_ctb_path
        dep_test_path = args.deptest_ctb_path


    print("Loading model from {}...".format(args.model_path_base))
    assert args.model_path_base.endswith(".pt"), "Only pytorch savefiles supported"

    info = torch_load(args.model_path_base)
    assert 'hparams' in info['spec'], "Older savefiles not supported"
    parser = Zparser.ChartParser.from_spec(info['spec'], info['state_dict'], use_cat = args.use_cat, pre_emb_path = args.embedding_path)
    parser.eval()

    dep_test_reader = CoNLLXReader(dep_test_path, parser.type_vocab)
    print('Reading dependency parsing data from %s' % dep_test_path)

    dep_test_data = []
    test_inst = dep_test_reader.getNext()
    dep_test_headid = np.zeros([40000, 300], dtype=int)
    dep_test_type = []
    dep_test_word = []
    dep_test_pos = []
    dep_test_lengs = np.zeros(40000, dtype=int)
    cun = 0
    while test_inst is not None:
        inst_size = test_inst.length()
        dep_test_lengs[cun] = inst_size
        sent = test_inst.sentence
        dep_test_data.append((sent.words, test_inst.postags, test_inst.heads, test_inst.types))
        for i in range(inst_size):
            dep_test_headid[cun][i] = test_inst.heads[i]
        dep_test_type.append(test_inst.types)
        dep_test_word.append(sent.words)
        dep_test_pos.append(sent.postags)
        # dep_sentences.append([(tag, word) for i, (word, tag) in enumerate(zip(sent.words, sent.postags))])
        test_inst = dep_test_reader.getNext()
        cun = cun + 1

    dep_test_reader.close()
    print("Loading test trees from {}...".format(const_test_path))
    test_treebank = trees.load_trees(const_test_path, dep_test_headid, dep_test_type)
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    print("Parsing test sentences...")
    start_time = time.time()

    punct_set = '.' '``' "''" ':' ','

    test_predicted = []
    for start_index in range(0, len(test_treebank), args.eval_batch_size):
        subbatch_trees = test_treebank[start_index:start_index + args.eval_batch_size]

        subbatch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in subbatch_trees]

        predicted, _ = parser.parse_batch(subbatch_sentences)
        del _
        test_predicted.extend([p.convert() for p in predicted])

    test_fscore = evaluate.evalb(args.evalb_dir, test_treebank, test_predicted)
    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )

    test_pred_head = [[leaf.father for leaf in tree.leaves()] for tree in test_predicted]
    test_pred_type = [[leaf.type for leaf in tree.leaves()] for tree in test_predicted]
    assert len(test_pred_head) == len(test_pred_type)
    assert len(test_pred_type) == len(dep_test_type)
    stats, stats_nopunc, stats_root, test_total_inst = dep_eval.eval(len(test_pred_head), dep_test_word, dep_test_pos,
                                                                     test_pred_head,
                                                                     test_pred_type, dep_test_headid, dep_test_type,
                                                                     dep_test_lengs, punct_set=punct_set,
                                                                     symbolic_root=False)

    test_ucorrect, test_lcorrect, test_total, test_ucomlpete_match, test_lcomplete_match = stats
    test_ucorrect_nopunc, test_lcorrect_nopunc, test_total_nopunc, test_ucomlpete_match_nopunc, test_lcomplete_match_nopunc = stats_nopunc
    test_root_correct, test_total_root = stats_root

    print(
        'best test W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% ' % (
            test_ucorrect, test_lcorrect, test_total, test_ucorrect * 100 / test_total,
            test_lcorrect * 100 / test_total,
            test_ucomlpete_match * 100 / test_total_inst, test_lcomplete_match * 100 / test_total_inst
            ))
    print(
        'best test Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
            test_ucorrect_nopunc, test_lcorrect_nopunc, test_total_nopunc,
            test_ucorrect_nopunc * 100 / test_total_nopunc,
            test_lcorrect_nopunc * 100 / test_total_nopunc,
            test_ucomlpete_match_nopunc * 100 / test_total_inst,
            test_lcomplete_match_nopunc * 100 / test_total_inst
            ))
    print('best test Root: corr: %d, total: %d, acc: %.2f%%' % (
        test_root_correct, test_total_root, test_root_correct * 100 / test_total_root))
    print(
        '============================================================================================================================')

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    hparams = make_hparams()
    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=lambda args: run_train(args, hparams))
    hparams.populate_arguments(subparser)
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--embedding-path", required=True)
    subparser.add_argument("--embedding-type", default="random")
    subparser.add_argument("--model-name", default="test")
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--dataset", default="ptb")
    subparser.add_argument("--train-ptb-path", default="data/02-21.10way.clean")
    subparser.add_argument("--dev-ptb-path", default="data/22.auto.clean")
    subparser.add_argument("--dep-train-ptb-path", default="data/ptb_train_3.3.0.sd")
    subparser.add_argument("--dep-dev-ptb-path", default="data/ptb_dev_3.3.0.sd")

    subparser.add_argument("--train-ctb-path", default="data/train_ctb.txt")
    subparser.add_argument("--dev-ctb-path", default="data/dev_ctb.txt")
    subparser.add_argument("--dep-train-ctb-path", default="data/train_ctb.conll")
    subparser.add_argument("--dep-dev-ctb-path", default="data/dev_ctb.conll")

    subparser.add_argument("--batch-size", type=int, default=250)
    subparser.add_argument("--subbatch-max-tokens", type=int, default=2000)
    subparser.add_argument("--eval-batch-size", type=int, default=50)
    subparser.add_argument("--epochs", type=int)
    subparser.add_argument("--checks-per-epoch", type=int, default=4)
    subparser.add_argument("--print-vocabs", action="store_true")

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--embedding-path", default="data/glove.6B.100d.txt.gz")
    subparser.add_argument("--dataset", default="ptb")
    subparser.add_argument("--use-cat", action='store_true')
    subparser.add_argument("--consttest-ptb-path", default="data/23.auto.clean")
    subparser.add_argument("--deptest-ptb-path", default="data/ptb_test_3.3.0.sd")
    subparser.add_argument("--consttest-ctb-path", default="data/test_ctb.txt")
    subparser.add_argument("--deptest-ctb-path", default="data/test_ctb.conll")
    subparser.add_argument("--eval-batch-size", type=int, default=100)


    args = parser.parse_args()
    args.callback(args)

# %%
if __name__ == "__main__":
    main()
