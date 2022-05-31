import json
import os
from typing import List, Tuple, Callable, Any

from colors import color, ansilen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sentencepiece as spm
import matplotlib.ticker as ticker
import requests
import transformers

MWAPIKEY = os.environ["MERRIAM_WEBSTER"]
DATA_LOCATION = os.path.expanduser("~/data/tokenization/")


class FormattedTable:

    def __init__(self, tokenizers: List[Tuple[str, Any]],
                 max_word_len: int = 14,
                 max_separators: int = 5,
                 max_width: int = 72):
        self.tokenizers = tokenizers
        self.max_word_len = max_word_len
        self.max_separators = max_separators
        self.max_width = max_width
        self.words = []
        self.gold_standard = []
        self.tokenizations = {t: list() for name, t in tokenizers}

    @property
    def col_width(self):
        return self.max_word_len + self.max_separators

    @property
    def heading_width(self):
        tokenizer_names = [name for name, _ in self.tokenizers] + ["Word"]
        if self.gold_standard:
            tokenizer_names.append("Gold standard")
        return max([len(x) for x in tokenizer_names]) + 1

    @staticmethod
    def ansicenter(s, width):
        extra_chars = width - ansilen(s)
        left = int(extra_chars / 2)
        right = extra_chars - left
        return " " * left + s + " " * right

    @staticmethod
    def ljust(s, width):
        extra_chars = width - ansilen(s)
        return s + " " * extra_chars

    def _format_tokenization(self, tokens: List[str], gold_standard: List[str] = None) -> str:
        cell = ""
        if gold_standard:
            tokens, subwords, separators = subwords_are_morphologically_coherent(tokens, gold_standard)
            assert len(subwords) == len(tokens)
            assert len(separators) == len(tokens) + 1
            for i, (token, sw_valid, sep_valid) in enumerate(zip(tokens, subwords, separators)):
                if i >= 1:
                    cell += color("-", fg='dark green', bg=194, style='bold') \
                        if sep_valid else color("-", fg='dark red', style='bold', bg=224)
                cell += color(token, fg='dark green') if sw_valid else color(token, fg='dark red')
            return cell
        else:
            tokens = clean_tokens(tokens)
            return "-".join(tokens)

    def _run_on_word(self, word: str, gold_standard: List[str] = None):
        self.words.append(word)
        if gold_standard:
            gs = clean_tokens(gold_standard)
            self.gold_standard.append("-".join(gs))
        for name, tokenizer in self.tokenizers:
            formatted = self._format_tokenization(tokenize(tokenizer, word), gold_standard)
            self.tokenizations[tokenizer].append(self.ansicenter(formatted, self.col_width))

    def _fmt_top_heading(self, heading):
        return heading.center(self.col_width)

    def _fmt_side_heading(self, heading):
        return heading.rjust(self.heading_width) + ' '

    def _fmt_gold_standard(self, gs):
        return self.ansicenter(gs, self.col_width)

    def _display_words(self, word_indices: List[int]):
        row = self._fmt_side_heading("") + " "
        for word_idx in word_indices:
            row += self._fmt_top_heading(self.words[word_idx])
        print(color(row, bg='light blue', style='bold'))
        if self.gold_standard:
            row = self._fmt_side_heading("Gold standard") + " "
            for word_idx in word_indices:
                row += self._fmt_gold_standard(self.gold_standard[word_idx])
            print(color(row, bg='light green', style='bold'))
        for name, tokenizer in self.tokenizers:
            row = color(self._fmt_side_heading(name) + " ", bg='light blue', style='bold')
            for word_idx in word_indices:
                row += self.tokenizations[tokenizer][word_idx]
            print(row)

    def _display(self):
        n_words = len(self.words)
        row_prefix_width = self.heading_width + self.col_width if self.gold_standard else self.heading_width
        words_per_table = (self.max_width - row_prefix_width) // self.col_width
        for start_pos in range(0, n_words, min(words_per_table, n_words)):
            subset = range(start_pos, min(start_pos + words_per_table, n_words))
            self._display_words(subset)
            print("")

    def print_table_from_words(self, *examples: str):
        for word in examples:
            self._run_on_word(word)
        self._display()

    def print_table_from_gold_standard(self, *examples: List[str]):
        for ex in examples:
            self._run_on_word("".join(ex), ex)
        self._display()

    def lookup_and_print(self, *examples: List[str]):
        for example in examples:
            pron = get_pronunciation(example)
            self._run_on_word("".join(pron), pron)
        self._display()


def get_definition(word):
    url = f'https://www.dictionaryapi.com/api/v3/references/collegiate/json/{word}?key={MWAPIKEY}'
    r = requests.get(url)
    return word, json.loads(r.content.decode())


def get_pronunciation(word):
    try:
        word, defn = get_definition(word)
        return defn[0]['hwi']['hw'].split("*")
    except:
        return "?"


def clean_token(token):
    token = token.replace('Ġ', '')  # Used by GPT-2 to indicate a word-start
    token = token.replace('▁', '')  # Used by sentencepiece to indicate a word-start
    token = token.replace('##', '')   # Used by Bert to indicate an internal word
    return token


def clean_tokens(tokens):
    tokens = [clean_token(token) for token in tokens]
    if tokens[0] == "":
        tokens = tokens[1:]  # E.g. ulm.encode_as_pieces('pictorial') -> ['▁', 'pic', 'torial']
    return tokens


def load_vocab(model_type: str, vocab_size_in_thousands: int) -> List[str]:
    vocab = list()
    path = os.path.join(DATA_LOCATION,
                        "pretrained_lms",
                        f"{model_type.lower()}10M_{vocab_size_in_thousands}k.vocab")
    with open(path, 'r') as fp:
        for line in fp.readlines():
            token, score = line.split("\t")
            vocab.append(token)
    return clean_tokens(vocab)


def load_tokenizer(model_type: str, vocab_size_in_thousands: int) -> spm.SentencePieceProcessor:
    tokmodel = spm.SentencePieceProcessor()
    path = os.path.join(DATA_LOCATION,
                        "pretrained_lms",
                        f"{model_type.lower()}10M_{vocab_size_in_thousands}k.model")
    tokmodel.load(path)
    return tokmodel


def tokenize(tokenizer, s: str) -> List[str]:
    """Standard interface to tokenize either Transformers or SentencePiece tokenizers"""
    if isinstance(tokenizer, transformers.tokenization_gpt2.GPT2Tokenizer):
        s = " " + s  # Need to add prefix space. See https://huggingface.co/transformers/model_doc/gpt2.html#transformers.GPT2TokenizerFast
    if isinstance(tokenizer, transformers.tokenization_utils.PreTrainedTokenizer):
        func = tokenizer.tokenize
    elif isinstance(tokenizer, spm.SentencePieceProcessor):
        func = tokenizer.encode_as_pieces
    else:
        raise TypeError
    return func(s)


def display_tokens(tokens):
    tokens = [clean_token(token) for token in tokens]
    return "-".join(tokens)


def load_gold_standard() -> List[str]:
    df = pd.read_csv(os.path.join(DATA_LOCATION, "merriam_webster_pronunciation_guides.csv"), header=None)
    pronunciations = df[1].values
    examples = [p.split("*") for p in pronunciations if isinstance(p, str)]
    return examples


def plot_subword_growth():
    examples = load_gold_standard()
    unique_subwords = set()
    n_unique_subwords = list()
    for word in examples:
        unique_subwords = unique_subwords.union(word)
        n_unique_subwords.append(len(unique_subwords))
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    sns.lineplot(range(len(n_unique_subwords)), n_unique_subwords, palette='dark')  # , c='b')
    plt.xlabel("Number of words")
    plt.ylabel("Unique subwords")
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    plt.show()


def frac_splits_hit_hyphen_spots(tokenized: List[str], gold_standard: List[str]):
    tokenized = clean_tokens(tokenized)
    if len(tokenized) == 1:
        return 1  # Our tokenizer has the word in its vocabulary.
    token_splits = set(np.cumsum([len(x) for x in tokenized[:-1]]))
    valid_points = set(np.cumsum([len(x) for x in gold_standard[:-1]]))
    return len(token_splits.intersection(valid_points)) / (len(token_splits))


def subwords_are_morphologically_coherent(tokenized: List[str], gold_standard: List[str]):
    tokenized = clean_tokens(tokenized)
    token_splits = [0, *np.cumsum([len(x) for x in tokenized])]
    token_boundaries = [(token_splits[i], token_splits[i + 1]) for i in range(len(tokenized))]
    valid_points = set(np.cumsum([len(x) for x in gold_standard])).union([0])
    subwords = [a in valid_points and b in valid_points for a, b in token_boundaries]
    splits = [boundary in valid_points for boundary in token_splits]
    return tokenized, subwords, splits


def frac_subwords_are_morphologically_coherent(tokenized: List[str], gold_standard: List[str]):
    _, subwords_are_valid, _ = subwords_are_morphologically_coherent(tokenized, gold_standard)
    return np.mean(subwords_are_valid)


def plot_tokenizer_evaluation_scores():
    res = pd.read_csv(os.path.join(DATA_LOCATION, "tokenizer_evaluations.csv"))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
    sns.lineplot(x="useful_vocab_size", y="score", hue="model_type",
                 ci=99, data=res, marker='o', ax=ax, err_style='band',
                 markersize=8)
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim([0, 0.7])
    ax.set_xlabel("Relevant vocab size")
    ax.set_ylabel("% morphologically sound subwords")

    bert = res[res['model_type'] == 'Bert'].mean()
    gpt = res[res['model_type'] == 'GPT2'].mean()
    plt.text(bert.useful_vocab_size + 250, bert.score + 0.0125, "Bert", size='large')
    plt.text(gpt.useful_vocab_size + 400, gpt.score - 0.025, "GPT-2", size='large')

    bpe = res[res['model_type'] == 'BPE'].max()
    ulm = res[res['model_type'] == 'ULM'].max()
    bpe = res[(res['model_type'] == 'BPE') & (res['vocab_size'] == 64000)].mean()
    ulm = res[(res['model_type'] == 'ULM') & (res['vocab_size'] == 64000)].mean()
    plt.text(bpe.useful_vocab_size + 750, bpe.score, "BPE", size='large')
    plt.text(ulm.useful_vocab_size + 750, ulm.score, "Unigram LM", size='large')
    ax.legend().set_visible(False)
    plt.show()


def make_thumbnail():
    res = pd.read_csv(os.path.join(DATA_LOCATION, "tokenizer_evaluations.csv"))

    fig, ax = plt.subplots(1, 1, figsize=(4, 5), sharex=True)
    sns.lineplot(x="useful_vocab_size", y="score", hue="model_type",
                 ci=99, data=res, marker='o', ax=ax, err_style='band',
                 markersize=8)
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim([0, 0.7])
    ax.set_xlabel("Relevant vocab size")
    ax.set_ylabel("% morphologically sound subwords")

    bert = res[res['model_type'] == 'Bert'].mean()
    gpt = res[res['model_type'] == 'GPT2'].mean()
    plt.text(bert.useful_vocab_size + 350, bert.score + 0.015, "Bert", size='large')
    plt.text(gpt.useful_vocab_size + 650, gpt.score - 0.035, "GPT-2", size='large')

    bpe = res[res['model_type'] == 'BPE'].max()
    ulm = res[res['model_type'] == 'ULM'].max()
    bpe = res[(res['model_type'] == 'BPE') & (res['vocab_size'] == 64000)].mean()
    ulm = res[(res['model_type'] == 'ULM') & (res['vocab_size'] == 64000)].mean()
    plt.text(bpe.useful_vocab_size - 1000, bpe.score - 0.01, "BPE", size='large', ha='right', rotation=24)
    plt.text(ulm.useful_vocab_size - 5000, ulm.score - 0.07, "Unigram LM", size='large', ha='right', rotation=22)
    ax.legend().set_visible(False)
    plt.savefig("../images/tokenization-preview.png", bbox_inches='tight')


def plot_learning_speed():
    learn_times = pd.read_csv(os.path.join(DATA_LOCATION, "time_to_learn.csv"))
    learn_times['time'] = learn_times['time'] / 60
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    sns.lineplot(x="vocab_size", y="time", hue="model_type",
                 ci=95, data=learn_times, marker='o', ax=ax,
                 markersize=8)
    plt.xlabel("Vocab size")
    ax.set_ylabel("Time to build (minutes)")
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    plt.show()


def plot_inference_speed():
    res = pd.read_csv(os.path.join(DATA_LOCATION, "tokenizer_evaluations.csv"))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plt.ylabel("Frac. morphologically sound subwords")
    plt.ylim(0, 120)
    sns.lineplot(x="useful_vocab_size", y="time_per_million", hue="model_type",
                 ci=95, data=res, marker='o', ax=ax)
    plt.xlabel("Relevant vocab size")
    plt.ylabel("Seconds per million tokens")
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    plt.show()
