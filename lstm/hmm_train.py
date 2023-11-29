# coding: utf-8
import logging
import pickle
import re
import string

import numpy as np
from hmmlearn import hmm
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.DEBUG)

punctuation = string.punctuation + "«»„“”‛’—–‒"
spaces = " 　 \r\n\t"
RE_PUNCT = re.compile(f"[{re.escape(punctuation)}]")
RE_SP = re.compile(rf"[{spaces}\s]+")

ALL_CHARS = list(string.printable + "йцукенгшщзхъфывапролджэячсмитьбюё")


def clean(_str):
    _str0 = re.sub(RE_PUNCT, " ", _str)
    _str1 = re.sub(RE_SP, " ", _str0).lower()
    fin_str = re.sub(r"\s+", " ", _str1)
    return fin_str.strip()


def make_tensor(string, char_set=ALL_CHARS, train=False):
    tensor = []
    for c in string:
        try:
            tensor.append([char_set.index(c)])
        except ValueError:
            if train is True:
                char_set.append(c)
                tensor.append([char_set.index(c)])
            else:
                tensor.append([char_set.index("@")])
    return np.array(tensor)


def train_hmm():
    _tensors = []
    with open("nlp_scripts/train_0.txt") as f:
        i = 0
        pbar = tqdm(total=500_000)
        for line in f:
            if len(line.strip()) > 0:
                line = clean(line)[:100]
                tensor = make_tensor(line, train=True)
                if len(tensor) > 0:
                    _tensors.append(tensor)
                    i += 1
                    pbar.update(1)
            if i >= 500_000:
                break

    tensors = np.concatenate(_tensors)
    lengths = [len(t) for t in _tensors]

    gauss = hmm.CategoricalHMM(n_components=len(ALL_CHARS), verbose=True, n_iter=100)
    gauss.fit(tensors, lengths)
    pickle.dump(gauss, open("character_hmm.pickle", "wb"))


def simple_next_char(_model: hmm.BaseHMM, _text, all_chars=ALL_CHARS):
    tensor = make_tensor(_text)
    state = _model.predict(tensor)
    next_state = _model.transmat_[state[-1]].argmax()
    output = _model.emissionprob_[next_state].argmax()

    return all_chars[output]


train_hmm()
gauss = pickle.load(open("character_hmm.pickle", "rb"))

text = clean(
    """КЛАПАН ОБРАТНЫЙ В ПЛАСТИКОВОМ КОРПУСЕ, СИСТЕМЫ СЛИВА ОТРАБОТАННОЙ ВОДЫ
ИЗ РАКОВИНЫ ТУАЛЕТА ВС EMBRAER 170 ГРАЖД. АВИАЦИИ:"""
)
for w in range(len(text)):
    if text[w] == " ":
        gen = "" + (c := simple_next_char(gauss, text[:w]))
        while c != " ":
            c = simple_next_char(gauss, gen)
            gen += c
        print(text[:w], ",", gen)
        # if c != " ":
        #     continue
