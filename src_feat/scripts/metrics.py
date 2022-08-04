# import statements
import numpy as np
from re import finditer

# simple function to convert indices to characters for easier pattern match
f = lambda S: ''.join(["BIOES<>$"[i] for i in S])

def _compute(y_true, y_pred, value="recall", model="seq2seq"):
    # the pattern to identify for a feature the pattern is: 
    # 1. Either a single S
    # 2. Or B followed by 0 or more I, followed by an E
    # since words may be broken during tokenization with a hugging face transformer
    # adjust the pattern appropriately
    PATTERN = "S|BI*E" if model == "seq2seq" else "S+|B+I*E+"
    y_true, y_pred = f(y_true), f(y_pred)
    S = y_true if value == "recall" else y_pred
    # the below line helps us find all the set of indices where we find necessary pattern match in y_pred or y_true
    match_indices = [(m.start(0), m.end(0)) for m in finditer(PATTERN, S)]
    # find how many matches are present, we can call them retrieved and relevent
    retrieved_n_relevant = len([1 for (i, j) in match_indices if y_true[i:j] == y_pred[i:j]])
    return [retrieved_n_relevant, len(match_indices)]

def _generic_metric(y_true, y_pred, value="recall", model="seq2seq"):
    s = np.sum([_compute(yt, yp, value=value, model=model) for (yt, yp) in zip(y_true, y_pred)], axis=0)
    return s[0] / (s[1] + 1e-8)

# recall := how many relevent items have been retrieved ?
def recall(y_true, y_pred, model="seq2seq"):
    return _generic_metric(y_true, y_pred, value="recall", model=model)

# precision := how many retrieved items are relevant ?
def precision(y_true, y_pred, model="seq2seq"):
    return _generic_metric(y_true, y_pred, value="precision", model=model)

def f1score(y_true, y_pred, model="seq2seq", return_precision_recall=True):
    r = recall(y_true, y_pred, model=model)
    p = precision(y_true, y_pred, model=model)
    f1 = (2 * p * r)/(p + r + 1e-8)
    return (r, p, f1) if return_precision_recall else f1