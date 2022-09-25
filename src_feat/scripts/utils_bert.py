# import statements
from transformers import RobertaTokenizerFast, BertTokenizerFast
import pandas as pd
from ast import literal_eval

# a major function to help preprocess the text, tokenize it and extend the labels appropriately
def get_encoded_input(fname, tag2idx=None, tokenizer_name="roberta-base", return_classification_targets=False):

    # sub-function to extend the labels
    def _extend_labels(txt, lbls):
        extended_labels = [tag2idx['<']]
        for (t, l) in zip(txt, lbls):
            n = len(tokenizer.tokenize(t))
            extended_labels.extend([l] * n)
        extended_labels.append(tag2idx['>'])
        rem = max_len - len(extended_labels)
        extended_labels.extend([tag2idx['$']] * rem)
        return extended_labels

    # use appropriate tokenizer
    if tokenizer_name == "roberta-base":
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", do_lower_case=True, add_prefix_space=True)
    elif tokenizer_name == "bert-base-uncased":
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", do_lower_case=True)

    # read the data and decode it
    data = pd.read_csv(fname,
                       sep=' ',
                       header=None,
                       names=['a', 'b', 'c'],
                       encoding="utf-8",
                       converters={'a': literal_eval, 
                                   'b': literal_eval})

    text = data['a'].tolist()
    # tokenize the data
    ext_txt = tokenizer(text, 
                        padding="longest",
                        return_attention_mask=True,
                        is_split_into_words=True,
                        return_length=True)

    max_len = ext_txt["length"][0]
    
    if not return_classification_targets:
        labels = [[tag2idx[l.split('-')[0]] for l in labels] for labels in data['b']]
        # extend the labels appropriately (as per the tokenization of the words)
        ext_labels = [_extend_labels(txt, lbl) for (txt, lbl) in zip(text, labels)]
        return ext_txt, ext_labels
    else:
        data['c'] = data['c'].apply(lambda x: 0 if not x else 1)
        return ext_txt, data['c'].tolist()
