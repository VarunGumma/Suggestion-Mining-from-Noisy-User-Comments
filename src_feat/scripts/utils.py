# import statements
import spacy
import pandas as pd

# function to help post-pad sequences with PAD tokens
# also generates masks for the pad tokens when necessary
def post_pad_sequences(sequences, 
                       maxlen=None,
                       start="<START>",
                       end="<END>",
                       pad="<PAD>",
                       return_masks=True):

    # find the max-length of all sequences
    # if maxlen is None, maxlen is assumed as length of longest sequence
    L = max([len(x) for x in sequences])
    # if maxlen is None or we are asked to pad to more than the longest sequences
    # pad only to the longest sequence. Don't use extra pad tokens.
    # the final length after truncating the sentences, adding special tokens will be maxlen, if it is specified
    if maxlen is None or maxlen-2 > L:
        sequences = [([start] + x + [end] + [pad]*(L-len(x))) for x in sequences]
        # generate the mask appropriately. 
        # only pad tokens should be masked
        masks = [[(0 if x == pad else 1) for x in s] for s in sequences] if return_masks else None
    else:
        # all this creative stuff below is to appropriate truncate the sentences and add special tokens
        # so that they fit the maxlen critera
        sequences = [([start] + (x[:maxlen-2] if len(x) >= maxlen-2 else x) + [end] + [pad]*(max(maxlen-len(x)-2, 0))) for x in sequences]
        masks = [[(0 if x == pad else 1) for x in s] for s in sequences] if return_masks else None
    # returns a dictionary of padded sequences and masks
    return {"seq": sequences, "mask": masks}
    

# reads the data, pre-processes it and returns the input_ids and masks      
def get_encoded_input(fname, 
                      tag2idx=None, 
                      maxlen=256, 
                      vocab2idx=None, 
                      pos_tags2idx=None,
                      return_pos_tags=False):

    data = pd.read_csv(fname,
                       sep=' ',
                       header=None,
                       names=['a', 'b', 'c'],
                       encoding="utf-8",
                       converters={'a': pd.eval, 
                                   'b': pd.eval})

    # text = [tokenizer.tokenize(' '.join(words)) for words in data['a']]
    text = post_pad_sequences(data['a'], maxlen=maxlen, return_masks=True)
    encoded_input = [[vocab2idx[w] for w in sent] for sent in text["seq"]]
    
    labels = [[l.split('-')[0] for l in labels] for labels in data['b']]
    # don't forget to appropriate pad the labels as well will special label tokens
    # for this operation, we dont need the masks
    # set return_masks to False to save some computation
    labels = post_pad_sequences(labels,
                                start='<', 
                                end='>', 
                                pad='$', 
                                maxlen=maxlen, 
                                return_masks=False)["seq"]

    extended_labels = [[tag2idx[l] for l in lbls] for lbls in labels]

    # additionally, we can even return pos-tags of each word
    # this helps improve the performance we are providing complex data explicitly
    # here wre use spacy pos-tagger to generate the pos-tags
    if return_pos_tags and pos_tags2idx is not None:
        nlp = spacy.load("en_core_web_sm")
        pos_tags = [[pos_tags2idx[token.pos_] for token in nlp(' '.join(s))] for s in data['a']]
        # pad the pos-tags as well which appropriate special tokens
        pos_tags = post_pad_sequences(pos_tags, 
                                      maxlen=maxlen, 
                                      start=pos_tags2idx['X'], 
                                      end=pos_tags2idx['X'], 
                                      pad=pos_tags2idx['PAD_AUX'], 
                                      return_masks=False)["seq"]

        return encoded_input, pos_tags, text["mask"], extended_labels
    else:
        return encoded_input, text["mask"], extended_labels