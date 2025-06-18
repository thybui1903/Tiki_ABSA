
import numpy as np
import re
import unicodedata

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    """
    Pad and truncate a sequence to a fixed length
    """
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

def preprocess_sentence(sentence: str):
    sentence = sentence.lower()
    sentence = unicodedata.normalize("NFC", sentence)
    # remove all non-characters and punctuations
    sentence = re.sub(r"[-/“”!\*\&\$\.\?:;,\"'\(\[\]\(\)]", " ", sentence)
    sentence = " ".join(sentence.strip().split()) # remove duplicated spaces
    tokens = sentence.strip().split()
    
    return tokens