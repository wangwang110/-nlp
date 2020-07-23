# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:06:30 2017

@author: CVTE
"""

import re
from collections import Counter
import os



def words(text):
    text=re.sub(r"[^A-Za-z\'\-]",' ',text)
    text=re.sub(r"\s{2,}", " ", text)
    return text.split(' ')
    #r'\w+' 匹配字母或数字或下划线或汉字
    ## r"[^A-Za-z\'\-]"
dir_path=os.path.dirname(os.path.realpath(__file__))
lines=os.path.join(dir_path,"dictionary_youdao_1.txt")
WORDS = Counter(words(open(lines).read()))



#def correction(word): 
#    "Most probable spelling correction for word."
#    return max(candidates(word), key=P)
#
#'''def candidates(word): 
#    "Generate possible spelling corrections for word."
#    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])
#    # for true words error, can't write so  '''

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
    
def edits3(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits2(word) for e2 in edits1(e1))
    

if __name__ == '__main__':
    pass
    #print len(WORDS)
    #print edits1("becsmeo")
    #print known(edits1("becsmeo"))
    #print known(edits3('insgtiututi'))