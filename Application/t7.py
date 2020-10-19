import streamlit as st
import numpy as np
import pandas as pd

def app(seed, sidebar_top):
    #parameters
    sigma_l = 97
    sigma_r = 100
    sigma = [ chr(i) for i in range(sigma_l,sigma_r) ]
    #generate random text
    rg = np.random.default_rng(seed)
    tlength = rg.integers(12,24)
    t = rg.integers(sigma_l,sigma_r,tlength) 
    text = ''.join([chr(a) for a in t])

    #select random substring from text as pattern
    plength = rg.integers(4,9)
    pstart = rg.integers(0,len(text) - plength)
    pattern = text[pstart:pstart+plength]

    #create pattern with optional letters
    pattern_opt_letters = rg.integers(0,2,plength)
    pattern_opt_list = []
    for i,c in enumerate(pattern):
        if pattern_opt_letters[i]:
            pattern_opt_list.append(c+'?')
        else: pattern_opt_list.append(c)
    pattern_opt = ''.join(pattern_opt_list)

    descr = 'Gegeben sei der Text $T=\\texttt{' + text + '}$ und das erweiterte Muster $P=\\texttt{' + pattern_opt + '}$.'
    st.write(descr)
    sidebar_top.info('$T=\\texttt{' + text + '}$\n\r$P=\\texttt{' + pattern_opt + '}$')