import streamlit as st
import numpy as np
import t3
import graphviz as gv

def app(seed,sidebar_top):
    #parameters
    sigma_l = 97
    sigma_r = 100
    sigma = [ chr(i) for i in range(sigma_l,sigma_r) ]

    #generate random text
    rg = np.random.default_rng(seed)
    tlength = rg.integers(12,24)
    t = rg.integers(sigma_l,sigma_r,tlength)
    text = ''.join([chr(a) for a in t])

    #select random substrings from text as patterns
    plengths = rg.integers(3,5,rg.integers(3,6))
    pstarts = [ rg.integers(0,tlength - pl) for pl in plengths ]
    patterns_duplicates = [ text[ps:ps+pl] for ps,pl in zip(pstarts,plengths)]
    patterns = [ p for k,p in enumerate(patterns_duplicates) if p not in patterns_duplicates[:k] ]
    iteration_count = 0
    while len(patterns) < 3 and iteration_count < 128:
        new_plength = rg.integers(3,5)
        new_pstart = rg.integers(0,tlength - new_plength)
        new_pattern = text[new_pstart : new_pstart + new_plength]
        if new_pattern not in patterns: patterns.append(new_pattern)
        iteration_count += 1

    st.write(r'Gegeben sei der Text $T=\texttt{'+text+r'}$ und die Mustermenge $P = \{\texttt{'+','.join(patterns)+'}\}$.')
    sidebar_top.info(r'$T=\texttt{'+text+'}$\n\r'+ r'$P = \{\texttt{'+','.join(patterns)+'}\}$')

    #A1
    st.subheader('Aufgabe 1')
    conc_pattern = ''.join(patterns)
    sorted_set_patterns = sorted(set(conc_pattern))
    st.write('Erstellen Sie alle Masken für den Shift-And-Algorithmus für die Mustermenge $P$ über dem Alphabet $\Sigma =\{\\texttt{'+','.join(sorted_set_patterns)+'}\}$. Konkatenieren Sie die Muster dazu in der Reihenfolge, in der sie oben angegeben sind.')
    masksInput = dict()
    for c in sorted_set_patterns:
        st.markdown('$\\textit{\\textsf{mask}}^\mathtt{'+c+'}$')
        masksInput[c] = st.text_input('','',len(conc_pattern),'t5a1'+c)
    st.markdown(r'$\textit{\textsf{accept}}$')
    masksInput['accept'] =  st.text_input('','',len(conc_pattern))
    masks = t3.a2_create_masks(conc_pattern)
    masks['accept'] = a1_accept_mask(plengths)
    if st.button('Aufgabe überprüfen', key = 't5_a1_ev_btn'):
        a1_eva = t3.eva_a3(masks,masksInput,len(conc_pattern))
        if not a1_eva:
            fba1 = '>**Lösung korrekt.**'
        elif len(a1_eva) == 1:
            fba1 = f'>**Fehler in der Eingabe für $' + t3.a3_eva_katex(a1_eva)[0] +'$.**'
        else:
            a1_eva_list = t3.a3_eva_katex(a1_eva)
            fba1 = '>**Fehler in den Eingaben für $'+ ','.join(a1_eva_list[:-1]) + '$ und $'+ a1_eva_list[-1] + '$.**'
        st.markdown(fba1)

    #A2
    st.subheader('Aufgabe 2')
    st.write('Erstellen Sie für den Aho-Corasick-Algorithmus einen Trie samt lps-Kanten und Ausgabemengen für die Mustermenge $P$.')
    if st.checkbox('Lösung anzeigen', key = 't5_a2_check'):
        st.write(a2_draw_aho_corasick_trie(patterns))

    #A3
    st.subheader('Aufgabe 3')
    st.write('Führen Sie den Aho-Corasick-Alogrithmus mit der Mustermenge $P$ auf dem Text $T$ aus. Zeigen Sie, in welchem Zustand sich der Automat bei der Mustersuche nach jedem Textzeichen befindet. ')

def a1_accept_mask(plengths):
    a = 1
    acc = 0
    for m in plengths:
        a <<= m
        acc |= a
    return acc>>1
         
def a2_draw_aho_corasick_trie(patterns):
    d = gv.Digraph()
    return d