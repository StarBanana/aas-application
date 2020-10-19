import streamlit as st
import numpy as np
import string
import graphviz as gv
import re

def app(seed,sidebar_top):
    sigma_l = 97
    sigma_r = 100
    #generate random text
    rg = np.random.default_rng(seed)
    tlength = rg.integers(12,21)
    t = rg.integers(sigma_l,sigma_r,tlength)
    text = ''.join([chr(a) for a in t]) 
    tlength = len(text)
    #select random substring from text as pattern
    plength = rg.integers(4,9)
    pstart = rg.integers(0,len(text) - plength)
    pattern = text[pstart:pstart+plength]

    descr = 'Gegeben sei der Text $T=\\texttt{' + text + '}$ und das Muster $P=\\texttt{' + pattern + '}$.'
    st.markdown(descr)
    sidebar_top.info('$T=\\texttt{' + text + '}$\n\r$P=\\texttt{' + pattern + '}$')

    #A1
    st.subheader('Aufgabe 1')
    descr11 = 'Zeichnen Sie den NFA zu $P$.'
    st.markdown(descr11)
    a1_exp = st.beta_expander('Lösung anzeigen')
    with a1_exp:
        st.write(a1_draw_nfa(pattern))
    masks = a2_create_masks(pattern)

    #A2
    st.subheader('Aufgabe 2')
    descr21 = 'Geben Sie die aktive Zustandsmenge des NFA nach Lesen der folgenden Präfixe von $T$ an.'
    st.markdown(descr21)
    a2_indexes = sorted(rg.choice(np.arange(4,tlength),3,replace = False))
    a2_input = []
    for k,i in enumerate(a2_indexes):
        st.markdown(f'$T\\lbrack..{i-1}\\rbrack=\\texttt{{{text[:i]}}}$')
        a2_input.append(st.text_input('',key='t3a2'+str(k)))
    if(st.button('Aufgabe überprüfen',key = 't3_a2_ev_btn')):
        a2_ev = ev_a2(a2_indexes, a2_input,text,masks,plength)
        if not a2_ev:
            fba2 = '>**Lösung korrekt.**'
        elif len(a2_ev) == 1:
            fba2 = f'>**Fehler in der Eingabe für $T\\lbrack ..{a2_indexes[a2_ev[0]]-1}\\rbrack$.**'
        else:
            a2_ev_list = ['T\\lbrack ..'+str(a2_indexes[i]-1)+'\\rbrack' for i in a2_ev]
            fba2 = '>**Fehler in den Eingaben für $' + ','.join(a2_ev_list[:-1]) + '$ und $' + a2_ev_list[-1] + '$.**'
        st.markdown(fba2)
        
    #A3
    st.subheader('Aufgabe 3')
    descr31 = 'Erstellen Sie für den Shift-And-Algorithmus die Masken für $P$ über dem Alphabet $\Sigma = \{\\mathtt ' + ',\\mathtt '.join(sorted(set(pattern))) + '\}$.'
    st.markdown(descr31)
    
    masksInput = dict()
    for c in sorted(set(pattern)):
        st.markdown('$\\textit{\\textsf{mask}}^\mathtt{'+c+'}$')
        masksInput[c] = st.text_input('','',len(pattern),'t3a3'+c)
    st.markdown(r'$\textit{\textsf{accept}}$')
    masksInput['accept'] =  st.text_input('','',len(pattern))
    if st.button('Aufgabe überprüfen', key = 't3_a3_ev_btn'):
        a3_ev = ev_a3(masks,masksInput,plength) 
        if not a3_ev:
            fba3 = '>**Lösung korrekt.**'
        elif len(a3_ev) == 1:
            fba3 = f'>**Fehler in der Eingabe für $' + a3_ev_katex(a3_ev)[0] +'$.**'
        else:
            a3_ev_list = a3_ev_katex(a3_ev)
            fba3 = '>**Fehler in den Eingaben für $'+ ','.join(a3_ev_list[:-1]) + '$ und $'+ a3_ev_list[-1] + '$.**'
        st.markdown(fba3)

    #A4
    st.subheader('Aufgabe 4')
    descr41 = 'Führen Sie eine Mustersuche für das Muster $P$ mit dem Shift-And-Algorithmus auf dem Text $T$ aus. Zeigen Sie das aktive Bitmuster nach jedem Schritt.'
    st.markdown(descr41)
    a4_input = []
    for k in range(0,tlength-1,2):
        cols = st.beta_columns([8,1,8])
        #a4_input.append(st.text_input(f"Iteration {k+1}: Bitmuster nach Lesen von T[{str(k)}] = '{text[k]}'.",'',plength,'t3a4'+str(k)))
        #a4_input.append(st.text_input(f"Iteration {k+2}: Bitmuster nach Lesen von T[{str(k+1)}] = '{text[k+1]}'.",'',plength,'t3a4'+str(k+1)))
        a4_input.append(cols[0].text_input(f"Iteration {k+1}: Bitmuster nach Lesen von T[{str(k)}] = '{text[k]}'.",'',plength,'t3a4'+str(k)))
        a4_input.append(cols[2].text_input(f"Iteration {k+2}: Bitmuster nach Lesen von T[{str(k+1)}] = '{text[k+1]}'.",'',plength,'t3a4'+str(k+1)))
    cols = st.beta_columns([8,1,8])
    if tlength % 2 == 1:
        a4_input.append(cols[0].text_input(f"Iteration {tlength-1}: Bitmuster nach Lesen von T[{str(tlength-1)}] = '{text[tlength-1]}'.",'',plength,'t3a4'+str(tlength-1)))
    if st.button('Aufgabe überprüfen', key = 't3_a4_ev_btn'):
        a4_ev_result = a4_ev(text,masks,a4_input)
        if not a4_ev_result:
            fba4 = '>**Lösung korrekt.**'
        elif len(a4_ev_result) == 1:
            fba4 = f'>**Fehler in der Eingabe für Iteration** {a4_ev_result[0]+1}**.**'
        else:
            a4_ev_result_string = ','.join([str(i+1) for i in a4_ev_result[:-1]])
            a4_ev_result_string_last = str(a4_ev_result[-1]+1)
            fba4 = f'>**Fehler in den Eingaben für die Iterationen** {a4_ev_result_string} **und** {a4_ev_result_string_last}.'
        st.markdown(fba4)

@st.cache
def a1_draw_nfa(p):
    d = gv.Digraph()
    m = len(p)
    d.graph_attr["rankdir"] = "LR"
    d.node_attr["shape"] = "circle"
    d.node(str(-1), style='filled',fillcolor='#a5a6f9')
    d.edge(str(-1), str(-1), label = chr(120506))
    for i in range(1,m-1):
        d.node(str(i-1))
        d.edge(str(i-1),str(i), label = p[i])
    if len(p) > 0: 
        d.edge(str(-1),str(0),label = p[0])
        d.node(str(m-1),style = 'filled', fillcolor='#e76960')
        d.edge(str(m-2),str(m-1),label = p[m-1])
    return d

def a2_create_masks(pattern):
    masks = {c : 0 for c in [chr(i) for i in range(97,123)]}
    for i,c in enumerate(pattern):
        masks[c] |= (1<<i)
    masks['accept'] = (1 << len(pattern)-1)
    return masks

def ev_a2(a2_indexes, a2_input,text,masks,m):
    wrongInputs = []
    for k in range(len(a2_input)):
        if a2_input[k] and a2_input[k][0] == '{' and a2_input[k][-1] == '}':
            a2_input[k] = a2_input[k][1:-1]
    for k,i in enumerate(a2_input):
        if not set(re.split('\s*,\s*|\s*;\s*|\s+',i)) == set(map(str,shift_and_active_to_list(shift_and_return_active(text[:a2_indexes[k]],masks),m))): wrongInputs.append(k)
    return wrongInputs

def ev_a3(masks, masksInput,m):
    wrongInputs = []
    for c in masksInput.keys():
        if not masks[c] == string_base2_to_int(masksInput[c]) or not len(masksInput[c]) == m:
            wrongInputs.append(c)
    return wrongInputs

def a3_ev_katex(a3_ev):
    klist = []
    for c in a3_ev:
        if len(c) == 1:
            klist.append(r'\textit{\textsf{mask}}^\texttt '+c)
        else: klist.append(r'\textit{\textsf{accept}}')
    return klist

def a4_ev(text,masks,input):
    active_list = shift_and_return_active_list(text,masks)
    wrongInputs = []
    for i in range(len(input)):
        if not string_base2_to_int(input[i]) == active_list[i] or not len(input[i]) == np.log2(masks['accept'])+1:
            wrongInputs.append(i)
    return wrongInputs

def string_base2_to_int(s):
    for c in s:
        if not ord(c) == 48 and not ord(c) == 49: return -1
    if not s: return -1
    return int(s,2)

def shift_and_return_active(text, masks):
    A = 0
    for c in text:
        A = ((A<<1) | 1) & masks[c]
    return A

def shift_and_return_active_list(text, masks):
    A = 0
    active_list = []
    for c in text:
        A = ((A<<1) | 1) & masks[c]
        active_list.append(A)
    return active_list

def shift_and_active_to_list(active, m):
    active_list = [-1]
    for i in range(m):
        if (active&1) == 1: active_list.append(i)
        active >>= 1
    return active_list
