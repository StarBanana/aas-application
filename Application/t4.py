import streamlit as st
import numpy as np
import graphviz as gv
import re
import t3
import time

def app(seed,sidebar_top):
    #parameters
    sigma_l = 97
    sigma_r = 100
    #generate random text
    rg = np.random.default_rng(seed)
    tlength = rg.integers(12,24)
    t = rg.integers(sigma_l,sigma_r,tlength)
    text = ''.join([chr(a) for a in t])

    #select random substring from text as pattern
    plength = rg.integers(4,9)
    pstart = rg.integers(0,len(text) - plength)
    pattern = text[pstart:pstart+plength]
    descr = 'Gegeben sei der Text $T=\\texttt{' + text + '}$ und das Muster $P=\\texttt{' + pattern + '}$, sowie der NFA für $P$:'
    st.write(descr)
    st.write(t3.a1_draw_nfa(pattern))
    sidebar_top.info('$T=\\texttt{' + text + '}$\n\r$P=\\texttt{' + pattern + '}$')
    #A1
    st.subheader('Aufgabe 1')
    st.markdown(r'Zeichnen Sie den DFA für das Muster $P$, ohne Kanten, die auf den Startzustand zeigen. Nehmen Sie als Ausgangsbasis den oben dargestellten NFA für $P$.')
#    if st.checkbox('Lösung anzeigen', key = ' t4_a1_check'):
#            st.write(a1_draw_dfa(pattern,a1_create_lps(pattern)))
    
    a1_exp = st.beta_expander('Lösung anzeigen')
    with a1_exp:
        st.write(a1_draw_dfa(pattern,a1_create_lps(pattern)))

    #A2
    st.subheader('Aufgabe 2')
    st.markdown(r'Geben Sie die lps-Funktion zu $P$ an.')
    a2_input_lps = []
    for i in range(plength):
        a2_input_lps.append(st.text_input(f'lps[{i}]','',1,key= 't4a2'+str(i)))
    if st.button('Aufgabe überprüfen', key = 't4_a2_ev_btn'):
        a2_ev_result = a2_ev(pattern, a2_input_lps)
        if not a2_ev_result:
            fba2 = '>**Lösung korrekt.**'
        elif len(a2_ev_result) == 1:
            fba2 = f'>**Fehler in der Eingabe für** lps[{str(a2_ev_result[0])}]**.**'
        else:
            fba2 = '>**Fehler in den Eingaben für **' + ', '.join([f'lps[{str(i)}]' for i in a2_ev_result[:-1]]) + f'** und **lps[{a2_ev_result[-1]}]**.**' 
        st.markdown(fba2)

    #A3
    st.subheader('Aufgabe 3')
    st.write(r'Führen Sie den Knuth-Morris-Pratt-Algorithmus mit dem Muster $P$ auf dem Text $T$ aus.')
    st.write('Im KMP-Algorithmus wird für jedes Textzeichen $T\\lbrack i\\rbrack =$ c die Funktion `delta` aufgerufen. Geben Sie für jeden Aufruf alle Werte an, die die Variable q annimmt (beginnend mit dem Wert für q mit dem die Funktion aufgerufen wird).')
    st.code('''
        def delta(q,c,P,lps):
            m = len(P)
            while q == m-1 or (P[q+1] != c and q > -1):
                q = lps[q] - 1
            if P[q+1] == c:
                q += 1
            return q
    ''',language='python')
    a3_input = []
    for k,c in enumerate(text):
        a3_input.append(st.text_input("T["+str(k)+"] = c = '"+c+"'", '', key = 't4_a3' + str(k)))
    if st.button('Aufgabe überprüfen', key = 't4_a3_ev_btn'):
        a3_ev_result = a3_ev(text,pattern,a3_input)
        if not a3_ev_result:
            fba3 = '>**Lösung korrekt.**'
        elif len(a3_ev_result) == 1:
            fba3 = f'>**Fehler in der Eingabe für** T[{str(a3_ev_result[0])}]**.**'
        else:
            fba3 = '>**Fehler in den Eingaben für **' + ', '.join([f'T[{str(i)}]' for i in a3_ev_result[:-1]]) + f'** und **T[{a3_ev_result[-1]}]**.**' 
        st.markdown(fba3)

@st.cache
def a1_create_lps(p):
    m = len(p)
    lps = [0]
    for i in range(1,m):    
        q= lps[i-1]-1
        while q == m-1 or (p[q+1] != p[i] and q > -1):
            q = lps[q]-1
        if p[q+1] == p[i]:
            lps.append(q+2)
        else:
            lps.append(0)
    return lps

def a1_draw_dfa(p,lps):
    d = gv.Digraph()
    m , set_p = len(p), set(p)
    d.graph_attr['rank'] = 'same'
    d.graph_attr['rankdir'] = 'LR'
    d.node_attr['shape'] = 'circle'
    d.node(str(-1), style='filled',fillcolor='#a5a6f9')
    d.edge(str(-1), str(-1), label = chr(120506)+r'\\{'+p[0]+'}')
    if len(p) > 0:
        d.node(str(m-1),style = 'filled', fillcolor='#e76960')
    for i in range(0,m): 
        d.node(str(i))
        d.edge(str(i-1),str(i), label = p[i])
        if i != m-1 : set_p.remove(p[i+1])
        for c in set_p:
            q = i
            while q == m-1 or (c != p[q+1] and q > -1):
                q = lps[q] - 1
            if c == p[q+1]:
                d.edge(str(i),str(q+1),c, constraint = 'false')
        if i != m-1 : set_p.add(p[i+1])
    return d

def a2_ev(p, input):
    wrong_inputs = []
    lps = a1_create_lps(p)
    for k,(i,l) in enumerate(zip(input,lps)):
        try:
            if not int(i) == l:
                wrong_inputs.append(k)
        except ValueError:
            wrong_inputs.append(k)
    return wrong_inputs

def a3_kmp(t,p,lps):
    list_q = []
    m = len(p)
    q = -1
    for c in t:
        ql = [q]
        while q == m-1 or (p[q+1] != c and q > -1):
            q = lps[q] -1
            ql.append(q)
        if p[q+1] == c:
            q += 1
            ql.append(q)
        list_q.append(ql)
    return list_q
        
def a3_ev(t,p,input):
    wrong_inputs = []
    lps = a1_create_lps(p)
    list_q = a3_kmp(t,p,lps)
    for k,i in enumerate(input):
        try:            
            if not list_q[k] == list(map(int, re.split('\s*,\s*|\s*;\s*|\s+',i))):
                wrong_inputs.append(k)
        except ValueError:
            wrong_inputs.append(k)
    return wrong_inputs