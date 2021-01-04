import graphviz
from numpy.core.numerictypes import find_common_type
import streamlit as st
import pandas as pd
import numpy as np
import graphviz as gv
import re
import streamlit.components.v1 as components
import copy

st.set_page_config(page_title = 'Algorithmen auf Sequenzen')

#utility functions
def ut_check_task(wrong_inputs, solution, key, opt_string_sgl = '', opt_string_pl = ''):    
    check_exp = st.beta_expander('Aufgabe überprüfen')
    wrong_inputs_len = len(wrong_inputs)
    with check_exp:
        if not wrong_inputs:
            st.markdown('**Lösung korrekt.**')            
        else:
            if wrong_inputs_len == 1:
                fb = f'**Fehler in der Eingabe für {opt_string_sgl} **{wrong_inputs[0]}**.**'
            else:
                a = ', '.join(wrong_inputs)
                fb = f'**Fehler in den Eingaben für {opt_string_pl} **{a}**.**'
            st.markdown(fb)
            if st.button('Lösung anzeigen', key=key):
                if(type(solution) == pd.DataFrame):
                    st.table(solution)
                else: st.write(solution)

def ut_int_to_base2_string(a, min_length = 0): 
    sl = []
    while a:
        if a & 1:
            sl.append('1')
        else: 
            sl.append('0')
        a >>= 1
    while len(sl) < min_length:
        sl.append('0')
    sl.reverse()
    return ''.join(sl)

@st.cache
def ut_rnd_text_and_pattern(seed, sigma_l, sigma_r):
    rg = np.random.default_rng(seed)
    #generate random text
    tlength = rg.integers(12,21)
    t = rg.integers(sigma_l,sigma_r,tlength)
    text = ''.join([chr(a) for a in t]) 
    tlength = len(text)
    text_tex = f'$T=\\mathtt{{{text}}}$'
    #select random substring from text as pattern
    plength = rg.integers(4,9)
    pstart = rg.integers(0,len(text) - plength)
    pattern = text[pstart:pstart+plength]
    pattern_tex = f'$P=\\mathtt{{{pattern}}}$'
    return (text, text_tex, tlength, pattern, pattern_tex, plength)

def ut_rnd_string(seed, sigma_l, sigma_r, min_length, max_length, sentinel = False):
    rg = np.random.default_rng(seed)
    #generate random text
    if sentinel:
        tlength = rg.integers(min_length,max_length)
        t = rg.integers(sigma_l,sigma_r,tlength)
        text = ''.join([chr(a) for a in t] + ['$'])
        text_t = ''.join([chr(a) for a in t] + ['\\$'])
        text_tex = f'$T=\\mathtt{{{text_t}}}$'
    else:
        tlength = rg.integers(min_length,max_length+1)
        t = rg.integers(sigma_l,sigma_r,tlength)
        text = ''.join([chr(a) for a in t])
        text_tex = f'$T=\\mathtt{{{text}}}$'
    
    return (text, text_tex, tlength)

def ut_string_base2_to_int(s):
    for c in s:
        if not ord(c) == 48 and not ord(c) == 49: return -1
    if not s: return -1
    return int(s,2)

def home(seed,sidebar_top):
    st.write('---text---')

    st.subheader('Hinweise')
    st.markdown("""
        - Bei Eingaben von Folgen oder Mengen können die Elemente durch die Zeichen ' ', ',' oder ';' separiert werden.
        - Bei Eingaben von Mengen sind die Mengenklammern '{', '}' optional.
        - Strings werden zufällig mithilfe eines Seeds generiert.
        - Die Indizierung von Strings beginnt an Position 0.
    """)

def t1(seed, sidebar_top):

    def t1_a1(text, seed,rg):
        n = len(text)    
        st.subheader('Aufgabe 1')    
        inputs = []
        prefix_len = rg.integers(3,n-2)
        inputs.append(st.text_input(f'a) Geben Sie das Präfix von T der Länge {prefix_len} an.','',key = 'T1_a1_1'))
        suffix_len = rg.integers(3,n-2)
        inputs.append(st.text_input(f'b) Geben Sie das Suffix von T der Länge {suffix_len} an.','',key = 'T1_a1_2'))        
        substring_len = rg.integers(3,len(text)/2)
        substring_pos = rg.integers(3,n-substring_len) 
        inputs.append(st.text_input(f'c) Geben Sie den Teilstring von T der Länge {substring_len}, der an Position {substring_pos} beginnt an.','',key = 'T1_a1_3'))
        solution = [text[:prefix_len], text[n-suffix_len:], text[substring_pos:substring_pos+substring_len]]
        solution_df = pd.DataFrame(solution,index = [f'Präfix von T der Länge {prefix_len}',f'Suffix von T der Länge {suffix_len}',f'Teilstring von T der Länge {substring_len}, der an Position {substring_pos} beginnt'], columns = ['Lösung'])
        wrong_inputs = []
        for i in range(3):
            if inputs[i] != solution[i]:
                wrong_inputs.append(f'{chr(97+i)})')
        ut_check_task(wrong_inputs, solution_df, key = 'T1_a1_check_task')

    #constants
    SIGMA_L = 97
    SIGMA_R = 101
    
    #generate random string    
    rg = np.random.default_rng(seed)    
    tlength = rg.integers(12,21)
    t = rg.integers(SIGMA_L,SIGMA_R,tlength)
    text = ''.join([chr(a) for a in t]) 
    text_tex = f'$T=\\texttt{{{text}}}$'
    tlength = len(text)
    sidebar_top.info(text_tex)
    st.write(f'Gegeben Sei der String {text_tex}.')

    #tasks
    t1_a1(text,seed,rg)

def t2(seed,sidebar_top):
    def t2_a1(text,pattern):
        def number_of_comparisons(text,pattern):
            count = 0
            n = len(text)
            m = len(pattern)
            for i in range(n):
                for j in range(m):
                    count += 1
                    if text[i+j] != pattern[j]: 
                        break                    
            return count

        st.subheader('Aufgabe 1')
        st.write('Bestimmen Sie die Anzahl der Vergleiche, die bei der Mustersuche mit dem naiven Algorithmus mit dem Muster $P$ auf dem Text $T$ benötigt werden.')
        a1_input = st.text_input('Anzahl der Vergleiche', '', key = 'T2_a1_input')        
        solution = number_of_comparisons(text,pattern)
        try:
            if int(a1_input) != solution:
                wrong_inputs = ['die Anzahl der Vergleiche']
            else: wrong_inputs = []
        except ValueError:
            wrong_inputs = ['die Anzahl der Vergleiche']
        ut_check_task(wrong_inputs, f'Es werden {solution} Vergleiche benötigt.', key = 'T2_a1_check_task')
    
    def t2_a2(pattern, SIGMA_L, SIGMA_R):
        st.subheader('Aufgabe 2')
        st.write('Geben Sie die erwartete Anzahl an Vergleichen... (Muster)  ')
        a2_input = st.text_input('Erwartete Anzahl an Vergleichen', '', key = 'T2_a2_input').replace(',','.')
        p = 1 / (SIGMA_R-SIGMA_L-1)
        m = len(pattern)
        solution = (m*p**m) + ((1-p) * np.sum([ j * p**(j-1) for j in range(m)]))
        try:
            input_float = float(a2_input)
            if input_float < 0.95*solution or input_float > 1.05*solution:
                wrong_inputs = ['die Anzahl der Vergleiche']
            else: wrong_inputs = []
        except ValueError:
            wrong_inputs = ['die Anzahl der Vergleiche']
        ut_check_task(wrong_inputs, f'Die erwartete Anzahl an Vergleichen beträgt {solution}.', key = 'T2_a2_check_task')

    #constants
    SIGMA_L = 97
    SIGMA_R = 101
    #random text, pattern:
    text, text_tex, tlength, pattern, pattern_tex, plength = ut_rnd_text_and_pattern(seed,SIGMA_L,SIGMA_R)        
    #description:
    st.write(f'Gegeben sei der Text {text_tex} und das Muster {pattern_tex}.')
    sidebar_top.info(f'{text_tex}\n\r{pattern_tex}')

    t2_a1(text,pattern)
    t2_a2(pattern,SIGMA_L,SIGMA_R)

@st.cache
def t3_a1_draw_nfa(p):
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

def t3_shift_and_indices(indices, text, masks):
                a = 0
                l = []
                for i,c in enumerate(text):
                    a = ((a<<1) | 1) & masks[c]                    
                    if i in indices:
                        l.append(a)
                return l 

def t3_a2_create_masks(pattern,sigma_l,sigma_r):
    masks = {c : 0 for c in [chr(i) for i in range(sigma_l,sigma_r)]}
    for i,c in enumerate(pattern):
        masks[c] |= (1<<i)
    masks['accept'] = (1 << len(pattern)-1)
    return masks

def t3_a3_ev(masks,masksInput,m):
    def t3_a3_ev_katex(a3_ev):
        klist = []
        for c in a3_ev:
            if len(c) == 1:
                klist.append(r'$\textit{\textsf{mask}}^\texttt '+c+'$')
            else: klist.append(r'$\textit{\textsf{accept}}$')
        return klist
    wrongInputs = []
    for c in masksInput.keys():
        if not masks[c] == ut_string_base2_to_int(masksInput[c]):
            wrongInputs.append(c)
    return t3_a3_ev_katex(wrongInputs)

def t3(seed,sidebar_top):
    #constants
    SIGMA_L = 97
    SIGMA_R = 101
    #random text, pattern:
    text, text_tex, tlength, pattern, pattern_tex, plength = ut_rnd_text_and_pattern(seed,SIGMA_L,SIGMA_R)
    #description:
    st.write(f'Gegeben sei der Text {text_tex} und das Muster {pattern_tex}.')
    sidebar_top.info(f'{text_tex}\n\r{pattern_tex}')
    
    masks = t3_a2_create_masks(pattern,SIGMA_L,SIGMA_R)

    def t3_a1(text, pattern):
        st.subheader('Aufgabe 1')
        descr11 = 'Zeichnen Sie den NFA zu $P$.'
        st.markdown(descr11)
        a1_exp = st.beta_expander('Lösung anzeigen')
        with a1_exp:
            st.write(t3_a1_draw_nfa(pattern))

    def t3_a2_shift_and_return_active(text, masks):
        A = 0
        for c in text:
            A = ((A<<1) | 1) & masks[c]
        return A

    def t3_a2_shift_and_return_active_list(text, masks):
        A = 0
        active_list = []
        for c in text:
            A = ((A<<1) | 1) & masks[c]
            active_list.append(A)
        return active_list

    def t3_a2_shift_and_active_to_list(active, m):
        active_list = [-1]
        for i in range(m):
            if (active&1) == 1: active_list.append(i)
            active >>= 1
        return active_list
        
    def t3_a2_ev(a2_indices, a2_input,text,masks,m):
        wrongInputs = []
        for k in range(len(a2_input)):
            if a2_input[k] and a2_input[k][0] == '{' and a2_input[k][-1] == '}':
                a2_input[k] = a2_input[k][1:-1]
        for k,i in enumerate(a2_input):
            if not set(re.split('\s*,\s*|\s*;\s*|\s+',i)) == set(map(str,t3_a2_shift_and_active_to_list(t3_a2_shift_and_return_active(text[:a2_indices[k]],masks),m))): wrongInputs.append(k)
        return wrongInputs

    def t3_a2(text,pattern,sigma_l,sigma_r):
        st.subheader('Aufgabe 2')
        descr21 = 'Geben Sie die aktive Zustandsmenge des NFA nach Lesen der folgenden Präfixe von $T$ an.'
        st.markdown(descr21)
        rg = np.random.default_rng(seed)
        a2_indices = sorted(rg.choice(np.arange(4,tlength),3,replace = False))
        a2_input = []
        for k,i in enumerate(a2_indices):
            st.markdown(f'$T\\lbrack..{i-1}\\rbrack=\\texttt{{{text[:i]}}}$')
            a2_input.append(st.text_input('',key='t3a2'+str(k)))
        a2_solution = list(map(lambda x : t3_a2_shift_and_active_to_list(x, plength),map(lambda x : t3_a2_shift_and_return_active(x,masks),map(lambda x : text[:x], a2_indices))))
        a2_solution_string = map(lambda x : f'{{{x}}}',map(','.join, map(lambda  x : map(str,x), a2_solution)))
        a2_solution_write = pd.DataFrame([a2_solution_string], columns = list(map(lambda x : f'T[..{str(x-1)}]', a2_indices)), index = ['Zustandsmenge'])
        a2_wrong_inputs = [ f'$T\\lbrack :{a}\\rbrack$' for a in list(map(lambda x : a2_indices[x]-1,t3_a2_ev(a2_indices, a2_input, text, masks, plength)))]
        ut_check_task(a2_wrong_inputs, a2_solution_write, key = 't3_a2')

    def t3_a3(text,pattern):
        st.subheader('Aufgabe 3')
        descr31 = 'Erstellen Sie für den Shift-And-Algorithmus die Masken für $P$ über dem Alphabet $\Sigma = \{\\mathtt ' + ',\\mathtt '.join(sorted(set(pattern))) + '\}$.'
        st.markdown(descr31)
        
        masksInput = dict()
        sorted_set_pattern = sorted(set(pattern))
        for c in sorted_set_pattern:
            st.markdown('$\\textit{\\textsf{mask}}^\mathtt{'+c+'}$')
            masksInput[c] = st.text_input('','',len(pattern),'t3a3'+c)
        st.markdown(r'$\textit{\textsf{accept}}$')
        masksInput['accept'] =  st.text_input('','',len(pattern))
        a3_ev = t3_a3_ev(masks,masksInput,plength)
        a3_solution = pd.DataFrame([ [ ut_int_to_base2_string(masks[c], plength) , '' ] for c in sorted_set_pattern + ['accept'] ], index = [ f'mask[{c}]' for c in sorted_set_pattern ] + ['accept'], columns = ['Bitmuster', ''] )
        ut_check_task(a3_ev, a3_solution, key = 't3_a3')

    def t3_a4_ev(text,input):
        active_list = t3_a2_shift_and_return_active_list(text,masks)
        wrongInputs = []
        for i in range(len(input)):
            if not ut_string_base2_to_int(input[i]) == active_list[i]:
                wrongInputs.append(i)
        return wrongInputs

    def t3_a4(text,pattern):
        def t3_a4_ev(uinput,solution):
            wrong_inputs = []
            for k,i in enumerate(uinput):
                if ut_string_base2_to_int(i) != solution[k]:
                    wrong_inputs.append(k)
            return wrong_inputs
        st.subheader('Aufgabe 4')
        text_sep = (tlength-2)/3
        rg = np.random.default_rng(seed)
        indices = [rg.integers(2,text_sep), rg.integers(text_sep, 2*text_sep), rg.integers(2*text_sep, 3*text_sep)]        
        st.markdown(f'Führen Sie eine Mustersuche für das Muster $P$ mit dem Shift-And-Algorithmus auf dem Text $T$ aus. Geben Sie für den Shift-And-Algorithmus das Bitmuster nach Lesen der Textzeichen an den Positionen {indices[0]}, {indices[1]} und {indices[2]} an.')
        uinput = []
        for i in indices:
            uinput.append(st.text_input(f'Bitmuster nach Lesen von T[{i}]','',plength, key = f't3_a4_input{i}'))
        solution = t3_shift_and_indices(indices, text, masks)
        wrong_inputs = t3_a4_ev(uinput, solution)
        solution_df = pd.DataFrame(map(lambda x : ut_int_to_base2_string(x,plength), solution), index = indices, columns = ['Bitmuster'])
        ut_check_task(list(map(lambda x : str(indices[x]),wrong_inputs)), solution_df, opt_string_sgl='das Bitmuster an Position', opt_string_pl='die Bitmuster an den Positionen', key = 't7_gen_a3_check_task')
    
        
            

    #excercises
    t3_a1(text,pattern)
    t3_a2(text,pattern,SIGMA_L,SIGMA_R)
    t3_a3(text,pattern)
    t3_a4(text,pattern)

def t4(seed,sidebar_top):
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

    def a2_ev(p, input, lps):
        wrong_inputs = []
        for k,(i,l) in enumerate(zip(input,lps)):
            try:
                if not int(i) == l:
                    wrong_inputs.append(k)
            except ValueError:
                wrong_inputs.append(k)
        return wrong_inputs

    @st.cache
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
    st.write(t3_a1_draw_nfa(pattern))
    sidebar_top.info('$T=\\texttt{' + text + '}$\n\r$P=\\texttt{' + pattern + '}$')
    #A1
    st.subheader('Aufgabe 1')
    st.markdown(r'Zeichnen Sie den DFA für das Muster $P$, ohne Kanten, die auf den Startzustand zeigen. Nehmen Sie als Ausgangsbasis den oben dargestellten NFA für $P$.')
    #if st.checkbox('Lösung anzeigen', key = ' t4_a1_check'):
    #st.write(a1_draw_dfa(pattern,a1_create_lps(pattern))   
    lps = a1_create_lps(pattern)
    a1_exp = st.beta_expander('Lösung anzeigen')
    with a1_exp:
        st.write(a1_draw_dfa(pattern,a1_create_lps(pattern)))

    #A2
    st.subheader('Aufgabe 2')
    st.markdown(r'Geben Sie die lps-Funktion zu $P$ an.')
    a2_input_lps = []
    for i in range(plength):
        a2_input_lps.append(st.text_input(f'lps[{i}]','',1,key= 't4a2'+str(i)))
    a2_wrong_inputs = [f'lps[{a}]' for a in a2_ev(pattern, a2_input_lps, lps)]
    a2_solution = pd.DataFrame([[str(a), '', ''] for a in lps], index = [f'lps[{i}]' for i in range(len(lps))], columns = ['','',''])
    ut_check_task(a2_wrong_inputs, a2_solution,key = 't4_a2')
    
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
    a3_wrong_inputs = [ f'T[{str(a)}]' for a in a3_ev(text,pattern,a3_input)]
    a3_solution = a3_kmp(text,pattern,lps)
    a3_solution_write = pd.DataFrame([ [','.join(a)] for a in map(lambda x : map(str,x),a3_solution) ], index = map(lambda x: f'T[{x}]' ,range(len(a3_solution))), columns = ['Werte für q'])
    a3_ev_result = a3_ev(text,pattern,a3_input)
    ut_check_task(a3_wrong_inputs, a3_solution_write, key = 't4_a3')
    #if st.button('Aufgabe überprüfen', key = 't4_a3_ev_btn'):
    #    a3_ev_result = a3_ev(text,pattern,a3_input)
    #    if not a3_ev_result:
    #        fba3 = '>**Lösung korrekt.**'
    #    elif len(a3_ev_result) == 1:
    #        fba3 = f'>**Fehler in der Eingabe für** T[{str(a3_ev_result[0])}]**.**'
    #    else:
    #        fba3 = '>**Fehler in den Eingaben für **' + ', '.join([f'T[{str(i)}]' for i in a3_ev_result[:-1]]) + f'** und **T[{a3_ev_result[-1]}]**.**' 
    #    st.markdown(fba3)

def t5(seed, sidebar_top):
    SIGMA_L = 97
    SIGMA_R = 100
    sigma = [ chr(i) for i in range(SIGMA_L,SIGMA_R) ]    
    #random text, pattern:
    text, text_tex, tlength, pattern, pattern_tex, plength = ut_rnd_text_and_pattern(seed,SIGMA_L,SIGMA_R)

    def a1_ev(sigma_p,shifts, input):
        wrong_inputs = []
        for i in range(len(input)):
            try:
                if not int(input[i]) == shifts[sigma_p[i]]:
                    wrong_inputs.append(i)
            except ValueError:
                wrong_inputs.append(i)
        return wrong_inputs

    def a1_create_shifts(pattern, sigma):
        m = len(pattern)
        shifts = {s:m for s in sigma}
        for i,c in enumerate(pattern[:m-1]):
            shifts[c] = m-1-i
        return shifts

    def a2_horspool(text, pattern, sigma, tlength, plength):
        shifts = a1_create_shifts(pattern,sigma)
        lastP = pattern[-1]
        last = plength-1
        shifts_e = []
        while True:
            while last < tlength:
                s = shifts[text[last]]
                last += s
                shifts_e.append(s)
            if last >= tlength:
                break
        return shifts_e

    def a2_ev(input_shifts_e, input_lasts ,shifts_e,plength):
        wrong_inputs_shifts_e = []
        wrong_inputs_lasts = []
        last = plength - 1
        for k,in_shift, in_last in zip(range(0,len(input_shifts_e)),input_shifts_e,input_lasts):
            last += shifts_e[k]
            try:
                if not int(input_shifts_e[k]) == shifts_e[k]:
                    wrong_inputs_shifts_e.append(k+1)
            except ValueError:
                wrong_inputs_shifts_e.append(k+1)
            try:
                if not int(input_lasts[k]) == last:
                    wrong_inputs_lasts.append(k+1)
            except ValueError:
                wrong_inputs_lasts.append(k+1)
        return (wrong_inputs_shifts_e,wrong_inputs_lasts)

    def a3_draw_suffix_automaton(pattern):
        m = len(pattern)
        d = gv.Digraph()    
        d.graph_attr["rankdir"] = "LR"
        d.node_attr["shape"] = "circle"
        d.node('start',style='filled',fillcolor='#a5a6f9')
        d.node(str(0))
        d.edge('start',str(0),label=u'\u03b5')
        for i in range(1,m):
            d.node(str(i))
            d.edge(str(i-1), str(i), label = pattern[-i])
            d.edge('start',str(i),label= u'\u03b5')
        if m > 1:
            d.node(str(m), style = 'filled', fillcolor='#e76960')
            d.edge(str(m-1), str(m), label = pattern[-m])
            d.edge('start',str(m),label= u'\u03b5')
        return d

    def a4_bndm(text, pattern, tlength, plength):
        shifts = []
        full = (1 << plength) - 1
        accept = 1 << (plength-1)    
        last = plength
        masks = t3_a2_create_masks(pattern[:-1],SIGMA_L,SIGMA_R)
        while last <= tlength:
            A = full
            lastsuffix = 0
            j = 1
            while A:
                A &= masks[text[last-j]]
                if A & accept:
                    if j == plength:
                        break
                    else:
                        lastsuffix = j
                j += 1
                A <<= 1
            shift = plength-lastsuffix
            last += shift
            shifts.append(shift)
        return shifts

 
    #generate random text
    rg = np.random.default_rng(seed)
    tlength = int(rg.integers(16,21))
    t = rg.integers(SIGMA_L,SIGMA_R,tlength)
    text = ''.join([chr(a) for a in t])
    sigma_t = sorted(set(text))

    #select random substring from text as pattern
    plength = rg.integers(4,9)
    pstart = rg.integers(0,len(text) - plength)
    pattern = text[pstart:pstart+plength]
    descr = 'Gegeben sei der Text $T=\\texttt{' + text + '}$ und das Muster $P=\\texttt{' + pattern + '}$.'
    st.write(descr)
    sidebar_top.info('$T=\\texttt{' + text + '}$\n\r$P=\\texttt{' + pattern + '}$')
    
    #A1
    st.subheader('Aufgabe 1')
    sigma_p = sorted(set(pattern))
    st.write('Erstellen Sie die Sprungtabelle für den Horspool-Algorithmus für das Muster $P$ über dem Alphabet $\Sigma = \{\mathtt ' + ',\\mathtt '.join(sigma_t) + '\}$.')
    a1_input = []
    shifts = a1_create_shifts(pattern,sigma)
    for a in sigma_t:
        a1_input.append(st.text_input(f'shift[{a}]','',key = f't5_a1{a}'))
    a1_wrong_inputs = [ f'shift[{sigma_t[a]}]' for a in a1_ev(sigma_p,shifts,a1_input) ]
    a1_solution = pd.DataFrame([[str(shifts[a]), '', ''] for a in shifts.keys()], index = [f'shift[{k}]' for k in shifts.keys()], columns = ['','',''])
    ut_check_task(a1_wrong_inputs, a1_solution,key = 't5_a1')

    #A2
    st.subheader('Aufgabe 2')
    st.markdown('Führen Sie den Horspool-Algorithmus mit dem Muster $P$ auf dem Text $T$ aus. Geben Sie die Längen der Sprünge an, sowie die Textpositionen, an denen das letzte Zeichen des Musters nach Ausführung der Sprünge steht (im Algorithmus aus der Vorlesung sind dies die Werte für die Variable `last`).')
    
    a2_vis = st.beta_container()
    a2_shifts_e = a2_horspool(text, pattern, sigma, tlength, plength)
    lasts = []
    last_last = plength-1
    for s in a2_shifts_e:
        lasts.append(last_last+s)        

    a2_input_shifts = []
    a2_input_lasts = []
    for k,i in enumerate(a2_shifts_e):
        cols = st.beta_columns([8,1,8])
        a2_input_shifts.append(cols[0].text_input(f'Iteration {k+1}: Sprunglänge','',key=f't5_a2_input_shifts_{k}'))
        a2_input_lasts.append(cols[2].text_input(f'Iteration {k+1}: last','',key=f't5_a2_input_lasts_{k}'))

    a2_last_filled_index = plength-1
    for k in range(len(a2_shifts_e)):
        try:
            if int(a2_input_shifts[k]) >= 0 and int(a2_input_lasts[k]) >= 0:
                a2_last_filled_index = int(a2_input_lasts[k])
            else: break
        except ValueError:
            break

    a2_vis_data_pattern = [ '' for c in text]
    for i,c in zip(range(a2_last_filled_index-plength+1, a2_last_filled_index+1),pattern):
        if i > tlength - 1 : break
        a2_vis_data_pattern[i] = c

    a2_vis_data = [list(text), a2_vis_data_pattern]
    a2_vis_df = pd.DataFrame(a2_vis_data,index = ['T','P'], columns = range(tlength) )
    with a2_vis:
        st.table(a2_vis_df)

    a2_check_task = st.beta_expander('Aufgabe überprüfen')
    with a2_check_task:
        a2_shifts_ev_result,a2_lasts_ev_result = a2_ev(a2_input_shifts,a2_input_lasts,a2_shifts_e,plength)
        a2_shifts_ev_result_len, a2_lasts_ev_result_len = len(a2_shifts_ev_result), len(a2_lasts_ev_result)
        if not a2_shifts_ev_result and not a2_lasts_ev_result:
            fba2 = '**Lösung korrekt.**'
        elif a2_shifts_ev_result_len == 1 and a2_lasts_ev_result_len == 0:
            fba2 = '**Fehler in der Eingabe für** Iteration '+str(a2_shifts_ev_result[0])+': Sprunglänge**.**'
        elif a2_shifts_ev_result_len == 0 and a2_lasts_ev_result_len == 1:
            fba2 = '**Fehler in der Eingabe für** Iteration '+str(a2_shifts_ev_result[0])+': last**.**' 
        else:
            a2_shifts_out = []
            a2_lasts_out = []
            for k in a2_shifts_ev_result:
                a2_shifts_out.append(f'- Iteration {k}: Sprunglänge\n\r')
            for k in a2_lasts_ev_result:
                a2_shifts_out.append(f'- Iteration {k}: last\n\r')
            a2_shifts_out_str = ''.join(a2_shifts_out)
            a2_lasts_out_str = ''.join(a2_lasts_out)
            fba2 = '>**Fehler in den Eingaben für:**\n\r '+ ''.join(a2_shifts_out) + ''.join(a2_lasts_out)
        st.markdown(fba2)
        a2_solution = pd.DataFrame([ [str(s),str(l)] for s,l in zip(a2_shifts_e,lasts) ], index = [ f'Iteration {i+1}' for i in range(len(a2_shifts_e))], columns = ['shift', 'last'])
        if st.button('Lösung anzeigen', key = 't5_a2_btn'):
            st.table(a2_solution)

    #A3
    st.subheader('Aufgabe 3')
    st.write('Zeichnen Sie den Suffixautomaten, der im BNDM-Algorithmus zur Mustersuche mit dem Muster $P$ auf dem Text $T$ simuliert wird.')
    a3_exp = st.beta_expander('Lösung anzeigen')
    with a3_exp:
        st.write(a3_draw_suffix_automaton(pattern))

    #A4
    st.subheader('Aufgabe 4')
    st.write('Führen Sie den BNDM-Algorithmus mit dem Muster $P$ auf dem Text $T$ aus. Geben Sie die Länge der Sprünge an, sowie die Textpositionen, an denen das letzte Zeichen des Musters nach Ausführung der Sprünge steht.')

    a4_vis = st.beta_container()
    a4_shifts_e = a4_bndm(text, pattern, tlength, plength)
    lasts = []
    last_last = plength-1
    for s in a4_shifts_e:
        lasts.append(last_last+s)        

    a4_input_shifts = []
    a4_input_lasts = []
    for k,i in enumerate(a4_shifts_e):
        cols = st.beta_columns([8,1,8])
        a4_input_shifts.append(cols[0].text_input(f'Iteration {k+1}: Sprunglänge','',key=f't5_a4_input_shifts_{k}'))
        a4_input_lasts.append(cols[2].text_input(f'Iteration {k+1}: last','',key=f't5_a4_input_lasts_{k}'))

    a4_last_filled_index = plength-1
    for k in range(len(a4_shifts_e)):
        try:
            if int(a4_input_shifts[k]) >= 0 and int(a4_input_lasts[k]) >= 0:
                a4_last_filled_index = int(a4_input_lasts[k])
            else: break
        except ValueError:
            break

    a4_vis_data_pattern = [ '' for c in text]
    for i,c in zip(range(a4_last_filled_index-plength+1, a4_last_filled_index+1),pattern):
        if i > tlength - 1 : break
        a4_vis_data_pattern[i] = c

    a4_vis_data = [list(text), a4_vis_data_pattern]
    a4_vis_df = pd.DataFrame(a4_vis_data,index = ['T','P'], columns = range(tlength) )
    with a4_vis:
        st.table(a4_vis_df)

    a4_check_task = st.beta_expander('Aufgabe überprüfen')
    with a4_check_task:
        a4_shifts_ev_result,a4_lasts_ev_result = a2_ev(a4_input_shifts,a4_input_lasts,a4_shifts_e,plength)
        a4_shifts_ev_result_len, a4_lasts_ev_result_len = len(a4_shifts_ev_result), len(a4_lasts_ev_result)
        if not a4_shifts_ev_result and not a4_lasts_ev_result:
            fba4 = '**Lösung korrekt.**'
        elif a4_shifts_ev_result_len == 1 and a4_lasts_ev_result_len == 0:
            fba4 = '**Fehler in der Eingabe für** Iteration '+str(a4_shifts_ev_result[0])+': Sprunglänge**.**'
        elif a4_shifts_ev_result_len == 0 and a4_lasts_ev_result_len == 1:
            fba4 = '**Fehler in der Eingabe für** Iteration '+str(a4_shifts_ev_result[0])+': last**.**' 
        else:
            a4_shifts_out = []
            a4_lasts_out = []
            for k in a4_shifts_ev_result:
                a4_shifts_out.append(f'- Iteration {k}: Sprunglänge\n\r')
            for k in a4_lasts_ev_result:
                a4_shifts_out.append(f'- Iteration {k}: last\n\r')
            a4_shifts_out_str = ''.join(a4_shifts_out)
            a4_lasts_out_str = ''.join(a4_lasts_out)
            fba4 = '>**Fehler in den Eingaben für:**\n\r '+ ''.join(a4_shifts_out) + ''.join(a4_lasts_out)
        st.markdown(fba4)
        a4_solution = pd.DataFrame([ [str(s),str(l)] for s,l in zip(a4_shifts_e,lasts) ], index = [ f'Iteration {i+1}' for i in range(len(a4_shifts_e))], columns = ['shift', 'last'])
        if st.button('Lösung anzeigen', key = 't5_a4_btn'):
            st.table(a4_solution)

class ACNode ():
    def __init__(self, parent=None, letter=None, depth=0, label=""):
        self.targets = dict()
        self.lps = None
        self.parent = parent
        self.letter = letter
        self.out = []
        self.bfs_order = None        

        if parent == None:
            self.depth = depth
            self.label = label
        else:
            self.depth = parent.depth + 1
            self.label = parent.label + letter

    def delta(self,a):
        q=self
        while q.lps != None and a not in q.targets:
            q = q.lps
        if a in q.targets: 
            q = q.targets[a] 
        return q

    def bfs(self):
        q = [self]
        while len(q) > 0:
            node = q.pop(0)
            yield node
            q.extend(node.targets.values())

def AC_build(P):
    root = ACNode()
    for (i,p) in enumerate(P):
        node = root
        for a in p:
            if a in node.targets:
                node = node.targets[a]
            else:
                newnode = ACNode(parent = node, letter = a)
                node.targets[a] = newnode
                node = newnode
        node.out.append(i)

    for i,node in enumerate(root.bfs()):
        node.bfs_order = i
        if node.parent == None: continue
        node.lps = node.parent.lps.delta(node.letter) if node.depth > 1 else root
        node.out.extend(node.lps.out)
    return root

def t6(seed,sidebar_top):
    #parameters
    sigma_l = 97
    sigma_r = 100
    sigma = [ chr(i) for i in range(sigma_l,sigma_r)]
    #generate random text
    rg = np.random.default_rng(seed)
    tlength = rg.integers(12,24)
    t = rg.integers(sigma_l,sigma_r,tlength)
    text = ''.join([chr(a) for a in t])

    #select random substrings from text as patterns
    plengths = rg.integers(3,5,rg.integers(3,5))
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
    patterns_string = ','.join(patterns)
    sidebar_top.info(f'$T=\\texttt{{{text}}}$\n\r$P = \\{{\\texttt{{{patterns_string}}}\\}}$')

    def t6_a1_accept_mask(plengths):
        a = 1
        acc = 0
        for m in plengths:
            a <<= m
            acc |= a
        return acc>>1
    
         
    def t6_a2_draw_aho_corasick_trie(root):        
        d = gv.Digraph()
        d.graph_attr["rankdir"] = "LR"
        d.node_attr["shape"] = "circle"
        d.node_attr["width"] = "0.6"        
        for node in root.bfs():
            if node.parent == None:
                d.node(str(node.bfs_order),style='filled',fillcolor='#a5a6f9')
            else:
                d.node(str(node.bfs_order))
                if node.out: 
                    d.node(str(node.bfs_order),style = 'filled', fillcolor='#e76960') 
                d.edge(str(node.parent.bfs_order), str(node.bfs_order), label = node.letter)   
                if node.lps != root:
                    d.edge(str(node.bfs_order), str(node.lps.bfs_order), style = 'dashed', constraint = 'false')
        return d

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
    masks = t3_a2_create_masks(conc_pattern, sigma_l, sigma_r)
    masks['accept'] = t6_a1_accept_mask(plengths)
    if st.button('Aufgabe überprüfen', key = 't5_a1_ev_btn'):
        a1_eva = t3_a3_ev(masks,masksInput,len(conc_pattern))
        if not a1_eva:
            fba1 = '**Lösung korrekt.**'
        elif len(a1_eva) == 1:
            fba1 = f'**Fehler in der Eingabe für ${t3_a3_ev_katex(a1_eva)[0]}$.**'
        else:
            a = ','.join(a1_eva[:-1])
            fba1 = f'**Fehler in den Eingaben für {a} und {a1_eva[-1]}.**'
        st.markdown(fba1)

    #A2
    st.subheader('Aufgabe 2')
    st.write('Erstellen Sie für den Aho-Corasick-Algorithmus einen Trie samt lps-Kanten und Ausgabemengen für die Mustermenge $P$.')
    t6_exp = st.beta_expander('Lösung anzeigen')
    ac_root = AC_build(patterns)
    with t6_exp:
        st.write(t6_a2_draw_aho_corasick_trie(ac_root))

    #A3
    st.subheader('Aufgabe 3')
    st.write('Führen Sie den Aho-Corasick-Alogrithmus mit der Mustermenge $P$ auf dem Text $T$ aus. Zeigen Sie, in welchem Zustand sich der Automat bei der Mustersuche nach jedem Textzeichen befindet.')
    q = ac_root
    q_list = []
    for c in text:
        q = q.delta(c)
        q_list.append(q.bfs_order)
    t6_a3_input = st.text_input('Zustandsfolge', key = 't6_a3_input')
    t6_a3_wrong = False
    try:
        t6_a3_input_int = list(map(int,re.split('\s*,\s*|\s*;\s*|\s+',t6_a3_input)))
        if t6_a3_input_int != q_list or not t6_a3_input_int:
            t6_a3_wrong = True
    except ValueError:
        t6_a3_wrong = True
    t6_a3_fb = []
    if t6_a3_wrong:
        t6_a3_fb.append('die Zustandsfolge')    
    t6_a3_solution = pd.DataFrame([','.join(map(str,q_list))], index = [''],columns = ['Zustandsfolge'])
    ut_check_task(t6_a3_fb, t6_a3_solution, key = 't6_a3_check')
    

def t7(seed, sidebar_top):
    #parameters
    sigma_l = 97
    sigma_r = 100
    sigma = [ chr(i) for i in range(sigma_l,sigma_r) ]
    def t7_rad_format(s):
        if s == t7_1:
            return 'Verallgemeinerte Strings'
        else:
            return 'Optionale Zeichen'

    def t7_opt_create_masks(pattern, pattern_opt_letters):
        masks = { c:0 for c in [chr(i) for i in range(sigma_l,sigma_r)]}
        masks['I'], masks['F'], masks['O'] = 0,0,0
        block = False
        for i,c in enumerate(pattern):
            masks[c] |= (1 << i)
            b = pattern_opt_letters[i]
            if b and block:
                masks['O'] |= (1<<i)
            elif b:
                masks['O'] |= (1<<i)
                masks['I'] |= (1<< i-1)
                block = True
            elif not b and block:
                masks['F'] |= (1<< i-1)
                block = False
        masks['accept'] = (1 << len(pattern)-1)
        return masks    

    def a1_ev(uinput, masks, labels):
        wrong_inputs = []
        for k in masks.keys():            
            if masks[k] != ut_string_base2_to_int(uinput[k]):
                wrong_inputs.append(labels[k])
        return wrong_inputs

    def t7_a3_ev(uinput,solution):
        wrong_inputs = []
        for k,i in enumerate(uinput):
            if ut_string_base2_to_int(i) != solution[k]:
                wrong_inputs.append(k)
        return wrong_inputs

    def t7_1():
        #generate random text
        rg = np.random.default_rng(seed)
        tlength = rg.integers(12,24)
        t = rg.integers(sigma_l,sigma_r,tlength) 
        text = ''.join([chr(a) for a in t])

        #select random substring from text as pattern
        plength = rg.integers(6,8)
        pstart = rg.integers(0,len(text) - plength)
        pattern = text[pstart:pstart+plength]
        #create pattern with wildcards, repetitions
        number_of_wildcards = rg.integers(3,4)
        pattern_wildcards_indices = rg.choice(range(1,plength-2), number_of_wildcards, False)
        pattern_wildcards = [ 0 for c in pattern ]
        for i in pattern_wildcards_indices:
            pattern_wildcards[i] = 1
        number_of_repetitions = rg.integers(2,3)
        pattern_repetitions = [(1,1)]
        for i in range(1,plength-1):
            if pattern_wildcards[i] and pattern_repetitions[i-1][1] == 1:
                u = rg.integers(1,3)
                v = rg.integers(u+1,u+3)
                pattern_repetitions.append((u,v))
            else: pattern_repetitions.append((1,1))
        pattern_repetitions.append((1,1))
        pattern_gen_list = []
        for i,c in enumerate(pattern):
            if pattern_wildcards[i]:
                s = '#'
            else:
                s = c
            if pattern_repetitions[i][1] > 1:
                s = f'{s}({pattern_repetitions[i][0]},{pattern_repetitions[i][1]})'
            pattern_gen_list.append(s)
        pattern_gen = ''.join(pattern_gen_list)
        pattern_gen_tex = pattern_gen.replace('#',r'\#')
        pattern_gen_max_list = []
        for i,c in enumerate(pattern):
            if pattern_wildcards[i]:
                pattern_gen_max_list.extend([ '#' for i in range(pattern_repetitions[i][1]) ])
            else: pattern_gen_max_list.append(c)
        pattern_gen_max = ''.join(pattern_gen_max_list)

        st.write(f'Gegeben sei der Text $T=\\texttt{{{text}}}$ und das erweiterte Muster $P=\\texttt{{{pattern_gen_tex}}}$.')
        sidebar_top.info(f'$T=\\texttt{{{text}}}$\n\r$P=\\texttt{{{pattern_gen_tex}}}$')

        def t7_gen_a1():
            @st.cache
            def t7_gen_a1_draw_nfa():
                d = gv.Digraph()
                d.node_attr["shape"] = "circle"
                d.graph_attr["rankdir"] = "LR"
                d.node(str(-1), style='filled',fillcolor='#a5a6f9')
                d.edge(str(-1), str(-1), label = chr(120506))
                j = 0
                for i,c in enumerate(pattern):
                    d.node(str(j))
                    if pattern_wildcards[i]:
                        l = 0
                        for k in range(pattern_repetitions[i][1]):
                            d.node(str(j))
                            if l < pattern_repetitions[i][1]-pattern_repetitions[i][0]:
                                d.edge(str(j-1),str(j), label = chr(949), constraint = 'false')
                                l+=1
                            d.edge(str(j-1),str(j), label = chr(120506))
                            j += 1                            
                    else:                        
                        d.edge(str(j-1), str(j), label = c)                        
                        j += 1
                d.node(str(j-1),style = 'filled', fillcolor='#e76960')
                return d
            
            st.subheader('Aufgabe 1')
            st.write('Zeichnen Sie den NFA zu $P$.')
            t7_gen_a1_exp = st.beta_expander('Lösung anzeigen')
            with t7_gen_a1_exp:
                st.write(t7_gen_a1_draw_nfa())    
        def t7_gen_a2_create_masks():
                masks = { c:0 for c in [chr(i) for i in range(sigma_l,sigma_r)]}
                masks['I'], masks['F'] = 0,0                
                j = 0
                for i,c in enumerate(pattern):
                    if pattern_wildcards[i]:
                        masks['I'] |= (1 << j)
                        for k in range(pattern_repetitions[i][1]):                            
                            for a in sigma:
                                masks[a] |= (1 << j)                        
                            j+=1
                        masks['F'] |= (1 << j-pattern_repetitions[i][0])
                    else:
                        masks[c] |= (1 << j)
                        j+=1
                masks['accept'] = (1 << len(pattern_gen_max)-1)
                return masks            
        
        masks = t7_gen_a2_create_masks()

        def t7_gen_a2():            
            st.subheader('Aufgabe 2')
            st.write('Geben Sie die Masken für das erweiterte Muster $P$ an.')            
            uinput = dict()
            labels = dict()
            for k in masks.keys():
                labels[k] = k        
                if len(k) == 1 and 97 <= ord(k) < 122: labels[k] = f'mask[{k}]'
                uinput[k] = st.text_input(labels[k],'',len(pattern_gen_max), key = f't7_a1{k}')
            wrong_inputs = a1_ev(uinput, masks, labels)            
            solution = pd.DataFrame([[ut_int_to_base2_string(s,len(pattern_gen_max))] for s in masks.values()], index = labels, columns = [''])
            ut_check_task(wrong_inputs, solution, key = 't7_gen_a2')

        def t7_gen_a3():
            def t7_gen_shift_and(indices):
                a = 0
                l = []
                for i,c in enumerate(text):
                    a = ((a<<1) | 1) & masks[c]
                    a |= (masks['F'] - (a & masks['I'])) & ~masks['F']
                    if i in indices:
                        l.append(a)
                return l


            st.subheader('Aufgabe 3')
            text_sep = (tlength-2)/3
            indices = [rg.integers(2,text_sep), rg.integers(text_sep, 2*text_sep), rg.integers(2*text_sep, 3*text_sep)]
            st.write(f'Geben Sie für den Shift-And-Algorithmus das Bitmuster nach Lesen der Textzeichen an den Positionen {indices[0]}, {indices[1]} und {indices[2]} an.')
            uinput = []
            for i in indices:
                uinput.append(st.text_input(f'Bitmuster nach Lesen des {i}. Textzeichens', '', len(pattern_gen_max), key = f't7_gen_a3_{i}'))
            solution = t7_gen_shift_and(indices)
            wrong_inputs = t7_a3_ev(uinput, solution)
            solution_df = pd.DataFrame(map(lambda x : ut_int_to_base2_string(x,len(pattern_gen_max)), solution), index = indices, columns = ['Bitmuster'])
            ut_check_task(list(map(lambda x : str(indices[x]),wrong_inputs)), solution_df, opt_string_sgl='das Bitmuster an Position', opt_string_pl='die Bitmuster an den Positionen', key = 't7_gen_a3_check_task')
            
        t7_gen_a1()
        t7_gen_a2()
        t7_gen_a3()

    def t7_2():
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
        plength = rg.integers(7,10)
        pstart = rg.integers(0,len(text) - plength)
        pattern = text[pstart:pstart+plength]
        #create pattern with optional letters
        #pattern_opt_letters =  [0] + list(rg.integers(0,2,plength-2)) + [0]
        number_of_optionals = rg.integers(2,5)
        pattern_opt_indices = rg.choice(range(1,plength-2), number_of_optionals, False)
        pattern_opt_letters = [0 for c in pattern]
        for i in pattern_opt_indices:
            pattern_opt_letters[i] = 1
        pattern_opt_list = []
        for i,c in enumerate(pattern):
            if pattern_opt_letters[i]:                
                pattern_opt_list.append(f'{c}?')
            else: pattern_opt_list.append(c)
        pattern_opt = ''.join(pattern_opt_list)
        st.write(f'Gegeben sei der Text $T=\\texttt{{{text}}}$ und das erweiterte Muster $P=\\texttt{{{pattern_opt}}}$.')
        sidebar_top.info(f'$T=\\texttt{{{text}}}$\n\r$P=\\texttt{{{pattern_opt}}}$')
        
        def t7_opt_a1():
            @st.cache
            def t7_opt_a1_draw_nfa():
                d = gv.Digraph()
                d.node_attr["shape"] = "circle"
                d.graph_attr["rankdir"] = "LR"
                d.node(str(-1), style='filled',fillcolor='#a5a6f9')
                d.edge(str(-1), str(-1), label = chr(120506))
                for i,c in enumerate(pattern):
                    d.node(str(i))
                    d.edge(str(i-1), str(i), label = c)
                    if pattern_opt_letters[i]:
                        d.edge(str(i-1), str(i), label = chr(949))
                d.node(str(plength-1),style = 'filled', fillcolor='#e76960')
                return d
            st.subheader('Aufgabe 1')
            st.write('Zeichnen Sie den NFA zu $P$.')
            t7_opt_a1_exp = st.beta_expander('Lösung anzeigen')
            with t7_opt_a1_exp:
                st.write(t7_opt_a1_draw_nfa())

        masks = t7_opt_create_masks(pattern, pattern_opt_letters)
        def t7_opt_a2():
            st.subheader('Aufgabe 2')
            st.write('Geben Sie die Masken für das erweiterte Muster $P$ an.')            
            a1_input = dict()
            a1_labels = dict()
            for k in masks.keys():
                a1_labels[k] = k        
                if len(k) == 1 and 97 <= ord(k) < 122: a1_labels[k] = f'mask[{k}]'
                a1_input[k] = st.text_input(a1_labels[k],'',plength, key = f't7_a1{k}')
            a1_wrong_inputs = a1_ev(a1_input, masks, a1_labels)
            a1_solution = pd.DataFrame([[ut_int_to_base2_string(s,plength)] for s in masks.values()], index = a1_labels, columns = ['Bitmuster'])
            ut_check_task(a1_wrong_inputs, a1_solution, key = 't7_a1')

        def t7_opt_a3():           
            def t7_opt_shift_and(indices):
                a = 0
                l = []
                for i,c in enumerate(text):
                    a = ((a<<1) | 1) & masks[c]
                    a |= (masks['F'] - (a & masks['I'])) & ~masks['F']
                    if i in indices:
                        l.append(a)
                return l 
            st.subheader('Aufgabe 3')
            text_sep = (tlength-2)/3
            indices = [rg.integers(2,text_sep), rg.integers(text_sep, 2*text_sep), rg.integers(2*text_sep, 3*text_sep)]
            st.write(f'Geben Sie für den Shift-And-Algorithmus das Bitmuster nach Lesen der Textzeichen an den Positionen {indices[0]}, {indices[1]} und {indices[2]} an.')
            uinput = []
            for i in indices:
                uinput.append(st.text_input(f'Bitmuster nach Lesen des {i}. Textzeichens', '', plength, key = f't7_gen_a3_{i}'))
            solution = t7_opt_shift_and(indices)
            wrong_inputs = t7_a3_ev(uinput, solution)
            solution_df = pd.DataFrame(map(lambda x : ut_int_to_base2_string(x,plength), solution), index = indices, columns = ['Bitmuster'])
            ut_check_task(list(map(lambda x : str(indices[x]),wrong_inputs)), solution_df, opt_string_sgl='das Bitmuster an Position', opt_string_pl='die Bitmuster an den Positionen', key = 't7_gen_a3_check_task')

        t7_opt_a1()
        t7_opt_a2()
        t7_opt_a3()

    t7_sel = st.radio('Art des erweiterten Musters', options = [t7_1,t7_2],format_func = t7_rad_format, key = 't7_sel')
    t7_sel()

class ST_node():
    def __init__(self,text, left, right = -1, parent=None):
        self.text = text
        self.children = []
        self.parent = parent        
        self.left = left
        self.right = right        
        self.suffix_link = None
        self.bfs_order = None
        self.str_dep = None
        self.suffix_start_pos = None        

    def find_child(self, c):
        for child in self.children:
            if self.text[child.left] == c:
                return child
        return None
        
    
    def canonize(self,left, offset):
        if offset == -1:
            return self, -1,0
        if left == -1:
            return self, left, offset
        else:
            child = self.find_child(self.text[left])
            left = child.left
            if offset == child.right-child.left-1:
                return child, -1 ,0
            if offset  < child.right  or child.right == -1:
                return self, left, offset
            else:
                return child.canonize(left+offset, offset - (child.right - child.left))
            

    def extend(self,active_left, active_offset,c,c_pos, suffix_link_node = None): 
        if active_left == -1:
            child = self.find_child(c)
            if suffix_link_node is not None:
                suffix_link_node.suffix_link = self
            if child is not None:                 
                return self.canonize(child.left, 0)  
            new_child = ST_node(self.text,c_pos, parent = self)
            self.children.append(new_child)
            if self.parent is None:
                return self,-1,0
            if self.suffix_link is not None:                
                return self.suffix_link.extend(-1,0,c,c_pos)
            else:
                suffix_link_node, suffix_link_left, suffix_link_right = self.parent.suffix_link.canonize(self.parent.left, self.parent.right)
                return suffix_link_node.extend(suffix_link_left, suffix_link_right, c, c_pos)
        else:               
            if self.text[active_left + active_offset+1] == c:
                return self.canonize(active_left, active_offset+1)
            else:
                child = self.find_child(self.text[active_left])
                split_node = ST_node(self.text,active_left,active_left+active_offset+1, parent = self)
                child.parent = split_node
                child.left = active_left+active_offset+1
                self.children.remove(child)
                self.children.append(split_node)
                new_child = ST_node(self.text,c_pos, parent = split_node)
                split_node.children.extend([child,new_child])                         

                if suffix_link_node is not None:
                    suffix_link_node.suffix_link = split_node

                if self.parent is None:
                    split_node.str_dep = split_node.right - split_node.left
                    node, left, offset = self.canonize(active_left+1, active_offset-1)
                    return node.extend(left,offset,c,c_pos, suffix_link_node=split_node)

                else:
                    split_node.str_dep = self.str_dep + (split_node.right - split_node.left)
                    if self.suffix_link is None:                                 
                        if active_offset - 1 == -1:
                            active_left = c_pos - 1
                        if self.find_child(self.text[active_left+1]) is None:
                            return self.extend(active_left+1, active_offset-1, self.text[active_left+1],active_left+1, suffix_link_node= split_node)
                        node, left, offset = self.canonize(active_left+1, active_offset-1)
                    else:
                        node, left, offset = self.suffix_link.canonize(split_node.left,split_node.right-split_node.left-1)
                    return node.extend(left,offset,c,c_pos, suffix_link_node = split_node)

    def bfs(self):
        q = [self]
        while len(q) > 0:
            node = q.pop(0)
            yield node
            q.extend(node.children)

    def to_string(self):
        node = self    
        substrings = []       
        while node.parent is not None:
            substrings.insert(0, self.text[node.left:node.right])
            node = node.parent
        return ''.join(substrings)


def st_build(text):
    root = ST_node(text,0)
    root.str_dep = 0
    active , active_left, active_offset = root,-1,0
    for i,c in enumerate(text):                        
        active, active_left, active_offset = active.extend(active_left,active_offset, c,i)
    for i,node in enumerate(root.bfs()):
        node.bfs_order=i
        node.children = sorted(node.children, key = lambda x: text[x.left])
    return root, active.bfs_order, active_left, active_offset

def st_final(root):
    tlength = len(root.text)
    for node in root.bfs():
        if not node.children:
            node.right = tlength
            node.suffix_start_pos = node.right-(node.right - node.left) - node.parent.str_dep
    return root

def st_to_gv(root, active_node = None, string_labels = False):
    d = gv.Digraph()
    d.node_attr["shape"] = "circle"
    d.node_attr["width"] = "0.3"
    for node in root.bfs():
        if node.parent is None:            
            d.node(str(node.bfs_order), label = '')
        else:
            d.node(str(node.bfs_order), label = '')
            if string_labels:
                d.edge(str(node.parent.bfs_order), str(node.bfs_order), label=root.text[node.left:node.right])
            else:
                d.edge(str(node.parent.bfs_order), str(node.bfs_order), label=f'({node.left}:{node.right})')
            if node.suffix_link is not None and node.suffix_link.parent is not None:
                d.edge(str(node.bfs_order), str(node.suffix_link.bfs_order), style = 'dashed', constraint = 'false', color = 'darkgreen')
        if not node.children and node.suffix_start_pos is not None:
            d.node(str(node.bfs_order), label = str(node.suffix_start_pos), width = '0.6')
    if active_node is not None:
        d.node(str(active_node), color = 'red')
    return d

def t8(seed, sidebar_top):
    SIGMA_L = 97
    SIGMA_R = 100
    sigma = [ chr(i) for i in range(SIGMA_L,SIGMA_R) ]    
    #random text, pattern:
    text, text_tex, tlength, pattern, pattern_tex, plength = ut_rnd_text_and_pattern(seed,SIGMA_L,SIGMA_R)
    text = f'{text}$'

    st.write(f'Gegeben Sei der Text {text_tex}$.')
    
    #st.write(st_to_gv(st_build('babacacb$')))

    #A1
    st.subheader('Aufgabe 1')
    st.write('Zeichnen Sie den Suffixbaum zu $T$.')
    st.write('**Lösung:**')
    iteration = st.slider('Iteration', -1, tlength)
    if iteration == -1:
        st.write('Iteration wählen')
    #elif iteration == tlength:
    #    st.write(st_to_gv(st_final(st_build(text))))
    else:
        root, active_node_bfs_order, active_left, active_offset = st_build(text[:iteration+1])
        if iteration == tlength:
            root = st_final(root)
        edge_strings = st.checkbox('Strings anzeigen')
        if edge_strings:
            st.write(st_to_gv(root,active_node = active_node_bfs_order, string_labels = True))
        else:
            st.write(st_to_gv(root,active_node = active_node_bfs_order))
        st.write(f'Aktive Position: {active_left} + {active_offset}')
        st.markdown('*Suffixlinks die auf die Wurzel zeigen sind nicht eingezeichnet.*')


    stree = st_final(st_build(text)[0])
    #A2
    st.subheader('Aufgabe 2')
    st.write('Geben Sie den/einen längsten wiederholten Teilstring von $T$ an.')
    
    a2_input = st.text_input('LRS', key = 't8_a2_input')

    max_str_dep = 0
    max_str_dep_node_list = [stree]
    for node in stree.bfs():
        if node.str_dep is not None:
            if node.str_dep > max_str_dep:
                max_str_dep = node.str_dep
                max_str_dep_node_list = [node]
            elif node.str_dep == max_str_dep:             
                max_str_dep_node_list.append(node)
    
    lrs_list = []  
    for node in max_str_dep_node_list:        
        lrs_substrings = []          
        while node.parent is not None:
            lrs_substrings.insert(0, text[node.left:node.right])
            node = node.parent
        lrs_list.append(''.join(lrs_substrings))
    a2_wrong_inputs = []
    if a2_input not in lrs_list:
        a2_wrong_inputs.append('den längsten wiederholten Teilstring')
    a2_solution_df = pd.DataFrame(lrs_list,index = [ '' for s in lrs_list ], columns = ['LRS']) 
    ut_check_task(a2_wrong_inputs,a2_solution_df,key = 't8_a2_check')

    #A3
    st.subheader('Aufgabe 3')
    st.write('Geben Sie den/einen kürzesten eindeutigen Teilstring von $T$ an.')

    sus_node_list = []
    sus_node_str_dep_min = tlength
    for node in stree.bfs():
        if not node.children:            
            if text[node.left] != '$':                
                if not sus_node_list or node.parent.str_dep == sus_node_str_dep_min:                    
                    sus_node_list.append(node)
                    sus_node_str_dep_min = node.parent.str_dep
                elif node.parent.str_dep < sus_node_str_dep_min:
                    sus_node_list = [node]
                    sus_node_str_dep_min = node.parent.str_dep
    
    sus_list = [] 
    for node in sus_node_list:
        sus_substrings = []
        sus_substrings.append(text[node.left])
        if node.parent is not None:
            node = node.parent
        while node.parent is not None:
            sus_substrings.insert(0, text[node.left:node.right])
            node = node.parent
        sus_list.append(''.join(sus_substrings))
    
    a3_input = st.text_input('SUS','',key = 't8_a3_input')
    a3_wrong_inputs = []
    if a3_input not in sus_list:
        a3_wrong_inputs.append('den kürzesten eindeutigen Teilstring')
    a3_solution_df = pd.DataFrame(sus_list,index = [ '' for s in sus_list ], columns = ['SUS'])    
    ut_check_task(a3_wrong_inputs,a3_solution_df, key = 't8_a3_check')

    #A4
    st.subheader('Aufgabe 4')
    rg1 = np.random.default_rng(seed)
    t1_chrs = rg1.integers(SIGMA_L,SIGMA_R, rg1.integers(6,9))
    t1 = ''.join([ chr(a) for a in t1_chrs] + [chr(35)])    
    t1_tex = ''.join([ chr(a) for a in t1_chrs] + ['\\#'])    
    rg2 = np.random.default_rng(seed*2)
    t2_chrs = rg2.integers(SIGMA_L,SIGMA_R, rg2.integers(6,9))    
    t2 = ''.join([ chr(a) for a in t2_chrs] + [chr(36)])
    t2_tex = ''.join([ chr(a) for a in t2_chrs] + ['\\$'])
    st.write(f'Sei $T_1=\\texttt{{{t1_tex}}}$  und $T_2=\\texttt{{{t2_tex}}}$.')
    st.write('Geben Sie den/einen längsten gemeinsamen Teilstring von $T_1$ und $T_2$ an.')    
    a4_stree = st_final(st_build(t1+t2)[0])
   
    a4_input = st.text_input('LCS', '', key = 't8_a4_input')

    a4_show_stree_exp = st.beta_expander('Suffixbaum anzeigen')
    with a4_show_stree_exp:        
       st.write(f'Suffixbaum für den Text $\\texttt{{{t1_tex}{t2_tex}}}$:')
       st.write(st_to_gv(a4_stree))

    def a4_choose_candidates(node,t1length):
        if not node.children:
            if node.suffix_start_pos in range(t1length):
                return 1,[]
            else:
                return 2,[]
        else:
            children_ret = list(map(lambda x : a4_choose_candidates(x,t1length), node.children))
            children_labels = set(map(lambda x : x[0], children_ret))
            children_node_lists = map(lambda x : x[1], children_ret)
            if len(children_labels) == 2:                
                return 3, sum(children_node_lists,[node])
            else:
                return children_labels.pop(), sum(children_node_lists,[])

    node_candidates = a4_choose_candidates(a4_stree, len(t1))[1]
    lcs_max_length = 0
    lcs_nodes = []
    for node in node_candidates:
        if node.str_dep == lcs_max_length:
            lcs_nodes.append(node)
        elif node.str_dep > lcs_max_length:
            lcs_max_length = node.str_dep
            lcs_nodes = [node]
    lcs_list = [ x.to_string() for x in lcs_nodes ]
    a4_wrong_inputs = []
    if a4_input not in lcs_list:
        a4_wrong_inputs.append('den längsten gemeinsamen Teilstring')
    a4_solution_df = pd.DataFrame(lcs_list, columns = ['LCS'],index = [ '' for s in lcs_list ])
    ut_check_task(a4_wrong_inputs, a4_solution_df, key = 't8_a4_check')




pages = {
    '00': home,
    '01': t1,
    '02': t2,
    '03': t3,
    '04': t4,
    '05': t5,
    '06': t6,
    '07': t7,
    '08': t8,
   # '09': t9
}

pages_titles = {
    '00': 'Home',
    '01': '01: Einführung',
    '02': '02: Naiver Algorithmus für die exakte Mustersuche',
    '03': '03: Exakte Mustersuche mit nichtdeterministischen endlichen Automaten',
    '04': '04: Exakte Mustersuche mit deterministischen endlichen Automaten',
    '05': '05: Exakte Mustersuche mit sublinearen Algorithmen',
    '06': '06: Exakte Mustersuche auf Mengen von Mustern',
    '07': '07: Exakte Mustersuche mit erweiterten Mustern',
    '08': '08: Suffixbäume',
    #'09': '09: Suffixarrays'
}

def radio_format(str):
    return pages_titles[str]    

st.header('Algorithmen auf Sequenzen')
sidebar_top = st.sidebar.empty()
st.sidebar.title('Teil')
selection = st.sidebar.radio("Gehe zu Teil",list(pages.keys()), format_func=radio_format)
seed = st.sidebar.number_input('Seed',0,None,42)
page = pages[selection]
st.title(pages_titles[selection])
page(seed, sidebar_top)