import streamlit as st
import numpy as np
import graphviz as gv
import re
import pandas as pd
import t3

def app(seed, sidebar_top):
    #parameters
    sigma_l = 97
    sigma_r = 100
    sigma = [ chr(i) for i in range(sigma_l,sigma_r) ]    
    #generate random text
    rg = np.random.default_rng(seed)
    tlength = int(rg.integers(16,21))
    t = rg.integers(sigma_l,sigma_r,tlength)
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
    for a in sigma_t:
        a1_input.append(st.text_input(f'shift[{a}]','',key = f't5_a1{a}'))
    if st.button('Aufgabe überprüfen', key = 't5_a1_ev_btn'):
        a1_ev_result = a1_ev(sigma_p,a1_create_shifts(pattern, sigma),a1_input)
        if not a1_ev_result:
            fba1 = '>**Lösung korrekt.**'
        elif len(a1_ev_result) == 1:
            fba1 = f'>**Fehler in der Eingabe für** shift[{str(sigma_t[(a1_ev_result[0])])}]**.**'
        else:
            fba1 = '>**Fehler in den Eingaben für **' + ', '.join([f'shift[{str(sigma_t[i])}]' for i in a1_ev_result[:-1]]) + f'** und **shift[{sigma_t[a1_ev_result[-1]]}]**.**' 
        st.markdown(fba1)

    #A2
    st.subheader('Aufgabe 2')
    st.markdown('Führen Sie den Horspool-Algorithmus mit dem Muster $P$ auf dem Text $T$ aus. Geben Sie die Längen der Sprünge an, sowie die Textpositionen, an denen das letzte Zeichen des Musters nach Ausführung der Sprünge steht (im Algorithmus aus der Vorlesung sind dies die Werte für die Variable `last`).')
    
    #too slow atm
    #a2_cols_ind = st.beta_columns(tlength+2)
    #a2_cols_text = st.beta_columns(tlength+2)
    #a2_cols_pattern = st.beta_columns(tlength+2)
    #for i in range(tlength):
    #        a2_cols_ind[i+2].write(str(i))
    #a2_cols_text[0].write('$T$')
    #a2_cols_text[1].write('$=$')
    #for i,c in enumerate(text):
    #        a2_cols_text[i+2].write(fr'$\mathtt {c}$')
    #a2_cols_pattern[0].write('$P$')
    #a2_cols_pattern[1].write('$=$')
    
    a2_vis = st.beta_container()

    shifts_e = a2_horspool(text, pattern, sigma, tlength, plength)
    lasts = []
    last_last = plength-1
    for s in shifts_e:
        lasts.append(last_last+s)        

    a2_input_shifts = []
    a2_input_lasts = []
    for k,i in enumerate(shifts_e):
        cols = st.beta_columns([8,1,8])
        a2_input_shifts.append(cols[0].text_input('Iteration '+str(k+1)+': Sprunglänge','',key='t5_a2_input_shifts'+str(k)))
        a2_input_lasts.append(cols[2].text_input('Iteration '+str(k+1)+': last','',key='t5_a2_input_lasts'+str(k)))

    a2_last_filled_index = plength-1
    for k in range(len(shifts_e)):
        try:
            if int(a2_input_shifts[k]) >= 0 and int(a2_input_lasts[k]) >= 0:
                a2_last_filled_index = int(a2_input_lasts[k])
            else: break
        except ValueError:
            break

    #too slow atm
    #for i,c in zip(range(a2_last_filled_index-plength+3, a2_last_filled_index+3),pattern):
    #    if i > tlength + 1: break
    #    a2_cols_pattern[i].write(fr'$\mathtt {c}$')

    a2_vis_data_pattern = [ '' for c in text]
    for i,c in zip(range(a2_last_filled_index-plength+1, a2_last_filled_index+1),pattern):
        if i > tlength - 1 : break
        a2_vis_data_pattern[i] = c

    a2_vis_data = [list(text), a2_vis_data_pattern]
    a2_vis_df = pd.DataFrame(a2_vis_data,index = ['T','P'], columns = range(tlength) )
    with a2_vis:
        st.table(a2_vis_df)

    if st.button('Aufgabe überprüfen', key = 't5_a2_ev_btn'):
        a2_shifts_ev_result,a2_lasts_ev_result = a2_ev(a2_input_shifts,a2_input_lasts,shifts_e,plength)
        a2_shifts_ev_result_len, a2_lasts_ev_result_len = len(a2_shifts_ev_result), len(a2_lasts_ev_result)
        if not a2_shifts_ev_result and not a2_lasts_ev_result:
            fba2 = '>**Lösung korrekt.**'
        elif a2_shifts_ev_result_len == 1 and a2_lasts_ev_result_len == 0:
            fba2 = '>**Fehler in der Eingabe für** Iteration '+str(a2_shifts_ev_result[0])+': Sprunglänge**.**'
        elif a2_shifts_ev_result_len == 0 and a2_lasts_ev_result_len == 1:
            fba2 = '>**Fehler in der Eingabe für** Iteration '+str(a2_shifts_ev_result[0])+': last**.**' 
        else:
            a2_shifts_out = []
            a2_lasts_out = []
            for k in a2_shifts_ev_result:
                a2_shifts_out.append('>- Iteration '+str(k)+': Sprunglänge\n\r')
            for k in a2_lasts_ev_result:
                a2_shifts_out.append('>- Iteration '+str(k)+': last\n\r')
            a2_shifts_out_str = ''.join(a2_shifts_out)
            a2_lasts_out_str = ''.join(a2_lasts_out)
            fba2 = '>**Fehler in den Eingaben für:**\n\r '+ ''.join(a2_shifts_out) + ''.join(a2_lasts_out)
        st.markdown(fba2)

    #a2_input = st.text_input('Folge der Sprunglängen')
    #if st.button('Aufgabe überprüfen', key = 't5_a2_ev_btn'):
    #    a2_ev_result = a2_ev(a2_input, a2_horspool(text,pattern, sigma, tlength, plength))
    #    if not a2_ev_result:
    #        fba2 = '>**Lösung korrekt.**'
    #    else:
    #        fba2 = '>**Fehler in der Eingabe.**' 
    #    st.markdown(fba2)

    #A3
    st.subheader('Aufgabe 3')
    st.write('Zeichnen Sie den Suffixautomaten, der im BNDM-Algorithmus zur Mustersuche mit dem Muster $P$ auf dem Text $T$ simuliert wird.')
    a3_exp = st.beta_expander('Lösung anzeigen')
    with a3_exp:
        st.write(a3_draw_suffix_automaton(pattern))
    

    #A4
    st.subheader('Aufgabe 4')
    st.write('Führen Sie den BNDM-Algorithmus mit dem Muster $P$ auf dem Text $T$ aus. Geben Sie die Länge der Sprünge an, sowie die Textpositionen, an denen das letzte Zeichen des Musters nach Ausführung der Sprünge steht.')
    a3_input = st.text_input('')

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

#def a2_ev(input,shifts_e):
#    wrong_inputs = []
#    input_list = re.split('\s*,\s*|\s*;\s*|\s+',input)
#    for k,i in enumerate(input_list):
#        try:
#            if not int(i) == shifts_e[k]:
#                wrong_inputs.append(i)
#        except ValueError:
#                wrong_inputs.append(i)
#    return wrong_inputs

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


def a4_bdnm(text, pattern, tlength, plength, masks):
    shifts = []
    full = (1 << plength) - 1
    accept = 1 << (plength-1)    
    last = m
    masks = t3.a2_create_masks(pattern[:-1])
    while last <= n:
        A = full
        lastsuffix = 0
        j = 1
        while A:
            A &= masks[T[last-j]]
            if A & accept:
                if j == m:
                    break
                else:
                    lastsuffix = j
            j += 1
            A <<= 1
        shift = m-lastsuffix
        last += shift
        shifts.append(shift)
