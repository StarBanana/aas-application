import home, t1, t2, t3, t4, t5, t6, t7
import streamlit as st
import numpy as np

pages = {
    '00': home,
    '01': t1,
    '02': t2,
    '03': t3,
    '04': t4,
    '05': t5,
    '06': t6,
    '07': t7
}

pages_titles = {
    '00': 'Home',
    '01': '01: Einführung',
    '02': '02: Naiver Algorithmus für die exakte Mustersuche',
    '03': '03: Exakte Mustersuche mit nichtdeterministischen endlichen Automaten',
    '04': '04: Exakte Mustersuche mit deterministischen endlichen Automaten',
    '05': '05: Exakte Mustersuche mit sublinearen Algorithmen',
    '06': '06: Exakte Mustersuche auf Mengen von Mustern',
    '07': '07: Exakte Mustersuche mit erweiterten Mustern'
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
page.app(seed, sidebar_top)

