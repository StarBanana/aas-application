import streamlit as st

def app(seed, sidebar_top):
    st.write('Willkommen zu Algorithmen auf Sequenzen. ')

    st.subheader('Hinweise')
    st.markdown("""
        - Bei Eingaben von Folgen oder Mengen können die Elemente durch die Zeichen ' ', ',' oder ';' separiert werden.
        - Bei Eingaben von Mengen sind die Mengenklammern '{', '}' optional.
        - Strings werden zufällig mithilfe eines Seeds generiert.
    """)