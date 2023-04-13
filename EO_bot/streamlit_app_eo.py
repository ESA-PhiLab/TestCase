import streamlit as st
from streamlit_chat import message
import pandas as pd
import numpy as np
import requests
import query_api_eo
import  streamlit_toggle as tog

def main():

    st.set_page_config(
        page_title="EO Chat - Demo",
        page_icon=":robot:"
    )

    context = tog.st_toggle_switch(label="Show context",
                             key="Key1",
                             default_value=True,
                             label_after=False,
                             inactive_color='#D3D3D3',
                             active_color="#11567f",
                             track_color="#29B5E8"
                             )

    EO_bot = query_api_eo.eo_bot()
    st.header("EO Question & Answer Bot - Demo")
    st.markdown("[Github](https://github.com/LuytsA/NLP4EO)")

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if 'context' not in st.session_state:
        st.session_state['context'] = []

    if not st.session_state['generated']:
        st.session_state.generated.append('Hello, I am EO Bot. Please ask me any Earth Observation related questions '
                                          'based on my information sources.\n'
                                          f'Information sources: Arxiv Papers, EoPortal Articles, Medium Articles')
        st.session_state.past.append('Hi!')
        st.session_state.context.append('Context prompts will be shown by default. Press toggle to hide them!')


    def query(payload):
        top_sources = EO_bot.sematic_search(payload)
        response = EO_bot.summarise(payload, top_sources)
        return response

    def get_text():
        input_text = st.text_input("You: ", '', key="input")
        return input_text

    user_input = get_text()

    if user_input:
        response = query(user_input)
        output = {"generated_text": response['response'], 'context': response['context']}


        st.session_state.past.append(user_input)
        st.session_state.generated.append(output["generated_text"])
        st.session_state.context.append(output["context"])

    if st.session_state['generated']:

        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            if context:
                message(st.session_state["context"][i], key=str(i) + '_context')
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


if __name__ == '__main__':
    main()