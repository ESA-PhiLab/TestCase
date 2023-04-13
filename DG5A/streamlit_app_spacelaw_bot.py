import streamlit as st
from streamlit_chat import message
import pandas as pd
import numpy as np
import requests
import query_api_dg5a
import streamlit_toggle as tog

def main():


    st.set_page_config(
        page_title="Space Law  QA Bot - Demo",
        page_icon=":robot:"
    )



    sl_bot = query_api_dg5a.spacelaw_bot()


    # eo = tog.st_toggle_switch(label="EO Portal Articles",
    #                              key="Key1",
    #                              default_value=False,
    #                              label_after=True,
    #                              inactive_color='#D3D3D3',
    #                              active_color="#11567f",
    #                              track_color="#29B5E8"
    #                              )

    # API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
    # headers = {"Authorization": st.secrets['api_key']}

    st.header("Space Law  QA Bot - Demo")
    st.markdown("[Github](https://github.com/LuytsA/NLP4EO)")

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if not st.session_state['generated']:
        st.session_state.generated.append('Hello, I am Space Law Bot. Please ask me any Earth Observation related questions '
                                          'based on my information source.\n'
                                          f'Information source: Patrizia\'s PDFs')
        st.session_state.past.append('Hi!')


    def query(payload):
        top_sources = sl_bot.sematic_search(payload)
        response = sl_bot.summarise(payload, top_sources)
        return response

    def get_text():
        input_text = st.text_input("You: ", '', key="input")
        return input_text

    user_input = get_text()

    if user_input:
        response = query(user_input)
        output = {"generated_text": response['response']}

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output["generated_text"])

    if st.session_state['generated']:

        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


if __name__ == '__main__':
    main()