import streamlit as st
from streamlit_chat import message
import pandas as pd
import numpy as np
import requests
import query_api

def main():
    EO_bot = query_api.EO_bot()
    st.set_page_config(
        page_title="EO Chat - Demo",
        page_icon=":robot:"
    )

    # API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
    # headers = {"Authorization": st.secrets['api_key']}

    st.header("EO Chat - Demo")
    st.markdown("[Github](https://github.com/ai-yash/st-chat)")

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

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
        output = {"generated_text": response['response']}

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output["generated_text"])

    if st.session_state['generated']:

        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


if __name__ == '__main__':
    main()