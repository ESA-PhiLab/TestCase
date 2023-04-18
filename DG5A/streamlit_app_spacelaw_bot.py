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


    context = tog.st_toggle_switch(label="Show context",
                             key="Key1",
                             default_value=True,
                             label_after=False,
                             inactive_color='#D3D3D3',
                             active_color="#11567f",
                             track_color="#29B5E8"
                             )

    sl_bot = query_api_dg5a.spacelaw_bot()
    st.header("Space Law  QA Bot - Demo")
    st.markdown("[Github](https://github.com/LuytsA/NLP4EO)")
    st.markdown("ATTENTION: queries and ratings will be logged for development purposes!")

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if 'context' not in st.session_state:
        st.session_state['context'] = []

    if 'submitted' not in st.session_state:
        st.session_state['submitted'] = []

    if 'slider_value' not in st.session_state:
        st.session_state['slider_value'] = []

    if not st.session_state['generated']:
        st.session_state.generated.append('Hello, I am Space Law Bot. Please ask me any Earth Observation related questions '
                                          'based on my information source.\n'
                                          f'Information source: Patrizia\'s PDFs')
        st.session_state.past.append('Hi!')
        st.session_state.context.append('Context prompts will be shown by default. Press toggle to hide them!')

    @st.cache_data
    def query(payload):
        top_sources = sl_bot.sematic_search(payload)
        response = sl_bot.summarise(payload, top_sources)
        return response

    def get_text():
        input_text = st.text_input("You: ", '', key="input")
        return input_text
    
    def log_rating():
        with open('log_queries.txt', 'a') as f:
            rating = str(st.session_state.rating_slider)
            message = f'RATING: {rating}\n'
            f.write(message)

    
    @st.cache_data
    def log_results(query, answer, context=''):
        with open('log_queries.txt', 'a') as f:
            f.write('***'*10 +'\n')
            message = f'QUERY: {str(query)} \nANSWER: \n{str(answer)} \n\nCONTEXT: \n{str(context)} \n'
            f.write(message)

    def instantiate_slider(default_val=1):
        c1,c2 = st.columns([0.5,2])
        with c2:
            val = st.session_state['slider_value']
            st.session_state['slider_value'].append(st.slider(':star: Rate most recent answer :star:',1,5,value=default_val, key='rating_slider'))
        with c1:
            st.session_state['submitted'] = st.button('Submit rating', key='submit_button', on_click = log_rating)
    
    user_input = get_text()
    instantiate_slider()

    if user_input and not st.session_state['submitted'] and (st.session_state['slider_value'][-1] == st.session_state['slider_value'][-2]):

        response = query(user_input)
        output = {"generated_text": response['response'], 'context': response['context']}
        log_results(user_input, output['generated_text'], output["context"])

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