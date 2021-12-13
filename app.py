"""
@author: nyamwamu
"""

import streamlit as st
import streamlit_analytics
import pickle
import tensorflow as tf
import numpy as np
import prep
import pandas as pd
import matplotlib.pyplot as plt

EMOT = ['anger', 'disgust', 'fear', 'joy',
        'sadness', 'surprise', 'neutral']
model = None


def main():
    # Setting the header of the page
    st.title("AI For Mental Health Analysis")

    menu = ['Home', 'Mentox Analysis', 'About']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.subheader('Home')
        st.write('A simple POC sentiment analysis implementing deep learning')
        st.write('Get to know the emotions for mental state evalution')

    elif choice == 'Mentox Analysis':
        st.subheader('Mentox Analysis')
        model = tf.keras.models.load_model('./model/smodel.h5')
        txt = take_inputs()

        if st.button('Analyse'):
            result = run_sentiment_analysis(txt, model)
            st.pyplot(plot_emotions(result))

    elif choice == 'About':
        st.subheader('About')
        st.write('This is simple POC application that focuses on analysing natural language response given by a potential patient by implementing deep learning sentiment analysis')
        st.write(
            'Application is developed by Bebeto Nyamwamu, Samuel Mandillah, and John Lotome to analyse emotion from natural language')
        st.warning(
            'Warning! `Nuff said: To talk to the team through :email: [email](mailto:nberbetto@gmail.com) or view the work on [Github](https://github.com/realonbebeto/https://github.com/realonbebeto/Mentox-App)')


def take_inputs():
    txt = st.text_area(
        'Text', '''The quick brown fox jumps over the lazy dog''')

    if not txt:
        st.error('Empty input! Enter a text and try again')

    return txt


def tagging(preds):
    preds = np.round(preds, 0) * 100
    result = pd.DataFrame(
        {'Emotion': EMOT, 'Score': preds.tolist()[0]})

    return result


def run_sentiment_analysis(txt: str, model):
    input_txt = prep.text_preprocessing_pipeline(txt)

    with st.spinner('Analysing...'):
        preds = model.predict([input_txt])
        results = tagging(preds)

    st.success('Done')

    return results


def plot_emotions(rslt: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Horizontal Bar Plot
    ax.barh(rslt['Emotion'], rslt['Score'], height=0.5)

    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    # Remove x, y Ticks
    ax.yaxis.set_ticks_position('none')

    ax.xaxis.set_visible(False)

    # Show top values
    ax.invert_yaxis()

    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.3,
                 str(round((i.get_width()), 0)),
                 fontsize=10, fontweight='bold',
                 color='grey')

    return fig


if __name__ == '__main__':
    with streamlit_analytics.track(unsafe_password='1'):
        # Setting the title bar page name
        st.set_page_config(page_title='Mentox Analysis', page_icon='😉',
                           layout='centered', initial_sidebar_state='auto')
        try:
            main()
        except:
            st.error(
                'Oops! Something went wrong...Please check your input.\nIf you think there is a bug, please open up an [issue](https://github.com/realonbebeto/Startup-App/issues) and help us improve. ')
            raise
