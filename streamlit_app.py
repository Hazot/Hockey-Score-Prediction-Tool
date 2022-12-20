import streamlit as st
import pandas as pd
import numpy as np
import os

from ift6758.client.serving_client import ServingClient
from ift6758.client.game_client import GameClient

# """
# General template for your streamlit app. 
# Feel free to experiment with layout and adding functionality!
# Just make sure that the required functionality is included as well
# """


################### (FALSE) NEEDED DATA

features = ['Feature1', 'Feature2','Feature3','Feature4','Feature5',
            'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10',
            'Feature11', 'Feature12', 'Feature13', 'Feature14', 'Feature15',
            'Feature16', 'Feature17', 'Feature18']

feature_values = np.random.rand(5, 18) 
model_pred = np.random.rand(5,1)

df = pd.DataFrame(feature_values, columns=features)
df['Model Output'] = model_pred


Teams = ['Team1', 'Team2'] #### How to get info?
real_goals = [2, 3] #### How to get info?

#################### SESSION STATE

if 'stored_game_id' not in st.session_state:
    st.session_state['stored_game_id'] = 0


# ------------ Real values

if 'gameClient' not in st.session_state:
    gameClient = GameClient()
    st.session_state['gameClient'] = gameClient

if 'servingClient' not in st.session_state:
    servingClient = ServingClient()
    st.session_state['servingClient'] = servingClient

if 'model_downloaded' not in st.session_state:
     st.session_state['model_downloaded'] = False

if 'model' not in st.session_state:
    st.session_state['model'] = None

if 'model_selection_change' not in st.session_state:
    st.session_state['model_selection_change'] = False

if 'stored_df' not in st.session_state: 
    st.session_state.stored_df = None

#################### FUNCTION DEFINITIONS

def calculate_game_goals(df: pd.DataFrame):
    """
    Sum over model_pred for each team 
        Input: df (DataFrame), with feature values and model prediction for every event
        Output: pred_goals (list), predicted number of goals for each team    
    """
    #### How to sum over a certain team? -> is there a certain 0/1 binary feature?
    pred_goals = [1.8, 3.4]
    return pred_goals



IP = os.environ.get("SERVING_IP", "0.0.0.0")
PORT = os.environ.get("SERVING_PORT", "5000")
base_url = f"http://{IP}:{PORT}"


st.title("Hockey Visualization App")

st.write("Got base URL:", base_url)


with st.sidebar:
    # TODO: Add input for the sidebar

    workspace = st.selectbox(label='Workspace', options=['ift-6758-projet-quipe-13'] )
    model = st.selectbox(label='Model', options=['xgboost-best-all-features', 'xgboost-best-select-features', 'xgboost-base-all-features'])
    version = st.selectbox(label='Model version', options=['1.0.0']) 

    model_button = st.button('Get Model')

    if model_button and model == st.session_state.model:
        st.write(f'Model {model} already downloaded!')
    
    elif model_button: # and model != st.session_state.model:
        st.session_state['model_downloaded'] = True 
        st.session_state['model'] = model
        st.write(st.session_state.servingClient)
        st.write(st.session_state.model)

        st.session_state.servingClient.download_registry_model(workspace, st.session_state.model, version)
        st.write(f'Downloaded model {st.session_state.model}!')

        st.session_state.stored_df = None # Reinitializing stored dataframe

    elif not model_button and st.session_state.model_downloaded:
        st.write(f'Downloaded model {st.session_state.model}!')
    
    else:
        st.write('Waiting on **Get Model** button press...')

    


with st.container():
    # TODO: Add Game ID input
    game_id = st.text_input(label='Input Game ID:', value='2021020329', max_chars=10, label_visibility='collapsed')

    pred_button = st.button('Ping Game')
    if pred_button:
        if st.session_state.model_downloaded == False: 
            st.write('Please download model first!')
        else: 
            st.write(f'**The current game ID is {game_id}!**') ### Add info on Teams, date of game


           
    
with st.container():
    # TODO: Add Game info and predictions
    st.write(st.session_state.stored_df)
    st.write(st.session_state.servingClient.features)
    st.subheader(f"Game goal predictions")
    if pred_button and st.session_state.model_downloaded:
        st.write(st.session_state.model)
        #### With Flask/clients:
        df_MODEL = st.session_state.gameClient.process_query(game_id, model_name=st.session_state.model) 
        # st.write(df_MODEL.shape)

        if df_MODEL is not None:
            pred_MODEL = st.session_state.servingClient.predict(df_MODEL)
            df = pd.DataFrame(df_MODEL, columns=st.session_state.servingClient.features)   # Features list arent updated
            df['Model Output'] = pred_MODEL
        else: 
            df = None
        
        if game_id == st.session_state.gameClient.gameId: # Comparing current and previous gameId
            df = pd.concat([st.session_state.stored_df, df], ignore_index=True)
            st.session_state.stored_df = df 
        else: 
            st.session_state.stored_df = df 
        

        pred_goals_MODEL = calculate_game_goals(df)

        col1, col2 = st.columns(2)
        delta1 = np.round(pred_goals_MODEL[0] - real_goals[0], decimals=1)
        delta2 = np.round(pred_goals_MODEL[1] - real_goals[1], decimals=1)
        col1.metric(label=f"{Teams[0]} xG (actual)", value=f"{pred_goals_MODEL[0]} ({real_goals[0]})", delta=delta1)
        col2.metric(label=f"{Teams[1]} xG (actual)", value=f"{pred_goals_MODEL[1]} ({real_goals[1]})", delta=delta2)


    else:
        st.write('Waiting on **Ping Game** button press...')
    

    st.subheader(f"Goal prediction per game event")
    if pred_button and st.session_state.model_downloaded:
        st.write(st.session_state.stored_df)
    else:
        st.write('Waiting on **Ping Game** button press...')


# with st.container():
#     # TODO: Add data used for predictions
#     st.subheader(f"Goal prediction per game event")
#     st.write(df_MODEL)
#     pass



####### TO DO: 
# - Append predictions to dataframe 💚
# - Save previous predictions 💚
    # What if previous game_id is one of an ended game 💚
    # What if model change: have to reset stored_df for the same game 💚

# - Game goal predictions (separate the 2 teams)

# - Afficher Period, Time left to Period
    # Ne pas afficher if game_ended


# Ping game when model isnt downloaded -> Error 💚

