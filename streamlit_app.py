"""
If you are in the same directory as this file (streamlit_app.py), you can run run the app using gunicorn:
    
    $ streamlit run streamlit_app.py --server.port=8892 --server.address=0.0.0.0

"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import time

from ift6758.client.serving_client import ServingClient
from ift6758.client.game_client import GameClient

# """
# General template for your streamlit app. 
# Feel free to experiment with layout and adding functionality!
# Just make sure that the required functionality is included as well
# """

IP = os.environ.get("SERVING_IP", "0.0.0.0")
PORT = os.environ.get("SERVING_PORT", 5000)
base_url = f"http://{IP}:{PORT}"

#################### STREAMLIT SESSION STATE OBJECTS


if 'gameClient' not in st.session_state:
    gameClient = GameClient()
    st.session_state['gameClient'] = gameClient

if 'servingClient' not in st.session_state:
    servingClient = ServingClient(ip=IP, port=PORT)
    st.session_state['servingClient'] = servingClient

if 'model_downloaded' not in st.session_state:
     st.session_state['model_downloaded'] = False

if 'model' not in st.session_state:
    st.session_state['model'] = None

if 'model_selection_change' not in st.session_state:
    st.session_state['model_selection_change'] = False

if 'stored_df' not in st.session_state: 
    st.session_state.stored_df = None

if 'pred_goals' not in st.session_state:
    st.session_state.pred_goals = [0,0]
    
if 'real_goals' not in st.session_state:
    st.session_state.real_goals = [0,0]

if 'teams' not in st.session_state:
    st.session_state.teams = None


#################### FUNCTION DEFINITION

def calculate_game_goals(df: pd.DataFrame, pred: pd.DataFrame, teams_A_H: list):
    """
    Sum over model_pred for each team 
        Input: 
            df (DataFrame), with feature values 
            pred (DataFrame), model prediction for every event
            teams_A_H (list), teams tricode
        Output: 
            pred_goals (list), predicted number of goals for each team
            teams (list), abbreviation for each team

    """
    df = df.reset_index(drop=True)

    df['Model Output'] = pred

    # teams = df['team'].unique()

    pred_team1 = df.loc[df['team']==teams_A_H[0] , 'Model Output']
    sum_pred_team1 = pred_team1.sum()
    pred_team2 = df.loc[df['team']==teams_A_H[1] , 'Model Output']
    sum_pred_team2 = pred_team2.sum()

    # real_team1 = df.loc[df['team']==teams[0] , 'isGoal']
    # sum_real_team1 =real_team1.sum()
    # real_team2 = df.loc[df['team']==teams[1] , 'isGoal']
    # sum_real_team2 = real_team2.sum()
 
    pred_goals = [sum_pred_team1, sum_pred_team2]
    # real_goals = [sum_real_team1, sum_real_team2]

    return pred_goals


# def map_teams(teams: list, teams_A_H: list, pred: list):
    
#     if teams[0] == teams_A_H[0]:
#         a = 0 # Already correctly mapped, 1st team is away team
#     else: 
#         a = 1
    
#     # Map predictions:
#     if a == 0:
#         pred_a = pred[0]
#         pred_h = pred[1]
#     else: 
#         pred_a = pred[1]
#         pred_h = pred[0]
        
#     pred_mapped = [pred_a, pred_h]
#     return pred_mapped


#################### STREAMLIT APP


st.title("Hockey Visualization App")

st.write("Got base URL:", base_url)


with st.sidebar:
    # TODO: Add input for the sidebar

    workspace = st.selectbox(label='Workspace', options=['ift-6758-projet-quipe-13'] )
    model = st.selectbox(label='Model', options=['xgboost-best-all-features', 'xgboost-best-select-features', 'xgboost-base-all-features'])
    version = st.selectbox(label='Model version', options=['1.0.0']) 

    model_button = st.button('Get Model')
    
    # (If button click and no model change)
    if model_button and model == st.session_state.model: 
        st.write(f'Model {model} already downloaded!')
    
    # (If button click and model change)
    elif model_button: # and model != st.session_state.model: 
        st.session_state['model_downloaded'] = True 
        st.session_state['model'] = model
        st.write(st.session_state.servingClient)
        st.write(st.session_state.model)

        st.session_state.servingClient.download_registry_model(workspace, st.session_state.model, version)
        st.write(f'Downloaded model:\n **{st.session_state.model}**!')

        # Reinitialize session state objects if model changes:
        st.session_state.stored_df = None 
        st.session_state.real_goals = [0,0]
        st.session_state.pred_goals = [0,0]

    # (If no button click, but page rerun: show previous model)
    elif not model_button and st.session_state.model_downloaded:  
        st.write(f'Downloaded model:\n **{st.session_state.model}**!')
    
    # (If no button click, and no previous downloaded model)
    else: 
        st.write('Waiting on **Get Model** button press...')

    


with st.container():
    # TODO: Add Game ID input
    game_id = st.text_input(label='Input Game ID:', value='2021020329', max_chars=10, label_visibility='collapsed')
    
    # Reinitialize session state objects if game_id changes:
    if game_id != st.session_state.gameClient.gameId: 
        st.session_state.stored_df = None
        st.session_state.real_goals = [0,0]
        st.session_state.pred_goals = [0,0]

    pred_button = st.button('Ping Game')
    if pred_button:
        if st.session_state.model_downloaded == False: 
            st.write('Please download model first!')
        else: 
            st.write(f'**The current game ID is {game_id}!**') 
        

              
with st.container():
    # TODO: Add Game info and predictions
    st.write('STORED DF')
    st.write(st.session_state.stored_df)

    st.subheader(f"Game goal predictions")
    if pred_button and st.session_state.model_downloaded:

        df_MODEL = st.session_state.gameClient.process_query(game_id, model_name=st.session_state.model) 
        st.write('DF_MODEL')
        st.write(df_MODEL)
        

        if df_MODEL is not None:
            pred_MODEL = st.session_state.servingClient.predict(df_MODEL)
            st.write('PRED_MODEL')
            st.write(pred_MODEL)
            
            df = pd.DataFrame(df_MODEL, columns=st.session_state.servingClient.features)   # Features list arent updated
            df = df.reset_index(drop=True)
            df['Model Output'] = pred_MODEL

            # Calculate game goal predictions (and actual)
            real_goals, teams_A_H = st.session_state.gameClient.get_real_goals()

            pred_goals = calculate_game_goals(df_MODEL, pred_MODEL, teams_A_H) ### !!!
            # st.write(teams)
            
            # pred_goals = map_teams(teams, teams_A_H, pred_goals)

            for i in range(len(teams_A_H)):
                st.session_state.pred_goals[i] += pred_goals[i]
                st.session_state.real_goals[i] = real_goals[i] 
            st.session_state.teams = teams_A_H

        else: 
            df = None
            st.write('Process query dataframe is None!')
        
        # Comparing current and previous gameId:
        if game_id == st.session_state.gameClient.gameId: 
            st.write('Concat!')
            df = pd.concat([st.session_state.stored_df, df], ignore_index=True)
            st.session_state.stored_df = df 
        else: 
            st.session_state.stored_df = df 
        

        # Getting Game info:
        if st.session_state.gameClient.game_ended: 
            st.write('**Game ended!**')
        else: 
            st.write('**Game live!**')
            last_period = int(st.session_state.stored_df['period'].values[-1:])
            last_period_sec = int(st.session_state.stored_df['periodTimeSec'].values[-1:])
            last_period_time = last_period_sec
            st.write(f'**Period: {last_period}, Period time: {last_period_time} sec**')      


        # Display Game goal predictions and info:
        col1, col2 = st.columns(2)
        pred_goals_round = np.round(st.session_state.pred_goals, decimals=1)

        delta1 = np.round(pred_goals_round[0] - st.session_state.real_goals[0], decimals=1)
        delta2 = np.round(pred_goals_round[1] - st.session_state.real_goals[1], decimals=1)
        col1.metric(label=f"**{st.session_state.teams[0]}** xG (actual)", value=f"{pred_goals_round[0]} ({st.session_state.real_goals[0]})", delta=delta1)
        col2.metric(label=f"**{st.session_state.teams[1]}** xG (actual)", value=f"{pred_goals_round[1]} ({st.session_state.real_goals[1]})", delta=delta2)

    else:
        st.write('Waiting on **Ping Game** button press...')
    
    # Display feature values and model predictions per game event:
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

# - Game goal predictions (separate the 2 teams) 💚

# - Afficher Period, Time left to Period 💚
    # Ne pas afficher if game_ended 💚


# Ping game when model isnt downloaded -> Error 💚

