"""
If you are in the same directory as this file (streamlit_app.py), you can run run the app using gunicorn:
    
    $ streamlit run streamlit_app.py --server.port=8892 --server.address=0.0.0.0

"""
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

if 'teams_full' not in st.session_state:
    st.session_state.teams_full = None



#################### FUNCTION DEFINITION

def calculate_game_goals(df: pd.DataFrame, pred: pd.DataFrame, teams_A_H: list):
    """
    Sum over model_pred for each team 
        Input: 
            df (DataFrame), with feature values 
            pred (DataFrame), model prediction for every event
            teams_A_H (list), teams tricode: [away, home]
        Output: 
            pred_goals (list), predicted number of goals for each team
            teams (list), abbreviation for each team

    """
    df = df.reset_index(drop=True)

    df['Model Output'] = pred

    pred_team1 = df.loc[df['team']==teams_A_H[0] , 'Model Output']
    sum_pred_team1 = pred_team1.sum()
    pred_team2 = df.loc[df['team']==teams_A_H[1] , 'Model Output']
    sum_pred_team2 = pred_team2.sum()

    pred_goals = [sum_pred_team1, sum_pred_team2]

    return pred_goals


#################### STREAMLIT APP

st.title("Hockey Visualization App")
st.write("Got base URL:", base_url)


with st.sidebar:
    # TODO: Add input for the sidebar

    workspace = st.selectbox(label='Workspace', options=['ift-6758-projet-quipe-13'] )
    model = st.selectbox(label='Model', options=['xgboost-best-all-features', 'xgboost-best-select-features', 'xgboost-base-all-features'])
    version = st.selectbox(label='Model version', options=['1.0.0']) 

    model_button = st.button('Get Model')
    
    # If button click and no model change
    if model_button and model == st.session_state.model: 
        st.write(f':red[Model {model} already chosen!]')
    
    # If button click and model change
    elif model_button: # and model != st.session_state.model: 
        st.session_state['model_downloaded'] = True 
        st.session_state['model'] = model

        st.session_state.servingClient.download_registry_model(workspace, st.session_state.model, version)
        st.write(f'Got model:\n **{st.session_state.model}**!')

        # Reinitialize session state objects if model changes
        st.session_state.stored_df = None 
        st.session_state.real_goals = [0,0]
        st.session_state.pred_goals = [0,0]
        st.session_state.gameClient.gameId = 0

    # If no button click, but page rerun: show previous model
    elif not model_button and st.session_state.model_downloaded:  
        st.write(f'Got model:\n **{st.session_state.model}**!')
    
    # If no button click, and no previous downloaded model
    else: 
        st.write('Waiting on **Get Model** button press...')

    

with st.container():
    # TODO: Add Game ID input
    st.write("Input game ID:")
    game_id = st.text_input(label='Input Game ID:', value='2021020329', max_chars=10, label_visibility='collapsed')
    
    # Reinitialize session state objects if game_id changes
    if game_id != st.session_state.gameClient.gameId: 
        st.session_state.stored_df = None
        st.session_state.real_goals = [0,0]
        st.session_state.pred_goals = [0,0]

    pred_button = st.button('Ping Game')
    if pred_button:
        # If no model was downloaded first, ask to download model
        if st.session_state.model_downloaded == False: 
            st.write(':red[Please download model first!]')
        else: 
            st.write(f'**The current game ID is {game_id}!**') 
        

st.write('')
st.write('')

              
with st.container():
    # TODO: Add Game info and predictions
    st.write('STORED DF')
    st.write(st.session_state.stored_df)

    st.header(f"Game goal predictions")
    if pred_button and st.session_state.model_downloaded:
        
        # Get dataframe of new events
        df_MODEL = st.session_state.gameClient.process_query(game_id, model_name=st.session_state.model) 
        st.write('DF_MODEL')
        st.write(df_MODEL)
        
        # If there are new events: 
        if df_MODEL is not None: 
            # Make predictions on events 
            pred_MODEL = st.session_state.servingClient.predict(df_MODEL)
            st.write('PRED_MODEL')
            st.write(pred_MODEL)
            
            df = pd.DataFrame(df_MODEL, columns=st.session_state.servingClient.features) 
            df = df.reset_index(drop=True)
            df['Model Output'] = pred_MODEL

            # Calculate game actual goals and goal predictions 
            real_goals, teams_A_H, teams_full = st.session_state.gameClient.get_scores()
            pred_goals = calculate_game_goals(df_MODEL, pred_MODEL, teams_A_H) 
             
            for i in range(len(teams_A_H)):
                st.session_state.pred_goals[i] += pred_goals[i]
                st.session_state.real_goals[i] = real_goals[i] 
            st.session_state.teams = teams_A_H
            st.session_state.teams_full = teams_full

        else: 
            df = None
            st.write(':red[No new events!]')
        
        # Comparing current and previous gameId:
        if game_id == st.session_state.gameClient.gameId: 
            # st.write('Concat!')
            df = pd.concat([st.session_state.stored_df, df], ignore_index=True)
            st.session_state.stored_df = df 
        else: 
            st.session_state.stored_df = df 
        

        # Getting Game info:
        st.subheader(f"{st.session_state.teams_full[0]} VS {st.session_state.teams_full[1]}")

        period, periodTimeRemaining = st.session_state.gameClient.get_period_info()
        if st.session_state.gameClient.game_ended: 
            st.write('**Game ended!**')
            st.write(f'Game end at: **Period:** {period}  --  **Period time remaining:** {periodTimeRemaining}')  
        else: 
            st.write('**Game live!**')        
            st.write(f'**Period:** {period}  --  **Period time remaining:** {periodTimeRemaining}')      


        # Display game goal predictions and info:
        col1, col2 = st.columns(2)
        pred_goals_round = np.round(st.session_state.pred_goals, decimals=1)

        delta1 = float(np.round(pred_goals_round[0] - st.session_state.real_goals[0], decimals=1))
        delta2 = float(np.round(pred_goals_round[1] - st.session_state.real_goals[1], decimals=1))
        col1.metric(label=f"**{st.session_state.teams[0]}** xG (actual)", value=f"{pred_goals_round[0]} ({st.session_state.real_goals[0]})", delta=delta1)
        col2.metric(label=f"**{st.session_state.teams[1]}** xG (actual)", value=f"{pred_goals_round[1]} ({st.session_state.real_goals[1]})", delta=delta2)

    else:
        st.write('Waiting on **Ping Game** button press...')
    
    st.write('')
    st.write('')

    # Display feature values and model predictions per game event:
    st.header(f"Game data and corresponding model predictions")
    if pred_button and st.session_state.model_downloaded:
        st.write(st.session_state.stored_df)
    else:
        st.write('Waiting on **Ping Game** button press...')

