import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input, Dense, Lambda  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.utils import register_keras_serializable  # type: ignore

# Disable oneDNN optimizations for potential compatibility issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

@register_keras_serializable()
def _combine_streams(inputs):
    val, adv = inputs
    return val + (adv - tf.reduce_mean(adv, axis=1, keepdims=True))

app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing) to allow requests from any domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define file paths - locally and on PythonAnywhere
weights_path = os.environ.get(
    'MODEL_WEIGHTS_PATH',
    '/home/Rdhirschel/mysite/public/assets/mancala_agent_final.weights.h5'
    if 'VERCEL_ENV' in os.environ
    else 'public/assets/mancala_agent_final.weights.h5'
)

# Manually rebuild the model architecture to make the loading process in the pythonanwhere server work
def build_model(lr=0.001, state_size=15, action_size=6):
    """Builds a dueling DQN model for Q-learning."""
    input_layer = Input(shape=(state_size,))
    
    x = Dense(512, activation='relu6', kernel_initializer='he_uniform')(input_layer)
    x = Dense(256, activation='relu6', kernel_initializer='he_uniform')(x)
    common = Dense(128, activation='relu6', kernel_initializer='he_uniform')(x)
    
    # Value stream
    value_fc = Dense(64, activation='relu6', kernel_initializer='he_uniform')(common)
    value_fc = Dense(32, activation='relu6', kernel_initializer='he_uniform')(value_fc)
    value_fc = Dense(16, activation='relu6', kernel_initializer='he_uniform')(value_fc)
    value = Dense(1, activation='linear', kernel_initializer='he_uniform')(value_fc)
    
    # Advantage stream
    adv_fc = Dense(64, activation='relu6', kernel_initializer='he_uniform')(common)
    adv_fc = Dense(32, activation='relu6', kernel_initializer='he_uniform')(adv_fc)
    advantage = Dense(action_size, activation='linear', kernel_initializer='he_uniform')(adv_fc)
    
    # Combine the streams
    q_values = Lambda(_combine_streams)([value, advantage])
    model = Model(inputs=input_layer, outputs=q_values)
    model.compile(optimizer=Adam(learning_rate=lr), loss='huber')
    return model


# Create model and load weights
agent = build_model()
try:
    agent.load_weights(weights_path)
    print("Model weights loaded successfully.")
except Exception as e:
    print("Error loading model weights:", e)
    raise RuntimeError("Failed to load model weights") from e

class BoardState(BaseModel):
    state: list

@app.get("/")
def read_root():
    return {"message": "Mancala Agent API"}

@app.post("/best_move/")
def get_best_move(board_state: BoardState):
    if len(board_state.state) != 15:
        raise HTTPException(status_code=400, detail="Input shape must be (15,)")

    # Switch board positions between player 0 and 1
    if board_state.state[-1] == 0:
        for i in range(7):
            board_state.state[i], board_state.state[i + 7] = board_state.state[i + 7], board_state.state[i]

    try:
        state = np.array(board_state.state).reshape(1, -1)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")

    act_values = agent.predict(state, verbose=0)
    best_moves = np.argsort(act_values[0])[::-1].tolist()

    return {"best_moves": best_moves}