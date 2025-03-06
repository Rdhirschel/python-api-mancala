import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to the ONNX model file
onnx_model_path = os.environ.get("MODEL_PATH", "public/assets/mancala_agent_final.onnx")

# Initialize ONNX Runtime session
try:
    # Load the ONNX model and get input/output names
    # Why name is needed is because ONNX model can have multiple inputs and outputs. The output result from .run() is a list of output tensors.
    # In our case, we only have 1 input tensor and 1 output tensor, so we can just get the first element of the list.
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
except Exception as e:
    print(f"Error loading ONNX model: {e}")

# Define the pydantic model for incoming board state
class BoardState(BaseModel):
    state: list

@app.get("/")
def read_root():
    return {"message": "Mancala Agent API"}

def predict_onnx(state: np.ndarray) -> np.ndarray:
    """Perform inference using ONNX Runtime."""
    outputs = session.run([output_name], {input_name: state.astype(np.float32)})
    return outputs[0] # outputs is a list of output tensors, we only have 1 output tensor which is act_values ([1, 6])

@app.post("/best_move/")
def get_best_move(board_state: BoardState):
    # Validate board shape
    if len(board_state.state) != 15:
        raise HTTPException(status_code=400, detail="Input shape must be (15,)")
    
    # Switch board positions between player 0 and player 1 if needed
    if board_state.state[-1] == 0:
        for i in range(7):
            board_state.state[i], board_state.state[i + 7] = board_state.state[i + 7], board_state.state[i]
    
    try:
        # Reshape input into a 2D array
        state_array = np.array(board_state.state).reshape(1, -1)
        
        # Run inference
        act_values = predict_onnx(state_array)
        
        # Get best moves by sorting output predictions
        best_moves = np.argsort(act_values[0])[::-1].tolist()
        
        return {"best_moves": best_moves}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")