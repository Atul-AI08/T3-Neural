import nest_asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
import numpy as np
from model import Model

import warnings

warnings.filterwarnings("ignore")

model = Model()
model_2 = Model()


# Function to convert the board state into one-hot encoding
def one_hot(state):
    current_state = []

    for square in state:
        if square == 0:
            current_state.append(1)
            current_state.append(0)
            current_state.append(0)
        elif square == 1:
            current_state.append(0)
            current_state.append(1)
            current_state.append(0)
        elif square == -1:
            current_state.append(0)
            current_state.append(0)
            current_state.append(1)

    return current_state


try:
    model.load_state_dict(torch.load("tic_tac_toe.pth"))
    model_2.load_state_dict(torch.load("tic_tac_toe_2.pth"))
    print("Pre-existing model found... loading data.")
except:
    pass


# Function to predict the best move
def predict(board, team):
    if team == "O":
        with torch.no_grad():
            pre = model.predict(torch.Tensor(np.array(one_hot(board))))
    elif team == "X":
        with torch.no_grad():
            pre = model_2.predict(torch.Tensor(np.array(one_hot(board))))

    highest = -1000
    num = -1
    for j in range(0, 9):
        if board[j] == 0:
            if pre[j] > highest:
                highest = pre[j].copy()
                num = j

    return num


class Item(BaseModel):
    team: str
    board: list


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}


@app.post("/predict")
async def predict_move(data: Item):
    try:
        turn = data.team
        board = data.board
        move = predict(board, turn)
        return {"move": move}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    nest_asyncio.apply()
    print("Starting server on port 8000...")
    uvicorn.run(app, port=8000)
