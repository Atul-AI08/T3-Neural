import random
import numpy as np
from tqdm import tqdm

import torch
from model import Model

import warnings

warnings.filterwarnings("ignore")


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        state = self.X[idx]
        reward = self.Y[idx]
        state_tensor = torch.Tensor(state)
        reward_tensor = torch.Tensor(reward)
        return state_tensor, reward_tensor


def one_hot(state):
    current_state = []

    for square in state:
        if square == 0:
            current_state.extend([1, 0, 0])
        elif square == 1:
            current_state.extend([0, 1, 0])
        elif square == -1:
            current_state.extend([0, 0, 1])

    return current_state


def get_outcome(state):
    total_reward = 0

    win_criterion = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ]

    for i in win_criterion:
        if state[i[0]] != 0:
            if state[i[0]] == state[i[1]] == state[i[2]]:
                return state[i[0]]

    return total_reward


# Train the given model with provided states and Q-values
def train_model(states, q_values, model):
    print(f"Training model with {len(states)} states...")
    model.train()

    new_states = np.array([one_hot(state) for state in states])
    q_values = np.array(q_values)

    dataset = CustomDataset(new_states, q_values)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model.fit(dataloader, epochs=4)


# Process games from replay memory to generate training data
def process_games(replay_memory, model, model_2):
    global x_train
    print("Processing replay memory...")
    states, q_values, states_2, q_values_2 = [], [], [], []

    for game in replay_memory:
        total_reward = get_outcome(game[-1])
        for i in range(0, len(game) - 1):
            if i % 2 == 0:
                for j in range(0, 9):
                    if not game[i][j] == game[i + 1][j]:
                        reward_vector = np.zeros(9)
                        reward_vector[j] = total_reward * (
                            gamma ** (len(game) - i // 2 - 1)
                        )
                        states.append(game[i].copy())
                        q_values.append(reward_vector.copy())
            else:
                for j in range(0, 9):
                    if not game[i][j] == game[i + 1][j]:
                        reward_vector = np.zeros(9)
                        reward_vector[j] = (
                            -1 * total_reward * (gamma ** (len(game) - i // 2 - 1))
                        )
                        states_2.append(game[i].copy())
                        q_values_2.append(reward_vector.copy())

    if x_train:
        train_model(states, q_values, model)
    else:
        train_model(states_2, q_values_2, model_2)

    x_train = not x_train
    print("Replay memory processing completed.")


model = Model()
model_2 = Model()

gamma = 0.9
x_train = True
training_rounds = 5

for _ in tqdm(range(training_rounds), desc="Training Rounds"):
    try:
        model.load_state_dict(torch.load("tic_tac_toe.pth"))
        model_2.load_state_dict(torch.load("tic_tac_toe_2.pth"))
        print("Loaded pre-trained models.")
    except FileNotFoundError:
        print("No pre-trained models found. Starting from scratch.")

    games = []
    total_games = 2000
    e_greedy = 0.7

    for i in range(0, total_games):
        board = [0] * 9
        current_game = [board.copy()]
        playing, nn_turn = True, True

        while playing:
            if nn_turn:
                if random.uniform(0, 1) <= e_greedy:
                    while True:
                        move = random.randint(0, 8)
                        if board[move] == 0:
                            board[move] = 1
                            break

                else:
                    pre = model.predict(torch.Tensor(np.array(one_hot(board))))
                    highest = -1000
                    num = -1
                    for j in range(0, 9):
                        if board[j] == 0:
                            if pre[j] > highest:
                                highest = pre[j].copy()
                                num = j

                    board[num] = 1

            else:
                if random.uniform(0, 1) <= e_greedy:
                    while True:
                        move = random.randint(0, 8)
                        if board[move] == 0:
                            board[move] = -1
                            break

                else:
                    pre = model_2.predict(torch.Tensor(np.array(one_hot(board))))
                    highest = -1000
                    num = -1
                    for j in range(0, 9):
                        if board[j] == 0:
                            if pre[j] > highest:
                                highest = pre[j].copy()
                                num = j

                    board[num] = -1

            current_game.append(board.copy())

            if get_outcome(board) != 0 or all(b != 0 for b in board):
                playing = False

            nn_turn = not nn_turn

        games.append(current_game)

    process_games(games, model, model_2)

# Save final models
torch.save(model.state_dict(), "tic_tac_toe.pth")
torch.save(model_2.state_dict(), "tic_tac_toe_2.pth")
print("Final models saved.")
