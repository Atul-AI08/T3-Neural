import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.input = nn.Linear(27, 128)
        self.hidden_1 = nn.Linear(128, 256)
        self.hidden_2 = nn.Linear(256, 128)
        self.hidden_3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 9)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = self.output(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)

        x = x.numpy()
        return x

    def fit(self, dataloader, epochs):
        """
        Train the model using the provided dataloader and number of epochs.
        """
        print("Starting training...")
        optim = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=5e-4)
        for i in range(epochs):
            curr_loss = 0
            print(f"Epoch {i+1}/{epochs}")
            for idx, (state, reward) in enumerate(dataloader):
                optim.zero_grad()
                output = self.forward(state)
                loss = self.criterion(output, reward)
                loss.backward()
                optim.step()
                curr_loss += loss.item()

            print(f"Total loss for epoch {i+1}: {curr_loss}")
        print("Training completed.")
