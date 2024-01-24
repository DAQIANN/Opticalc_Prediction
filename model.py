import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
import import_ipynb
import pandas as pd

from csvload import FULL_COLUMNS, df, train, test, all_column_train, all_column_y, all_column_test, all_columns_test_result, no_na_train, no_na_train_y, no_na_test, no_na_test_y

# X_train_full = torch.tensor(all_column_train[FULL_COLUMNS].values, dtype=torch.float32)
# X_test_full = torch.tensor(all_column_test[FULL_COLUMNS].values, dtype=torch.float32)
# y_train_full = torch.tensor(all_column_y.values, dtype=torch.float32)
# y_test_full = torch.tensor(all_columns_test_result.values, dtype=torch.float32)

X_train_full = torch.tensor(no_na_train.values, dtype=torch.float32)
X_test_full = torch.tensor(no_na_test.values, dtype=torch.float32)
y_train_full = torch.tensor(no_na_train_y.values, dtype=torch.float32)
y_test_full = torch.tensor(no_na_test_y.values, dtype=torch.float32)

class linearRegression(nn.Module):
    def __init__(self, input_dim):
        super(linearRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 3)
        self.fc4 = nn.Linear(3, 1)
    
    def forward(self, d):
        out = torch.relu(self.fc1(d))
        out=torch.relu(self.fc2(out))
        out=torch.relu(self.fc3(out))
        out=self.fc4(out)
        return out
    
input_dim = X_train_full.shape[1]
torch.manual_seed(42)
model = linearRegression(input_dim)

loss = nn.MSELoss()
optimizers = optim.Adam(params=model.parameters(), lr=0.01)

num_of_epochs = 1000
losses = []
print(y_train_full.squeeze().shape)
for i in range(num_of_epochs):
    y_train_prediction=model(X_train_full)
    loss_value = loss(y_train_prediction.squeeze(), y_train_full.squeeze())
    losses.append(loss_value)
    optimizers.zero_grad()
    loss_value.backward()
    optimizers.step()

with torch.no_grad():
    model.eval()
    y_test_prediction = model(X_test_full)
    test_loss=loss(y_test_prediction, y_test_full)
    print(y_test_prediction)
    print(f'Test loss value : {test_loss.item():.4f}')