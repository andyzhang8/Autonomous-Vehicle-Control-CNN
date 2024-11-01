# model.py
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        
        # Convolutional layers with batch normalization and ReLU activations
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 1, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(1)

        self.pool = nn.MaxPool2d(kernel_size=5, stride=5, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(169, 64)
        self.fc2 = nn.Linear(64, 64)

        # Output layers for steering and throttle with tanh and sigmoid activation
        self.steering_output = nn.Linear(64, 1)
        self.throttle_output = nn.Linear(64, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass through convolutional layers with ReLU activation and batch normalization
        x = self.bn1(self.conv1(x))
        x = torch.relu(x)
        x = self.bn2(self.conv2(x))
        x = torch.relu(x)
        x = self.bn3(self.conv3(x))
        x = torch.relu(x)
        x = self.pool(x)
        x = self.bn4(self.conv4(x))
        x = torch.relu(x)
        x = self.bn5(self.conv5(x))
        x = torch.relu(x)

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Steering output with tanh for range [-1, 1]
        steering = self.tanh(self.steering_output(x))
        
        # Throttle output with sigmoid for range [0, 1]
        throttle = self.sigmoid(self.throttle_output(x))
        
        return steering, throttle


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        
        # Convolutional layers for the state input with batch normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 1, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(1)

        self.pool = nn.MaxPool2d(kernel_size=5, stride=5, padding=1)

        # Fully connected layers with combined state and action inputs
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(170, 64)
        self.fc2 = nn.Linear(64, 64)
        
        self.q_output = nn.Linear(64, 1)

    def forward(self, state, action):
        # Pass through convolutional layers for the state input with ReLU and batch normalization
        x = self.bn1(self.conv1(state))
        x = torch.relu(x)
        x = self.bn2(self.conv2(x))
        x = torch.relu(x)
        x = self.bn3(self.conv3(x))
        x = torch.relu(x)
        x = self.pool(x)
        x = self.bn4(self.conv4(x))
        x = torch.relu(x)
        x = self.bn5(self.conv5(x))
        x = torch.relu(x)

        # Flatten and concatenate with the action input
        x = self.flatten(x)
        x = torch.cat([x, action], dim=1)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Output Q-value
        q_value = self.q_output(x)
        
        return q_value
