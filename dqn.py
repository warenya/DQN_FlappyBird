import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Implements a dueling deep Q network
    """
    def __init__(self, state_dim, action_dim, hidden_dim=32, enable_dueling_dqn=True):
        super(DQN, self).__init__()

        self.enable_dueling_dqn = enable_dueling_dqn

        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.enable_dueling_dqn:
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value_layer = nn.Linear(256, 1)

            self.fc_advantage = nn.Linear(hidden_dim, 256)
            self.advantage_layer = nn.Linear(256, action_dim)

        else:
            self.output_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            v = F.relu(self.fc_value(x))
            V = self.value_layer(v)

            a = F.relu(self.fc_advantage(x))
            A = self.advantage_layer(a)

            Q = V + A - torch.mean(A,dim=1, keepdim=True)

        else:
            Q = self.output(x)
        return Q

if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim)
    state = torch.randn(10, state_dim)
    output = net(state)
    print(output)