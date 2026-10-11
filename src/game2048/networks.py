import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Conv2048(BaseFeaturesExtractor):
    def __init__(self, observation_space, filters=128, features_dim=256):
        super().__init__(observation_space, features_dim)
        c = observation_space.shape[0]  # 18 one-hot channels
        self.h = nn.Conv2d(c, filters, (1, 2))   # horizontal pairs
        self.v = nn.Conv2d(c, filters, (2, 1))   # vertical pairs
        self.hh = nn.Conv2d(filters, filters, (1, 2))
        self.hv = nn.Conv2d(filters, filters, (2, 1))
        self.vh = nn.Conv2d(filters, filters, (1, 2))
        self.vv = nn.Conv2d(filters, filters, (2, 1))
        with torch.no_grad():
            n = self._features(torch.zeros(1, *observation_space.shape)).shape[1]
        self.head = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())

    def _features(self, x):
        h, v = torch.relu(self.h(x)), torch.relu(self.v(x))
        second = [self.hh(h), self.hv(h), self.vh(v), self.vv(v)]
        parts = [h, v, *(torch.relu(t) for t in second)]
        return torch.cat([t.flatten(1) for t in parts], dim=1)

    def forward(self, obs):
        return self.head(self._features(obs))