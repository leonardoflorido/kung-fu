import cv2
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributions as distributions
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

class Network(nn.Module):
    def __init__(self, action_size):
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 4, out_channels = 32)