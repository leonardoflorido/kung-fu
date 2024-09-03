import numpy as np
import tqdm

from agents.agent import Agent
from models.envbatch import EnvBatch
from utils import evaluate, make_env

def train():
  # Initializing the hyperparameters
  number_environments = 10

  env = make_env()
  state_shape = env.observation_space.shape
  number_actions = env.action_space.n
  
  print("State shape:", state_shape)
  print("Number actions:", number_actions)
  print("Action names:", env.env.env.get_action_meanings())
  
  agent = Agent(number_actions)
  env_batch = EnvBatch(number_environments)
  batch_states = env_batch.reset()

  with tqdm.trange(0, 3001) as progress_bar:
    for i in progress_bar:
      batch_actions = agent.act(batch_states)
      batch_next_states, batch_rewards, batch_dones, _ = env_batch.step(batch_actions)
      batch_rewards *= 0.01
      agent.step(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
      batch_states = batch_next_states

      if i % 1000 == 0:
        print("Average agent reward: ", np.mean(evaluate(agent, env, n_episodes = 10)))
  
  return agent, env
