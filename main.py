from train import train
from utils import render, view

if __name__ == '__main__':
    agent, env = train()
    render(agent, env)
    view()