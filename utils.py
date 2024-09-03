import gymnasium as gym
import glob
import io
import base64
import imageio
from IPython.display import HTML, display

from models.preprocessatari import PreprocessAtari

def make_env():
  env = gym.make("KungFuMasterDeterministic-v0", render_mode = 'rgb_array')
  env = PreprocessAtari(env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4)
  
  return env

def evaluate(agent, env, n_episodes = 1):
  episodes_rewards = []

  for _ in range(n_episodes):
    state, _ = env.reset()
    total_reward = 0
    
    while True:
      action = agent.act(state)
      state, reward, done, info, _ = env.step(action[0])
      total_reward += reward

      if done:
        break
    
    episodes_rewards.append(total_reward)
  
  return episodes_rewards

def render(agent, env):
  state, _ = env.reset()
  done = False
  frames = []
  
  while not done:
    frame = env.render()
    frames.append(frame)
    action = agent.act(state)
    state, reward, done, _, _ = env.step(action[0])
  
  env.close()
  imageio.mimsave('video.mp4', frames, fps=30)

def view():
    mp4list = glob.glob('*.mp4')
    
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")
