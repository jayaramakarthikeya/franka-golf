
from agent import SACAgent
from buffer import ReplayBuffer
from sai_rl import SAIClient
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import numpy as np



def train_sac(episodes=1000):
    sai = SAIClient(comp_id="franka-ml-hiring")
    env = sai.make_env()
    
    state_dim = env.observation_space.shape[0] #type: ignore
    action_dim = env.action_space.shape[0] #type: ignore
    action_bound = float(env.action_space.high[0]) #type: ignore

    agent = SACAgent(state_dim, action_dim, action_bound)
    replay_buffer = ReplayBuffer()

    writer = SummaryWriter(log_dir="runs")
    return_queue = deque(maxlen=10)
    total_steps = 0

    for ep in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        episode_reward = 0.0
        for step in range(1000):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += float(reward)
            total_steps += 1
            if len(replay_buffer) > 1000:
                agent.train(replay_buffer)
            if done:
                print(f"Episode {ep} ended at step {step}.")
                break
            
        # Log reward for every episode
        
        return_queue.append(episode_reward)
        avg_return = np.mean(return_queue)
        writer.add_scalar("AverageReturn", avg_return, total_steps)
        writer.flush()  # Ensure data is written immediately
        print(f"Episode {ep}: Total Reward = {episode_reward:.2f}, Buffer Size = {len(replay_buffer)}")

    env.close()
    writer.close()

if __name__ == "__main__":
    train_sac()