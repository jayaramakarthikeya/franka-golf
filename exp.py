from sai_rl import SAIClient

sai = SAIClient(comp_id="franka-ml-hiring")
env = sai.make_env(render_mode="human")

#run the environment and display the environment

while True:
    # Reset the environment at the start of each episode
    observation = env.reset()
    done = False
    
    while not done:
        # For now, just take random actions
        action = env.action_space.sample()
        
        # Step the environment
        observation, reward, done, truncated, info = env.step(action)
        
        # Render the environment
        env.render()
        
        if done or truncated:
            break
