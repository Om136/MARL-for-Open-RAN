from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from env import RANSlicingEnv

# Main script to train and evaluate Multi-Agent PPO for RAN slicing
# Main script to train and evaluate Multi-Agent PPO for RAN slicing
# Main script to train and evaluate Multi-Agent PPO for RAN slicing
# Main script to train and evaluate Multi-Agent PPO for RAN slicing
if __name__ == "__main__":

    # Create the RAN Slicing environment with the number of slices (agents)
    env = DummyVecEnv([lambda: RANSlicingEnv(num_slices=3)])

    # Define the PPO model with MLP (Multi-Layer Perceptron) policy
    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=1e-3,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                tensorboard_log="./ppo_ran_tensorboard/")

    # Train the PPO model
    model.learn(total_timesteps=1000)

    # Save the trained PPO model
    model.save("ppo_ran_slicing")

    # ***********************************************************************
    # Load the trained PPO model
    model = PPO.load("ppo_ran_slicing")

    # Create the environment with render_mode explicitly passed
    env = DummyVecEnv([lambda: RANSlicingEnv(num_slices=3, render_mode="human")])

    # Evaluate the trained model
    obs = env.reset()  # DummyVecEnv.reset() returns only the observation
    for step in range(100):
        # Get action from the trained model
        action, _states = model.predict(obs)
        
        # Take a step in the environment with the selected action
        obs, reward, done, info = env.step(action)
        
        # Access the original environment and call its render()
        env.get_attr('render', 0)[0]()  # Call render() for the first environment

        if done.any():  # Check if any environment is done
            break
