from robothor_env import RoboThorEnv
from stable_baselines3 import PPO
# from stable_baselines3.common.env_checker import check_env  # DO NOT USE check_env

# Create environment
env = RoboThorEnv()

# Do not call check_env() → it will crash with gym==0.21.0

# Create PPO model
model = PPO("CnnPolicy", env, verbose=1)

# Train agent
model.learn(total_timesteps=5000)

# Save model
model.save("ppo_robothor")

# Close env
env.close()
