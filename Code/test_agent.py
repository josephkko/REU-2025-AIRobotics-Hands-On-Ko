from robothor_env import RoboThorEnv
from stable_baselines3 import PPO
import json
from PIL import Image
import matplotlib.pyplot as plt
import openai

# --- Step 1: Run test episode and log actions ---

# Load environment and model
env = RoboThorEnv()
model = PPO.load("ppo_robothor")

obs = env.reset()
done = False
total_reward = 0
steps = 0

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    steps += 1
    print(f"Step {steps}, Reward: {reward}")

# --- Step 2: Save action log and last frame ---

# Save action log
with open("fail_log.json", "w") as f:
    json.dump(env.logs, f, indent=2)

# Save last frame
Image.fromarray(obs).save("failure_frame.png")

# Close environment
env.close()

# --- Step 3: Prepare LLM prompt ---

with open("fail_log.json") as f:
    logs = json.load(f)

prompt_template = """
You are an AI explainability expert. A robot was navigating in an indoor scene but failed to reach its goal.
Below is the action log and the final image seen by the robot.

Action Log:
{logs}

What obstacles or planning failures can you detect?
Explain possible reasons for failure, and suggest better strategies or corrective steps.
"""

# Use first 10 log entries for clarity
prompt = prompt_template.format(logs=json.dumps(logs[:10], indent=2))

# --- Step 4: Query GPT ---

# New client-based API (openai>=1.0.0)
client = openai.OpenAI(api_key="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")

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
)

# --- Step 5: Display visual feedback ---

# Show final frame
img = Image.open("failure_frame.png")
plt.imshow(img)
plt.title("Robot's Last View")
plt.axis("off")
plt.show()

# Print LLM explanation
print("\n=== LLM EXPLANATION ===")
print(response.choices[0].message.content)

# Final episode result
print(f"\n=== TEST RESULTS ===")
print(f"Total reward: {total_reward}")
print(f"Total steps: {steps}")
