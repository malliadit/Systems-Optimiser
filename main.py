import sys
import os
import pickle
sys.path.append("src")
from src.sys_env import SystemEnvironment

# Setup
systems = {
    'CoolingSystem': 3,
    'PowerUnit': 2.5,
    'NavigationAI': 1.5,
    'CommsArray': 1
}

target_efficiency = {
    'CoolingSystem': 6.0,
    'PowerUnit': 5.0,
    'NavigationAI': 4.5,
    'CommsArray': 3.0
}

# Create and train the environment
env = SystemEnvironment(systems, target_efficiency, energy_per_cycle=6)

print("\U0001F393 Training the optimization agent...")
env.train(episodes=500)
print()

# Test run with optimal plan
print("\U0001F4DA Learned Optimization Plan:\n")
optimal_plan = env.get_optimal_plan()

for step in optimal_plan:
    print(f"Allocate {step['effort']} units of energy to {step['system']} → Reward: {step['reward']:.2f}")

print(f"\nFinal Efficiency: {optimal_plan[-1]['efficiency_after']}")
print(f"Target Efficiency: {target_efficiency}")

# Save the trained Q-table
env.save_q_table()


# Interactive assistant mode
print("\n🧠 Starting interactive energy allocation session!\n")

env.reset()  # Reset for interactive session
done = False

while not done:
    current_state = env._get_state()

    print("\n📊 Current Efficiency Levels:")
    for sys_name, val in current_state['efficiency'].items():
        print(f"  {sys_name}: {val:.2f} / {target_efficiency[sys_name]}")

    # User inputs what system to allocate energy to
    system = input("\nWhich system do you want to allocate energy to? ").strip()
    effort = int(input("How many units of energy to allocate? ").strip())

    # Step the environment with user input
    state, reward, done = env.allocate(system, effort)
    print(f"✅ Updated efficiency. Reward: {reward:.2f}")

    if done:
        print("\n🎉 You've completed the allocation cycle.")
        print("\n📊 Final Efficiency Levels at cycle end:")
        for sys_name, val in state['efficiency'].items():
            print(f"  {sys_name}: {val:.2f} / {target_efficiency[sys_name]}")
        break

    # Get recommendation for next system allocation
    try:
        next_action = env.get_recommendation()
        print(f"\n👉 Based on your progress, recommended allocation: {next_action[1]} units of energy to {next_action[0]}.")
    except Exception as e:
        print(f"\n⚠️ Could not get recommendation: {e}")
