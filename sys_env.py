import numpy as np
import pickle
import os
import random
from datetime import datetime

class SystemEnvironment:
    def __init__(self, systems, target_efficiency, energy_per_cycle=6):
        self.systems = systems  
        self.target_efficiency = target_efficiency 
        self.default_energy_per_cycle = energy_per_cycle
        self.energy_per_cycle = energy_per_cycle
        self.reset()
        self.Q = {}  # Q-table
        self.is_trained = False
        self.training_history = []  
        self.metadata = {
            'systems': systems,
            'target_efficiency': target_efficiency,
            'energy_per_cycle': energy_per_cycle,
            'episodes_trained': 0,
            'last_training_date': None
        }
        self.user_inputs = []  # Store user inputs

    def reset(self, energy_per_cycle=None, preserve_efficiency=False):
        """Reset the environment, optionally with a new energy_per_cycle value"""
        if energy_per_cycle is not None:
            self.energy_per_cycle = energy_per_cycle
            print(f"🔄 Energy per cycle updated to: {self.energy_per_cycle}")
        
        self.time_remaining = self.energy_per_cycle
        
        # Reset efficiency if preserve_efficiency is False
        if not preserve_efficiency:
            self.efficiency = {sys: 0.0 for sys in self.systems}
        else:
            print("📊 Preserving current efficiency levels for new cycle")
        
        self.time_spent = {sys: 0 for sys in self.systems}
        self.current_system = None
        self.total_reward = 0.0
        return self._get_state()

    def _get_state(self):
        return {
            'current_system': self.current_system,
            'time_remaining': self.time_remaining,
            'time_spent': self.time_spent.copy(),
            'efficiency': self.efficiency.copy()
        }

    def _check_target_efficiency_met(self):
        """Check if all target efficiencies have been met"""
        for sys in self.target_efficiency:
            if self.efficiency[sys] < self.target_efficiency[sys]:
                return False
        return True

    def allocate(self, system, effort):
        if self.time_remaining <= 0:
            raise Exception("No energy left in the cycle!")
        if effort > self.time_remaining:
            effort = self.time_remaining  # You cant have more time than what is allotted.

        difficulty = self.systems[system]
        total_gain = 0.0

        for _ in range(effort):
            gain = round(np.random.uniform(0.8, 1.2) * (1 / difficulty), 2)
            # make sure that efficiency does not go over the target and hence, if it does, just cap it to the target.
            new_efficiency = self.efficiency[system] + gain
            self.efficiency[system] = min(new_efficiency, self.target_efficiency[system])

            self.time_spent[system] += 1
            self.time_remaining -= 1
            total_gain += gain

        self.current_system = system

        reward = self._calculate_reward()
        self.total_reward += reward
        
        # Check if done: either no energy left or all targets met
        done = self.time_remaining == 0 or self._check_target_efficiency_met()
        
        # If targets are met, add bonus reward
        if self._check_target_efficiency_met():
            reward += 10  # Bonus for meeting all targets
            if self.time_remaining > 0:
                print(f"🎯 All target efficiencies met! Bonus reward added. Energy saved: {self.time_remaining}")

        return self._get_state(), reward, done

    def _calculate_reward(self):
        reward = 0
        for sys in self.target_efficiency:
            diff = abs(self.efficiency[sys] - self.target_efficiency[sys])
            reward -= diff  # less difference would mean better reward
        return reward

    def _state_key(self):
        key = []
        for sys in sorted(self.efficiency.keys()):
            key.append(str(round(self.efficiency[sys], 1)))
        return '_'.join(key)

    def get_possible_actions(self):
        max_effort = self.time_remaining  # only allow efforts within remaining energy
        return [(sys, dur) for sys in self.systems for dur in range(1, max_effort + 1)]

    def train(self, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.2):
        episode_rewards = []
        
        for episode in range(episodes):
            self.reset()
            done = False
            episode_reward = 0

            while not done:
                state_key = self._state_key()
                possible_actions = self.get_possible_actions()  # dynamically filtered

                if random.random() < epsilon:
                    action = random.choice(possible_actions)
                else:
                    q_vals = self.Q.get(state_key, {})
                    if not q_vals:
                        action = random.choice(possible_actions)
                    else:
                        # Filter q_vals keys by possible_actions only
                        filtered_q_vals = {a: q for a, q in q_vals.items() if a in possible_actions}
                        if filtered_q_vals:
                            action = max(filtered_q_vals, key=filtered_q_vals.get)
                        else:
                            action = random.choice(possible_actions)

                next_state, reward, done = self.allocate(*action)
                next_key = self._state_key()
                episode_reward += reward

                old_q = self.Q.get(state_key, {}).get(action, 0.0)
                future_qs = self.Q.get(next_key, {})
                max_future = max(future_qs.values()) if future_qs else 0.0
                new_q = old_q + alpha * (reward + gamma * max_future - old_q)

                if state_key not in self.Q:
                    self.Q[state_key] = {}
                self.Q[state_key][action] = new_q

            episode_rewards.append(episode_reward)
            
            if episode % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                print(f"Episode {episode} complete. Avg reward (last 50): {avg_reward:.2f}")

        self.is_trained = True
        self.metadata['episodes_trained'] += episodes
        self.metadata['last_training_date'] = datetime.now().isoformat()
        self.training_history.extend(episode_rewards)
        print("Training finished!")

    def get_recommendation(self, state=None):
        if not self.is_trained:
            raise Exception("Model not trained yet!")
        if state is None:
            state = self._get_state()
        state_key = self._state_key()
        q_vals = self.Q.get(state_key, {})
        if not q_vals:
            return random.choice(self.get_possible_actions())
        return max(q_vals, key=q_vals.get)

    def get_optimal_plan(self, energy_per_cycle=None, preserve_efficiency=False):
        if not self.is_trained:
            raise Exception("Model not trained yet! Train the model first! ")
        
        # Store current state if we need to preserve it
        if preserve_efficiency:
            current_efficiency = self.efficiency.copy()
        
        # Use custom energy if provided, otherwise use current setting
        if energy_per_cycle is not None:
            state = self.reset(energy_per_cycle, preserve_efficiency)
        else:
            state = self.reset(preserve_efficiency=preserve_efficiency)
            
        plan = []
        done = False
        while not done:
            action = self.get_recommendation(state)
            system, effort = action
            state, reward, done = self.allocate(system, effort)
            plan.append({
                'system': system,
                'effort': effort,
                'reward': reward,
                'efficiency_after': state['efficiency'].copy(),
                'energy_remaining': state['time_remaining']
            })
        
        # Restore original efficiency if we were preserving it
        if preserve_efficiency:
            self.efficiency = current_efficiency
            
        return plan

    def save_model(self, filename='system_model'):
        """
        Save the complete model with timestamp.
        
        Args:
            filename: Base filename (without extension)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data to save
        model_data = {
            'Q_table': self.Q,
            'metadata': self.metadata,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'timestamp': timestamp
        }
        
        # Save timestamped version
        pickle_filename = f"{filename}_{timestamp}.pkl"
        try:
            with open(pickle_filename, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"✅ Model saved: {pickle_filename}")
        except Exception as e:
            print(f"❌ Failed to save timestamped model: {e}")
        
        # Also save a "latest" version for easy loading
        latest_filename = f"{filename}_latest.pkl"
        try:
            with open(latest_filename, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"✅ Latest model saved: {latest_filename}")
        except Exception as e:
            print(f"❌ Failed to save latest: {e}")

    def load_model(self, filename='system_model_latest.pkl'):
        """
        Load a complete model from file.
        
        Args:
            filename: Full filename including extension
        """
        if not os.path.exists(filename):
            print(f"⚠️ Model file '{filename}' doesn't exist. Starting fresh.")
            self.Q = {}
            self.is_trained = False
            return False
        
        if os.path.getsize(filename) == 0:
            print(f"⚠️ Model file '{filename}' is empty. Starting fresh.")
            self.Q = {}
            self.is_trained = False
            return False
        
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            # Validate loaded data
            if not isinstance(model_data, dict) or 'Q_table' not in model_data:
                print(f"⚠️ Invalid model format in '{filename}'. Starting fresh.")
                self.Q = {}
                self.is_trained = False
                return False
            
            self.Q = model_data['Q_table']
            self.metadata.update(model_data.get('metadata', {}))
            self.training_history = model_data.get('training_history', [])
            self.is_trained = model_data.get('is_trained', False)
            
            print(f"📦 Model loaded from {filename}")
            print(f"   Episodes trained: {self.metadata.get('episodes_trained', 'Unknown')}")
            print(f"   Last training: {self.metadata.get('last_training_date', 'Unknown')}")
            print(f"   Q-table size: {len(self.Q)} states")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.Q = {}
            self.is_trained = False
            return False

    def start_interactive_session(self, user_inputs_file="user_inputs.pkl", preserve_efficiency=False):
      
        print("🧠 Starting interactive energy allocation session!")
        
        try:
            new_energy = input(f"\n⚡ Enter energy per cycle (current: {self.energy_per_cycle}, press Enter to keep current): ").strip()
            if new_energy:
                new_energy = int(new_energy)
                print(f"🔄 Energy per cycle set to: {new_energy}")
            else:
                new_energy = None
        except ValueError:
            print("⚠️ Invalid input. Using current energy setting.")
            new_energy = None

        # Only check for previous session if not preserving efficiency
        if not preserve_efficiency and os.path.exists(user_inputs_file):
            use_previous = input("📂 Found previous session! Continue from where you left off? (y/n): ").strip().lower()
            if use_previous == 'y':
                try:
                    with open(user_inputs_file, 'rb') as f:
                        self.user_inputs = pickle.load(f)
                    print(f"✅ Loaded {len(self.user_inputs)} previous inputs")
                    
                    # Display what was done before
                    print("\n📝 Previous session summary:")
                    for i, input_data in enumerate(self.user_inputs, 1):
                        print(f"  {i}. {input_data['system']}: {input_data['effort']} units")
                    
                except Exception as e:
                    print(f"❌ Failed to load previous inputs: {e}")
                    self.user_inputs = []
            else:
                self.user_inputs = []
        elif preserve_efficiency:
            # Starting fresh with preserved efficiency
            self.user_inputs = []
            print("🔧 Starting new cycle with preserved efficiency levels")
        else:
            self.user_inputs = []

        # Show optimal plan if trained
        if self.is_trained:
            print("\n🎯 Optimal Plan for this energy cycle:")
            try:
                if new_energy is not None:
                    optimal_plan = self.get_optimal_plan(new_energy, preserve_efficiency)
                else:
                    optimal_plan = self.get_optimal_plan(preserve_efficiency=preserve_efficiency)
                for i, step in enumerate(optimal_plan, 1):
                    print(f"  {i}. Allocate {step['effort']} units to {step['system']} → Reward: {step['reward']:.2f}")
                    if step['energy_remaining'] == 0:
                        print(f"     (Energy exhausted)")
                    elif all(step['efficiency_after'][sys] >= self.target_efficiency[sys] for sys in self.target_efficiency):
                        print(f"     (All targets met! Energy saved: {step['energy_remaining']})")
            except Exception as e:
                print(f"⚠️ Could not generate optimal plan: {e}")

        # Reset environment for interactive session
        self.reset(new_energy, preserve_efficiency=preserve_efficiency)
        done = False

        # If we have previous inputs, replay them first
        if self.user_inputs:
            print("\n⏩ Replaying previous moves...")
            for i, prev_input in enumerate(self.user_inputs):
                try:
                    print(f"  {i+1}. {prev_input['system']}: {prev_input['effort']} units")
                    state, reward, done = self.allocate(prev_input['system'], prev_input['effort'])
                    if done:
                        if self._check_target_efficiency_met():
                            print("🎯 All target efficiencies were met in previous session!")
                        else:
                            print("⚡ Energy was exhausted in previous session!")
                        break
                except Exception as e:
                    print(f"❌ Error replaying move {i+1}: {e}")
                    break

        # Continue with new inputs if not done
        if not done:
            print("\n➡️ Continue making moves...")
            
            while not done:
                current_state = self._get_state()
                print("\n📊 Current Efficiency Levels:")
                for sys_name, val in current_state['efficiency'].items():
                    target = self.target_efficiency[sys_name]
                    status = "✅" if val >= target else "❌"
                    print(f"  {sys_name}: {val:.2f} / {target} {status}")
                
                print(f"\n⚡ Energy remaining: {current_state['time_remaining']}")
                
                # Check if all targets are already met
                if self._check_target_efficiency_met():
                    print("🎯 All target efficiencies already met! You can continue or finish here.")
                
                # User inputs what system to allocate energy to
                system = input("\nWhich system do you want to allocate energy to? ").strip()
                if system not in self.systems:
                    print(f"⚠️ Invalid system. Available: {list(self.systems.keys())}")
                    continue
                    
                try:
                    effort = int(input("How many units of energy to allocate? ").strip())
                    if effort <= 0 or effort > current_state['time_remaining']:
                        print(f"⚠️ Invalid effort. Must be between 1 and {current_state['time_remaining']}")
                        continue
                except ValueError:
                    print("⚠️ Invalid input. Please enter a number.")
                    continue
                
                # Add new input to our list
                self.user_inputs.append({
                    'system': system,
                    'effort': effort,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Step the environment with user input
                state, reward, done = self.allocate(system, effort)
                print(f"✅ Updated efficiency. Reward: {reward:.2f}")
                
                if done:
                    if self._check_target_efficiency_met():
                        print("\n🎯 All target efficiencies met! Excellent work!")
                        if state['time_remaining'] > 0:
                            print(f"💡 You saved {state['time_remaining']} units of energy!")
                    else:
                        print("\n⚡ Energy exhausted!")
                    
                    print("\n📊 Final Efficiency Levels:")
                    for sys_name, val in state['efficiency'].items():
                        target = self.target_efficiency[sys_name]
                        status = "✅" if val >= target else "❌"
                        print(f"  {sys_name}: {val:.2f} / {target} {status}")
                    break
                
                # Get recommendation for next system allocation
                try:
                    next_action = self.get_recommendation()
                    print(f"\n👉 Recommended: {next_action[1]} units to {next_action[0]}")
                except Exception as e:
                    print(f"\n⚠️ Could not get recommendation: {e}")

        # Save updated user inputs
        try:
            with open(user_inputs_file, 'wb') as f:
                pickle.dump(self.user_inputs, f)
            print(f"\n💾 Session saved to: {user_inputs_file}")
        except Exception as e:
            print(f"\n❌ Failed to save session: {e}")

        # Ask if user wants to start a new cycle
        if done:
            new_cycle = input("\n🔄 Start a new cycle? (y/n): ").strip().lower()
            if new_cycle == 'y':
                # Ask if they want to preserve efficiency levels
                preserve = input("🔧 Keep current efficiency levels for the new cycle? (y/n): ").strip().lower()
                preserve_efficiency = preserve == 'y'
                
                if not preserve_efficiency:
                    # Clear previous inputs for fresh start
                    self.user_inputs = []
                    try:
                        os.remove(user_inputs_file)
                    except:
                        pass
                else:
                    # Keep some context but start new cycle
                    print("📊 Starting new cycle with preserved efficiency levels")
                    self.get_recommendation(state)
                
                # Start fresh session with preserved efficiency option
                self.start_interactive_session(user_inputs_file, preserve_efficiency)

        return self.user_inputs

    def get_training_stats(self):
        """Get statistics about the training process."""
        if not self.training_history:
            return "No training history available."
        
        stats = {
            'total_episodes': len(self.training_history),
            'average_reward': np.mean(self.training_history),
            'best_reward': max(self.training_history),
            'worst_reward': min(self.training_history),
            'recent_performance': np.mean(self.training_history[-50:]) if len(self.training_history) >= 50 else np.mean(self.training_history),
            'q_table_size': len(self.Q)
        }
        return stats

    # Legacy methods for backward compatibility
    def save_q_table(self, filename='q_table.pkl'):
        """Legacy method - use save_model() instead"""
        print("⚠️ save_q_table() is deprecated. Use save_model() instead.")
        self.save_model(filename.replace('.pkl', ''))

    def load_q_table(self, filename='q_table.pkl'):
        """Legacy method - use load_model() instead"""
        print("⚠️ load_q_table() is deprecated. Use load_model() instead.")
        return self.load_model(filename)