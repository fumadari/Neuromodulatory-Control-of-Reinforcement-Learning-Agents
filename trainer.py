import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter # For optional TensorBoard logging
import numpy as np
import pandas as pd
import os
import time
from collections import deque

# Assuming agent_model.py and pacman_env.py are in the same directory or accessible
# from agent_model import ActorCriticNetwork # Or SophisticatedActorCriticNetwork
# from pacman_env import PacManEnv # Or PacManEnvEnhanced

# To make it runnable, let's use the sophisticated versions:
from agent_model import SophisticatedActorCriticNetwork as ActorCriticNetwork
from pacman_env import PacManEnvEnhanced as PacManEnv


class A2CTrainer:
    def __init__(self, env_config, agent_config, training_config, neuromod_config, experiment_name="default_exp"):
        """
        Trainer for the Actor-Critic agent with neuromodulatory controls.

        Args:
            env_config (dict): Configuration for the environment.
            agent_config (dict): Configuration for the agent model.
            training_config (dict): Configuration for training (learning rates, gamma, etc.).
            neuromod_config (dict): Configuration for neuromodulator levels (k_DA, k_5HT_entropy, k_5HT_risk).
            experiment_name (str): Name for this experiment run, used for saving logs/models.
        """
        self.env_config = env_config
        self.agent_config = agent_config
        self.training_config = training_config
        self.neuromod_config = neuromod_config
        self.experiment_name = experiment_name

        # --- Setup Device ---
        self.device = torch.device("cuda" if torch.cuda.is_available() and training_config.get("use_gpu", True) else "cpu")
        print(f"Using device: {self.device}")

        # --- Initialize Environment ---
        self.env = PacManEnv(**env_config)
        
        obs_space_shape = self.env.observation_space.shape
        # obs_space_shape is (channels, height, width)
        # For the network, input_channels = obs_space_shape[0], egocentric_view_size = obs_space_shape[1] (assuming square)
        
        # --- Initialize Agent ---
        self.agent = ActorCriticNetwork(
            input_channels=obs_space_shape[0],
            egocentric_view_size=obs_space_shape[1],
            num_actions=self.env.action_space.n,
            use_residual=agent_config.get("use_residual", True),
            num_res_blocks=agent_config.get("num_res_blocks", 1)
        ).to(self.device)

        # --- Initialize Optimizer ---
        self.optimizer = optim.Adam(
            self.agent.parameters(), 
            lr=training_config.get("learning_rate", 5e-4),
            eps=training_config.get("adam_eps", 1e-5) # Epsilon for numerical stability
        )

        # --- Training Hyperparameters ---
        self.gamma = training_config.get("gamma", 0.99) # Discount factor
        self.gae_lambda = training_config.get("gae_lambda", 0.95) # Lambda for Generalized Advantage Estimation
        self.num_steps_rollout = training_config.get("num_steps_rollout", 128) # N-step rollouts
        self.num_epochs_ppo = training_config.get("num_epochs_ppo", 4) # For PPO-style updates (if using, not fully PPO here)
        self.clip_param_ppo = training_config.get("clip_param_ppo", 0.2) # For PPO clipping
        
        self.value_loss_coef = training_config.get("value_loss_coef", 0.5)
        self.entropy_coef_baseline = training_config.get("entropy_coef_baseline", 0.01) # Baseline entropy for exploration

        self.max_grad_norm = training_config.get("max_grad_norm", 0.5)

        # --- Neuromodulator Parameters ---
        self.k_DA = neuromod_config.get("k_DA", 1.0)
        self.k_5HT_entropy = neuromod_config.get("k_5HT_entropy", 0.0) # Additional entropy beyond baseline
        self.k_5HT_risk = neuromod_config.get("k_5HT_risk", 0.0)

        # --- Logging and Saving ---
        self.log_dir = os.path.join(training_config.get("log_dir_base", "experiment_logs"), experiment_name)
        self.model_save_dir = os.path.join(self.log_dir, "models")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "tensorboard"))
        self.episode_data = [] # To store data for saving to CSV
        self.reward_history = deque(maxlen=100) # For tracking moving average of rewards

        print(f"Experiment '{experiment_name}' initialized.")
        print(f"  Neuromodulators: k_DA={self.k_DA}, k_5HT_entropy={self.k_5HT_entropy}, k_5HT_risk={self.k_5HT_risk}")


    def _compute_advantages_gae(self, rewards, values, dones, next_value):
        """
        Computes Generalized Advantage Estimation (GAE).
        Args:
            rewards (torch.Tensor): Tensor of rewards from rollout. Shape (num_steps_rollout, num_envs).
            values (torch.Tensor): Tensor of state values from rollout. Shape (num_steps_rollout + 1, num_envs).
                                   Includes value of the state *after* the last step.
            dones (torch.Tensor): Tensor of done flags (0 or 1). Shape (num_steps_rollout, num_envs).
            next_value (torch.Tensor): Value of the state after the last rollout step. Shape (num_envs, 1).
        Returns:
            advantages (torch.Tensor): Computed advantages. Shape (num_steps_rollout, num_envs).
            returns (torch.Tensor): Computed N-step returns (targets for value function). Shape (num_steps_rollout, num_envs)
        """
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = 0
        
        # Ensure values include the next_value for the last step's return calculation
        # values should be (num_steps_rollout + 1, num_envs) if next_value is the last element
        # Here, we explicitly use next_value for the step after the last collected one.
        
        num_steps = rewards.size(0) # Should be self.num_steps_rollout
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                # If it's the last step of the rollout, the "next_done" is effectively True if the episode ended,
                # and the "next_value" is the bootstrapped value of S_{t+1}
                next_non_terminal = 1.0 - dones[t] # if done[t] is for S_t -> A_t -> S_{t+1}, this is correct.
                                                    # if dones[t] means S_t was terminal, then this is slightly off.
                                                    # Assuming dones[t] is True if S_{t+1} is terminal.
                value_next_step = next_value.squeeze(-1) # remove last dim if (num_envs, 1)
            else:
                next_non_terminal = 1.0 - dones[t+1] # Done flag for the actual next state in buffer
                value_next_step = values[t+1]

            # TD error for GAE: δ_t = R_t + γ * V(S_{t+1}) * (1-done_{t+1}) - V(S_t)
            # Here, dones[t] corresponds to the transition S_t -> A_t -> S_{t+1}
            # So, if S_{t+1} is terminal, (1-dones[t]) will be 0.
            delta = rewards[t] + self.gamma * value_next_step * (1.0 - dones[t]) - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae_lam
        
        returns = advantages + values[:num_steps] # Returns for value function targets: Q_t = A_t + V(S_t)
        return advantages, returns

    def train(self, total_timesteps):
        start_time = time.time()
        
        # Initialize environment state
        # Assuming single environment for now, can be extended to parallel envs
        current_obs_np, _ = self.env.reset()
        current_obs = torch.from_numpy(current_obs_np).float().unsqueeze(0).to(self.device) # Add batch dim
        current_done = torch.zeros(1, 1).to(self.device) # Batch dim for done

        num_episodes = 0
        total_updates = 0

        # --- Storage for Rollout Data ---
        # For a single environment
        obs_buffer = torch.zeros((self.num_steps_rollout + 1, 1) + self.env.observation_space.shape).to(self.device)
        reward_buffer = torch.zeros(self.num_steps_rollout, 1, 1).to(self.device)
        action_buffer = torch.zeros(self.num_steps_rollout, 1, 1).long().to(self.device) # Store action index
        log_prob_buffer = torch.zeros(self.num_steps_rollout, 1, 1).to(self.device)
        value_buffer = torch.zeros(self.num_steps_rollout + 1, 1, 1).to(self.device) # +1 for V(S_N)
        done_buffer = torch.zeros(self.num_steps_rollout + 1, 1, 1).to(self.device) # +1 for V(S_N)
        entropy_buffer = torch.zeros(self.num_steps_rollout, 1, 1).to(self.device)
        
        # Buffers for neuromodulation-relevant info from env
        risk_signal_buffer = torch.zeros(self.num_steps_rollout, 1, 1).to(self.device)

        obs_buffer[0] = current_obs # Store initial observation

        for timestep in range(0, total_timesteps, self.num_steps_rollout):
            self.agent.train() # Set model to training mode
            
            # --- Collect Rollout (N steps) ---
            episode_rewards_in_rollout = []
            episode_lengths_in_rollout = []
            current_episode_reward = 0
            current_episode_length = 0
            
            # Temporary storage for full episode stats if an episode ends mid-rollout
            completed_episode_stats = []


            for step in range(self.num_steps_rollout):
                with torch.no_grad(): # No gradient calculation during action selection
                    action, log_prob, entropy, value_pred = self.agent.get_action_and_value(current_obs)
                
                obs_np_next, reward_raw, terminated, truncated, info = self.env.step(action.cpu().item())
                done = terminated or truncated
                
                current_episode_reward += reward_raw
                current_episode_length += 1

                # Store data
                obs_buffer[step+1] = torch.from_numpy(obs_np_next).float().unsqueeze(0).to(self.device)
                reward_buffer[step] = torch.tensor([[reward_raw]], dtype=torch.float32).to(self.device)
                action_buffer[step] = action.unsqueeze(-1) # Ensure it's (1,1)
                log_prob_buffer[step] = log_prob.unsqueeze(-1)
                value_buffer[step] = value_pred
                done_buffer[step+1] = torch.tensor([[done*1.0]], dtype=torch.float32).to(self.device) # done for S_{t+1}
                entropy_buffer[step] = entropy.unsqueeze(-1)
                risk_signal_buffer[step] = torch.tensor([[info.get("risk_signal", 0.0)]], dtype=torch.float32).to(self.device)

                current_obs = obs_buffer[step+1] # Update current_obs for next iteration

                if done:
                    num_episodes += 1
                    episode_rewards_in_rollout.append(current_episode_reward)
                    self.reward_history.append(current_episode_reward) # For moving average
                    episode_lengths_in_rollout.append(current_episode_length)
                    
                    # Log full episode data for CSV
                    ep_data_point = {
                        "episode": num_episodes,
                        "total_timesteps": timestep + step + 1,
                        "score": info["score"],
                        "length": current_episode_length,
                        "reward_sum": current_episode_reward,
                        "k_DA": self.k_DA, "k_5HT_entropy": self.k_5HT_entropy, "k_5HT_risk": self.k_5HT_risk,
                        "avg_risk_signal_episode": -1, # Placeholder, can compute if storing all risk signals for ep
                        **{f"env_{k}": v for k,v in info.items() if "stats_" in k or k in ["min_dist_to_normal_ghost"]}
                    }
                    completed_episode_stats.append(ep_data_point)
                    self.episode_data.append(ep_data_point) # Append to main list for CSV
                    
                    self.writer.add_scalar("Rollout/Episode_Reward", current_episode_reward, num_episodes)
                    self.writer.add_scalar("Rollout/Episode_Length", current_episode_length, num_episodes)
                    self.writer.add_scalar("Rollout/Episode_Score", info["score"], num_episodes)
                    if len(self.reward_history) > 0:
                        self.writer.add_scalar("Rollout/Moving_Avg_Reward_100", np.mean(self.reward_history), num_episodes)

                    # Reset for next episode within rollout
                    current_obs_np, _ = self.env.reset()
                    current_obs = torch.from_numpy(current_obs_np).float().unsqueeze(0).to(self.device)
                    obs_buffer[step+1] = current_obs # Overwrite with reset state if episode ended
                    current_episode_reward = 0
                    current_episode_length = 0
            
            # Get value of the state after the last rollout step (S_N)
            with torch.no_grad():
                _, _, _, next_value_for_gae = self.agent.get_action_and_value(current_obs) # V(S_N)
            value_buffer[self.num_steps_rollout] = next_value_for_gae # Store V(S_N)
            
            # --- Modify Rewards based on Serotonergic Risk Modulation ---
            # R'_t = R_t - k_5HT_risk * RiskSignal(S_{t+1})
            # RiskSignal is associated with S_{t+1}, which is obs_buffer[t+1]
            # risk_signal_buffer stores RiskSignal(S_t) if info comes from env.step(A_{t-1}) -> S_t
            # Let's assume risk_signal_buffer[t] is RiskSignal(S_t) from previous step,
            # so we need RiskSignal for S_{t+1}.
            # For simplicity, we use risk_signal_buffer[step] as RiskSignal(S_t) for reward R_t at step `step`
            # (which is reward for A_{t-1} leading to S_t)
            # This implies RiskSignal should be for the *resulting state* of the action.
            # Our current `info['risk_signal']` in env is for the *current* agent_pos *before* ghost move.
            # Let's adjust: the risk signal from info after step() is for S_{t+1}
            
            # risk_signal_buffer[t] is RiskSignal(S_{t+1}) from taking action A_t at S_t
            modified_rewards = reward_buffer - self.k_5HT_risk * risk_signal_buffer # Element-wise
            
            # --- Compute Advantages and Returns using GAE ---
            # Reshape buffers for GAE: (num_steps, num_envs=1, feature_dim=1) -> (num_steps, num_envs)
            advantages, returns = self._compute_advantages_gae(
                modified_rewards.squeeze(-1),  # (num_steps, 1)
                value_buffer[:self.num_steps_rollout].squeeze(-1), # (num_steps, 1) V(S_0) to V(S_{N-1})
                done_buffer[1:].squeeze(-1), # (num_steps, 1) done flags for S_1 to S_N
                next_value_for_gae.squeeze(-1) # (1,) V(S_N)
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Normalize advantages

            # --- Perform Updates (Actor and Critic) ---
            # Flatten rollout data for batched updates if using minibatches (not implemented here for simplicity of A2C)
            # For A2C, typically update on the full rollout.
            
            # Get action_logits, values, and entropy for all states in the rollout
            # obs_buffer[:-1] gives S_0 to S_{N-1}
            all_action_logits, all_values_pred = self.agent(obs_buffer[:-1].squeeze(1)) # Squeeze out num_envs dim
            
            # Recompute log_probs and entropy based on current policy for importance sampling (more PPO like)
            # For simpler A2C, use stored log_probs. Here, we use stored.
            dist = torch.distributions.Categorical(logits=all_action_logits)
            new_log_probs = dist.log_prob(action_buffer.squeeze()) # Squeeze action_buffer
            entropy_loss_terms = dist.entropy() # This is H(π(·|S_t)) for each S_t in rollout

            # Dopaminergic Modulation on Advantage
            # A'_t = k_DA * A_t
            modulated_advantages = self.k_DA * advantages
            
            # Policy Loss (Actor Loss)
            # L_policy = - E_t [ A'_t * log π(A_t|S_t) ]
            policy_loss = -(modulated_advantages * log_prob_buffer.squeeze()).mean()
            
            # Value Loss (Critic Loss)
            # L_value = E_t [ (V_target_t - V_pred_t)^2 ]
            value_loss = F.mse_loss(all_values_pred.squeeze(), returns)
            
            # Entropy Loss (for encouraging exploration)
            # L_entropy = - E_t [ H(π(·|S_t)) ]
            # Total entropy bonus = baseline_entropy_coef + k_5HT_entropy
            total_entropy_bonus_coef = self.entropy_coef_baseline + self.k_5HT_entropy
            entropy_loss = -entropy_loss_terms.mean() # Maximize entropy, so minimize negative entropy
            
            # Total Loss
            total_loss = policy_loss + \
                         self.value_loss_coef * value_loss + \
                         total_entropy_bonus_coef * entropy_loss # Entropy bonus, so subtract if entropy_loss is -H

            # Optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total_updates += 1

            # --- Logging for this update batch ---
            current_global_step = timestep + self.num_steps_rollout
            self.writer.add_scalar("Loss/Total_Loss", total_loss.item(), current_global_step)
            self.writer.add_scalar("Loss/Policy_Loss", policy_loss.item(), current_global_step)
            self.writer.add_scalar("Loss/Value_Loss", value_loss.item(), current_global_step)
            self.writer.add_scalar("Loss/Entropy_Term", (total_entropy_bonus_coef * entropy_loss).item(), current_global_step)
            self.writer.add_scalar("Policy/Avg_Entropy", entropy_loss_terms.mean().item(), current_global_step)
            self.writer.add_scalar("Policy/Avg_Advantage_Modulated", modulated_advantages.mean().item(), current_global_step)
            self.writer.add_scalar("Policy/Avg_Value_Target_Return", returns.mean().item(), current_global_step)
            self.writer.add_scalar("Policy/Avg_Value_Predicted", all_values_pred.mean().item(), current_global_step)
            self.writer.add_scalar("Neuromod/k_DA", self.k_DA, current_global_step)
            self.writer.add_scalar("Neuromod/k_5HT_entropy", self.k_5HT_entropy, current_global_step)
            self.writer.add_scalar("Neuromod/k_5HT_risk", self.k_5HT_risk, current_global_step)
            self.writer.add_scalar("Neuromod/Avg_Raw_Reward_in_Rollout", reward_buffer.mean().item(), current_global_step)
            self.writer.add_scalar("Neuromod/Avg_RiskSignal_in_Rollout", risk_signal_buffer.mean().item(), current_global_step)
            self.writer.add_scalar("Neuromod/Avg_Modified_Reward_in_Rollout", modified_rewards.mean().item(), current_global_step)


            # Prepare for next rollout: set initial observation from the last step of current rollout
            obs_buffer[0].copy_(obs_buffer[-1]) # S_N becomes S_0 for next rollout
            done_buffer[0].copy_(done_buffer[-1])
            # value_buffer[0] is not needed as it's V(S_0) which will be recomputed

            # Print progress
            if total_updates % self.training_config.get("print_freq_updates", 10) == 0:
                elapsed_time = time.time() - start_time
                fps = int(current_global_step / elapsed_time) if elapsed_time > 0 else 0
                avg_reward_100 = np.mean(self.reward_history) if len(self.reward_history) > 0 else float('nan')
                print(f"Timesteps: {current_global_step}/{total_timesteps} | Updates: {total_updates} | Episodes: {num_episodes} | "
                      f"Avg Reward (last 100): {avg_reward_100:.2f} | FPS: {fps} | "
                      f"PL: {policy_loss.item():.3f} | VL: {value_loss.item():.3f} | EL: {entropy_loss.item():.3f}")

            # Save model periodically
            if total_updates % self.training_config.get("save_freq_updates", 500) == 0:
                self.save_model(total_updates)
        
        # Final save and cleanup
        self.save_model("final")
        self.save_episode_data_to_csv()
        self.writer.close()
        self.env.close()
        print(f"Training finished for {self.experiment_name}.")

    def save_model(self, checkpoint_name="latest"):
        model_path = os.path.join(self.model_save_dir, f"agent_model_{checkpoint_name}.pth")
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # Add any other things you want to save, like training step or neuromod_config
            'neuromod_config': self.neuromod_config,
            'training_config': self.training_config # To know parameters used
        }, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            print(f"Model path {model_path} does not exist. Cannot load.")
            return False
        checkpoint = torch.load(model_path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Optional: load optimizer state
        # self.neuromod_config = checkpoint.get('neuromod_config', self.neuromod_config) # Update if saved
        self.agent.to(self.device)
        self.agent.eval() # Set to evaluation mode after loading
        print(f"Model loaded from {model_path}")
        return True

    def save_episode_data_to_csv(self):
        if not self.episode_data:
            print("No episode data to save.")
            return
        df = pd.DataFrame(self.episode_data)
        csv_path = os.path.join(self.log_dir, f"episode_data_{self.experiment_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Episode data saved to {csv_path}")
