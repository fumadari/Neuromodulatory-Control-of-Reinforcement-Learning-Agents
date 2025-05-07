# Neuromodulatory-Control-of-Reinforcement-Learning-Agents


This project implements and simulates a Reinforcement Learning (RL) agent whose learning and behavior are dynamically influenced by computational analogs of the neuromodulators Dopamine (DA) and Serotonin (5-HT). The agent operates within a custom "Pac-Man"-like grid-world environment.

## Project Goal

The primary scientific goal is to investigate how varying simulated levels of DA and 5-HT affect an RL agent's:
1.  **Learning Dynamics:** How quickly and effectively it learns to perform the task.
2.  **Behavioral Strategies:** Its approach to exploration vs. exploitation, risk-taking vs. caution, and overall decision-making style.
3.  **Task Performance:** Its ability to maximize rewards and achieve objectives within the environment.

This project aims to provide a computational model that bridges concepts from neuroscience (neuromodulation) and artificial intelligence (reinforcement learning) to create agents with more nuanced, adaptive, and biologically plausible behaviors.

## Core Concepts Modeled

*   **Dopamine (DA):** Modeled as a modulator of the **Reward Prediction Error (RPE)** signal. Higher DA levels (`k_DA`) amplify the impact of RPEs on both value learning (critic) and policy updates (actor). This is expected to influence motivation, learning speed from rewards, and potentially reward-driven impulsivity.
*   **Serotonin (5-HT):** Modeled through two primary mechanisms:
    1.  **Policy Entropy (`k_5HT_entropy`):** Higher 5-HT levels increase a bonus for policy stochasticity (entropy), encouraging more exploratory or less deterministic actions. This relates to 5-HT's role in behavioral variability and potentially cautious exploration.
    2.  **Risk Sensitivity (`k_5HT_risk`):** Higher 5-HT levels increase the agent's aversion to risky situations. This is implemented by penalizing the perceived reward based on a `RiskSignal` from the environment (e.g., proximity to threats). This reflects 5-HT's association with harm avoidance and responses to aversive stimuli.

The agent uses an **Actor-Critic (A2C)** architecture with **Generalized Advantage Estimation (GAE)**.

## Project Structure

The project consists of the following Python files:

1.  **`pacman_env.py`:**
    *   Defines the custom "Pac-Man"-like environment (`PacManEnvEnhanced`).
    *   Handles game logic: maze generation (Randomized DFS), agent movement, pellet/power pellet collection, ghost AI (chase/scatter/vulnerable modes), collision detection, and reward calculation.
    *   Provides observations to the agent (an egocentric grid view) and an `info` dictionary containing game state details, including a `risk_signal`.

2.  **`agent_model.py`:**
    *   Defines the neural network architecture for the Actor-Critic agent (`SophisticatedActorCriticNetwork`).
    *   Uses a Convolutional Neural Network (CNN) frontend (with optional residual blocks) to process the egocentric view.
    *   Has separate output heads for the Actor (policy logits) and the Critic (state value).
    *   Includes methods for action selection (stochastic or deterministic) and calculating log probabilities and policy entropy.

3.  **`trainer.py`:**
    *   Contains the `A2CTrainer` class, which orchestrates the entire training process.
    *   Implements the A2C learning algorithm with GAE.
    *   **Crucially, integrates the neuromodulatory controls:**
        *   Applies `k_DA` to scale advantages.
        *   Modifies rewards based on `k_5HT_risk` and the environment's `risk_signal`.
        *   Adds `k_5HT_entropy` to the baseline entropy bonus in the actor's loss function.
    *   Manages rollout collection, batch updates, optimization, logging (to TensorBoard and CSV), and model saving/loading.

4.  **`run_experiment.py`:**
    *   The main script to launch and manage different experimental conditions.
    *   Defines base configurations for the environment, agent, and training.
    *   Allows specification of various experimental conditions by overriding neuromodulator parameters (`k_DA`, `k_5HT_entropy`, `k_5HT_risk`).
    *   Handles running experiments with multiple random seeds for robust analysis.
    *   Organizes output logs and saved models into uniquely named directories.

5.  **`README.md` (this file):**
    *   Provides an overview of the project.

## Requirements

*   Python 3.8+
*   PyTorch (`torch`, `torchvision`, `torchaudio`)
*   NumPy
*   Gymnasium (`gymnasium`) - Note: The environment class inherits from `gym.Env` for broader compatibility but uses Gymnasium's reset/step API style.
*   Pandas (for saving CSV data)
*   Matplotlib / Seaborn (optional, for your own offline plotting of results)
*   TensorBoard (for visualizing training metrics, installed with TensorFlow or separately: `pip install tensorboard`)

Install dependencies (primarily PyTorch, NumPy, Gymnasium, Pandas):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Adjust for your CUDA/CPU
pip install numpy gymnasium pandas matplotlib seaborn tensorboard


How to Run Experiments
Configure Experiments:
Open run_experiment.py.
Adjust TOTAL_TIMESTEPS_PER_RUN to set the duration of each training run. Start with a small value (e.g., 50,000) for initial testing, then increase for full experiments (e.g., 1,000,000+).
Modify the experimental_conditions list to define the different neuromodulator settings you want to test. Each entry should have a unique name and a neuromod_params dictionary specifying k_DA, k_5HT_entropy, and k_5HT_risk.
Adjust SEEDS_PER_EXPERIMENT if you want to run more or fewer random seeds per condition.
Review get_base_configurations() to change default environment, agent, or training parameters if needed. Pay attention to training_config["use_gpu"].
Execute the Script:
Navigate to the project's root directory in your terminal.
Run the command:
python run_experiment.py

Monitor Training (Optional):
While experiments are running (or after), you can launch TensorBoard to view live metrics:
tensorboard --logdir neuromod_experiments_results

(Assuming neuromod_experiments_results is your log_dir_base from training_config).
Open your browser and go to http://localhost:6006 (or the port TensorBoard indicates).
Expected Output and Results
Console Output: Progress messages, including current timestep, updates, episode count, average rewards, and losses.
Log Directory: A main directory (e.g., neuromod_experiments_results/) will be created.
Inside, subdirectories for each experimental condition and seed will be generated (e.g., High_DA_seed42/).
Each of these subdirectories will contain:
tensorboard/: TensorBoard log files.
models/: Saved PyTorch model checkpoints (.pth files).
episode_data_[experiment_name]_seed[seed].csv: A CSV file containing per-episode statistics (score, length, reward sum, neuromodulator settings, specific environment stats). This is the primary data for your analysis and plotting.
*.json files: Exact copies of the env_config, agent_config, training_config, and neuromod_config used for that specific run, ensuring full traceability of parameters.
Analyzing Results
CSV Data: Use the episode_data_*.csv files. Load them into Python with Pandas, R, Excel, or any other data analysis tool.
Key Plots to Generate:
Learning Curves: Plot average reward per episode (or score) against total timesteps or episodes. Smooth these curves (e.g., using a rolling mean) and average across different random seeds for each experimental condition. Compare these averaged curves between conditions.
Performance Comparison: Bar charts or box plots comparing final performance (e.g., average score over the last X% of training, total pellets collected, number of collisions) across the different neuromodulatory conditions.
Behavioral Metrics Analysis: Plots comparing average policy entropy, average experienced risk signal, average distance to ghosts, etc., across conditions.
TensorBoard: Use TensorBoard to inspect losses, policy entropy, value predictions, and other metrics logged during training to understand the learning dynamics.
Further Development and Experimentation Ideas
Implement more sophisticated ghost AI.
Allow for dynamic (endogenous) control of neuromodulator levels by the agent itself.
Explore interactions with other neuromodulators (e.g., norepinephrine, acetylcholine).
Test on a wider range of maze configurations or more complex environments.
Conduct detailed hyperparameter sweeps for training and neuromodulator parameters.
Implement alternative RL algorithms (e.g., PPO, SAC) with these neuromodulatory controls.
