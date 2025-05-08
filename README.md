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


