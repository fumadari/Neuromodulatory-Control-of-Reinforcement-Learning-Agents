import torch
import numpy as np
import random
import os
import json # For saving configs if needed

# Import the trainer
from trainer import A2CTrainer 

# --- Global Configuration & Seeding ---
def set_global_seeds(seed):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Potentially set environment seed if the env supports it directly in reset
    # and if gymnasium makes use of it for its internal randomness.
    # os.environ['PYTHONHASHSEED'] = str(seed) # Less common but sometimes used

def get_base_configurations():
    """Returns base configurations for environment, agent, and training."""

    env_config = {
        "grid_size": (15, 17),          # Odd numbers for maze generation
        "num_ghosts": 2,
        "num_pellets_target_ratio": 0.3, # Smaller ratio for sparser pellets, more challenge
        "num_power_pellets": 2,
        "egocentric_view_size": 7,      # Must be odd
        "max_steps_episode": 400,
        "ghost_scatter_time": 60,
        "ghost_chase_time": 180,
        "power_pellet_duration": 50,
        "reward_pellet": 10,
        "reward_power_pellet": 30,      # Lowered to make ghost eating more valuable
        "reward_eat_ghost": 150,
        "penalty_ghost_collision": -300,
        "penalty_step": -0.5,           # Slightly higher step penalty
        "penalty_wall_bump": -2,
        "reward_clear_level": 500
    }

    agent_config = {
        "use_residual": True,
        "num_res_blocks": 1 # Keep it relatively simple for faster training initially
    }

    training_config = {
        "learning_rate": 7e-4, # Common starting point for Adam with A2C
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "num_steps_rollout": 256, # How many steps to collect before an update
        "value_loss_coef": 0.5,
        "entropy_coef_baseline": 0.01, # Baseline exploration drive
        "max_grad_norm": 0.5,
        "adam_eps": 1e-5,
        "use_gpu": True, # Set to False if you don't have a capable GPU
        "log_dir_base": "neuromod_experiments_results", # Base directory for all experiments
        "print_freq_updates": 20,  # Print progress every X updates
        "save_freq_updates": 200 # Save model every X updates
    }
    return env_config, agent_config, training_config

def run_single_experiment(experiment_name, total_timesteps, seed,
                          base_env_config, base_agent_config, base_training_config,
                          neuromod_override_config):
    """
    Runs a single experiment with a specific neuromodulator configuration.
    """
    print(f"\n{'='*20} Starting Experiment: {experiment_name} (Seed: {seed}) {'='*20}")
    
    set_global_seeds(seed)

    # Create experiment-specific neuromodulator config
    current_neuromod_config = {
        "k_DA": neuromod_override_config.get("k_DA", 1.0),
        "k_5HT_entropy": neuromod_override_config.get("k_5HT_entropy", 0.0),
        "k_5HT_risk": neuromod_override_config.get("k_5HT_risk", 0.0)
    }

    # Instantiate and run the trainer
    trainer = A2CTrainer(
        env_config=base_env_config,
        agent_config=base_agent_config,
        training_config=base_training_config,
        neuromod_config=current_neuromod_config,
        experiment_name=f"{experiment_name}_seed{seed}" # Ensure unique log dirs per seed
    )
    
    # Save the exact configs used for this run
    run_config_dir = os.path.join(base_training_config["log_dir_base"], f"{experiment_name}_seed{seed}")
    os.makedirs(run_config_dir, exist_ok=True)
    with open(os.path.join(run_config_dir, "env_config.json"), "w") as f: json.dump(base_env_config, f, indent=4)
    with open(os.path.join(run_config_dir, "agent_config.json"), "w") as f: json.dump(base_agent_config, f, indent=4)
    with open(os.path.join(run_config_dir, "training_config.json"), "w") as f: json.dump(base_training_config, f, indent=4)
    with open(os.path.join(run_config_dir, "neuromod_config.json"), "w") as f: json.dump(current_neuromod_config, f, indent=4)

    try:
        trainer.train(total_timesteps=total_timesteps)
    except Exception as e:
        print(f"!!!!!! ERROR during training for {experiment_name}_seed{seed} !!!!!!")
        print(e)
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(trainer, 'writer') and trainer.writer:
             trainer.writer.close() # Ensure writer is closed even on error
        if hasattr(trainer, 'env') and trainer.env:
             trainer.env.close()
             
    print(f"{'='*20} Finished Experiment: {experiment_name} (Seed: {seed}) {'='*20}\n")


if __name__ == "__main__":
    TOTAL_TIMESTEPS_PER_RUN = 1_000_000 # Adjust as needed (e.g., 1M, 5M, 10M). Start small!
    # TOTAL_TIMESTEPS_PER_RUN = 50_000 # For quick test

    SEEDS_PER_EXPERIMENT = [42, 123, 789] # Run each condition with multiple seeds for robustness

    # --- Get Base Configurations ---
    base_env_config, base_agent_config, base_training_config = get_base_configurations()

    # --- Define Experimental Conditions (Neuromodulator Settings) ---
    # Each dictionary defines the k_DA, k_5HT_entropy, and k_5HT_risk for one condition.
    # Default values (1.0, 0.0, 0.0) will be used if a key is not present.
    
    experimental_conditions = [
        {
            "name": "Baseline", 
            "neuromod_params": {"k_DA": 1.0, "k_5HT_entropy": 0.0, "k_5HT_risk": 0.0} 
            # Here, k_5HT_entropy=0 means only the baseline_entropy_coef from training_config applies.
            # k_5HT_risk=0 means no additional risk penalty.
        },
        {
            "name": "High_DA",
            "neuromod_params": {"k_DA": 2.5, "k_5HT_entropy": 0.0, "k_5HT_risk": 0.0} 
            # k_DA=2.5 might be a strong boost. Values between 1.5-3.0 are worth exploring.
        },
        {
            "name": "High_5HT_Entropy",
            "neuromod_params": {"k_DA": 1.0, "k_5HT_entropy": 0.02, "k_5HT_risk": 0.0} 
            # This ADDS to the baseline_entropy_coef. If baseline is 0.01, total is 0.03.
        },
        {
            "name": "High_5HT_Risk",
            "neuromod_params": {"k_DA": 1.0, "k_5HT_entropy": 0.0, "k_5HT_risk": 0.5} 
            # k_5HT_risk=0.5 means RiskSignal is multiplied by 0.5 and subtracted from reward.
            # The impact depends on magnitude of RiskSignal and raw rewards.
        },
        {
            "name": "Cautious_Explorer_High_5HT_Both",
            "neuromod_params": {"k_DA": 1.0, "k_5HT_entropy": 0.02, "k_5HT_risk": 0.5}
        },
        {
            "name": "High_DA_High_5HT_Risk_Conflict", # The "conflict" condition
            "neuromod_params": {"k_DA": 2.5, "k_5HT_entropy": 0.0, "k_5HT_risk": 0.75}
        },
        # --- Add more conditions as needed ---
        # Example: Low DA
        # {
        #     "name": "Low_DA",
        #     "neuromod_params": {"k_DA": 0.5, "k_5HT_entropy": 0.0, "k_5HT_risk": 0.0}
        # },
        # Example: Very High Risk Aversion
        # {
        #     "name": "Extreme_Risk_Aversion",
        #     "neuromod_params": {"k_DA": 1.0, "k_5HT_entropy": 0.01, "k_5HT_risk": 1.5}
        # },
    ]

    # --- Run All Experiments ---
    for condition in experimental_conditions:
        for seed_val in SEEDS_PER_EXPERIMENT:
            run_single_experiment(
                experiment_name=condition["name"],
                total_timesteps=TOTAL_TIMESTEPS_PER_RUN,
                seed=seed_val,
                base_env_config=base_env_config.copy(), # Pass copies to avoid modification
                base_agent_config=base_agent_config.copy(),
                base_training_config=base_training_config.copy(),
                neuromod_override_config=condition["neuromod_params"]
            )
    
    print("\nAll defined experiments have been run.")
    print(f"Results, logs, and models saved in: {base_training_config['log_dir_base']}")
