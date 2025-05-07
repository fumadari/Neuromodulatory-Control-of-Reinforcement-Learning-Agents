import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """
    A simple Residual Block with two convolutional layers.
    Assumes input and output channels are the same, and spatial dimensions are preserved.
    """
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride, padding, bias=False) # Bias often False before BN
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # Skip connection
        out = F.relu(out)
        return out

class SophisticatedActorCriticNetwork(nn.Module):
    def __init__(self, input_channels, egocentric_view_size, num_actions, use_residual=True, num_res_blocks=1):
        """
        A more sophisticated Actor-Critic Network for the Pac-Man like environment.

        Args:
            input_channels (int): Number of channels in the input observation.
            egocentric_view_size (int): Height and width of the square egocentric view.
            num_actions (int): Number of discrete actions.
            use_residual (bool): Whether to use residual blocks in the CNN.
            num_res_blocks (int): Number of residual blocks to use if use_residual is True.
        """
        super(SophisticatedActorCriticNetwork, self).__init__()
        self.input_channels = input_channels
        self.view_size = egocentric_view_size
        self.num_actions = num_actions
        self.use_residual = use_residual

        # --- Convolutional Feature Extractor ---
        # Initial convolutional layer to increase channel depth
        self.conv_initial = nn.Conv2d(in_channels=self.input_channels, 
                                      out_channels=32, 
                                      kernel_size=3, 
                                      stride=1, 
                                      padding=1, # Preserves spatial dimensions with kernel 3x3
                                      bias=False) 
        self.bn_initial = nn.BatchNorm2d(32)
        current_channels = 32

        # Optional Residual Blocks
        if self.use_residual:
            self.residual_layers = nn.ModuleList()
            for _ in range(num_res_blocks):
                self.residual_layers.append(ResidualBlock(channels=current_channels, kernel_size=3, padding=1))
        
        # Additional convolutional layer to potentially change channel depth or further extract features
        # This layer could also be responsible for spatial dimension reduction if stride > 1 or padding is different.
        self.conv_final_features = nn.Conv2d(in_channels=current_channels, 
                                             out_channels=64, 
                                             kernel_size=3, 
                                             stride=1, 
                                             padding=1, # Still preserving spatial dimensions
                                             bias=False)
        self.bn_final_features = nn.BatchNorm2d(64)
        final_conv_channels = 64

        # Calculate the flattened size after all convolutional layers
        # Assuming no change in spatial dimensions due to padding strategy (padding = kernel_size // 2 for odd kernels)
        self.conv_output_spatial_dim = self.view_size 
        self.flattened_size = final_conv_channels * self.conv_output_spatial_dim * self.conv_output_spatial_dim

        # --- Fully Connected Layers ---
        # Shared dense layer
        self.fc_shared = nn.Linear(self.flattened_size, 256)
        self.ln_shared = nn.LayerNorm(256) # LayerNorm often preferred over BatchNorm for FC layers in RL

        # Actor Head: outputs action logits
        self.actor_head = nn.Linear(256, self.num_actions)

        # Critic Head: outputs state value (a single scalar)
        self.critic_head = nn.Linear(256, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization (Kaiming init) is often good for ReLUs
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) # Xavier for layers before non-ReLU or for general purpose
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                if m.weight is not None: nn.init.constant_(m.weight, 1)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()

        # Initial convolution
        x = F.relu(self.bn_initial(self.conv_initial(x)))

        # Residual blocks
        if self.use_residual:
            for res_layer in self.residual_layers:
                x = res_layer(x)
        
        # Final feature convolution
        x = F.relu(self.bn_final_features(self.conv_final_features(x)))

        # Flatten
        x = x.view(-1, self.flattened_size)

        # Shared dense layer
        x_shared = F.relu(self.ln_shared(self.fc_shared(x)))

        # Actor and Critic heads
        action_logits = self.actor_head(x_shared)
        value = self.critic_head(x_shared)

        return action_logits, value

    def get_action_and_value(self, state, action_mask=None, deterministic=False):
        if not isinstance(state, torch.Tensor):
            # Assuming state is a numpy array, convert to tensor.
            # If using GPU, add .to(self.conv_initial.weight.device) to move data to the model's device
            device = next(self.parameters()).device # Get device from model parameters
            state = torch.from_numpy(state).float().to(device)
        
        if state.ndim == 3: # (channels, height, width) -> add batch dim
            state = state.unsqueeze(0)

        action_logits, value = self.forward(state)
        
        probs_epsilon = 1e-8 # For numerical stability with Categorical distribution
        
        if action_mask is not None:
            action_mask = action_mask.to(action_logits.device).float()
            # Use a large negative number for masked logits to ensure near-zero probability after softmax
            # Adding to logits is generally more stable than multiplying probabilities
            masked_logits = action_logits + (action_mask - 1.0) * 1e9 
            probs = F.softmax(masked_logits, dim=-1)
        else:
            probs = F.softmax(action_logits, dim=-1)

        dist = torch.distributions.Categorical(probs + probs_epsilon) # Add epsilon for stability

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value


if __name__ == '__main__':
    # Example Usage and Test
    INPUT_CHANNELS = 6
    VIEW_SIZE = 9 
    NUM_ACTIONS = 4

    # Test with residual blocks
    print("--- Testing Model with Residual Blocks ---")
    model_res = SophisticatedActorCriticNetwork(INPUT_CHANNELS, VIEW_SIZE, NUM_ACTIONS, use_residual=True, num_res_blocks=2)
    print(model_res)

    dummy_observation = np.random.rand(1, INPUT_CHANNELS, VIEW_SIZE, VIEW_SIZE).astype(np.float32)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_res.to(device) # Move model to device
    # dummy_state_tensor = torch.from_numpy(dummy_observation).to(device) # Move data to device

    # For CPU test:
    dummy_state_tensor = torch.from_numpy(dummy_observation)


    logits_res, val_res = model_res(dummy_state_tensor)
    print("\nRaw Logits (Residual Model):", logits_res)
    print("Raw Value (Residual Model):", val_res)

    action_res, log_p_res, H_res, v_res = model_res.get_action_and_value(dummy_state_tensor)
    print("\nSelected Action (Residual Model):", action_res.item())
    print("Log Prob (Residual Model):", log_p_res.item())
    print("Entropy (Residual Model):", H_res.item())
    print("Value (Residual Model):", v_res.item())

    # Test without residual blocks
    print("\n--- Testing Model without Residual Blocks ---")
    model_no_res = SophisticatedActorCriticNetwork(INPUT_CHANNELS, VIEW_SIZE, NUM_ACTIONS, use_residual=False)
    print(model_no_res)
    logits_no_res, val_no_res = model_no_res(dummy_state_tensor) # Reusing dummy_state_tensor
    print("\nRaw Logits (No Residual):", logits_no_res)
    print("Raw Value (No Residual):", val_no_res)

    # Batch test with masking
    dummy_batch_obs = np.random.rand(4, INPUT_CHANNELS, VIEW_SIZE, VIEW_SIZE).astype(np.float32)
    dummy_batch_tensor_np = torch.from_numpy(dummy_batch_obs) #.to(device) if using GPU

    mask = torch.ones((4, NUM_ACTIONS)) #.to(device) if using GPU
    mask[0, 0] = 0 
    mask[0, 2] = 0
    mask[1, 3] = 0
    print("\n--- Batch & Masking Test (Residual Model) ---")
    actions_masked_res, log_ps_masked_res, _, _ = model_res.get_action_and_value(dummy_batch_tensor_np, action_mask=mask)
    print("Mask (sample 0):", mask[0])
    print("Actions with mask (Residual):", actions_masked_res)
    assert actions_masked_res[0].item() != 0 and actions_masked_res[0].item() != 2, "Action masking failed for sample 0 (Residual)."
    assert actions_masked_res[1].item() != 3, "Action masking failed for sample 1 (Residual)."
    print("Action masking test passed (Residual).")
