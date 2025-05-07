import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import collections # For deque in maze generation

# --- Entity Codes ---
EMPTY = 0
WALL = 1
AGENT = 2
PELLET = 3
POWER_PELLET = 4
GHOST_NORMAL = 5  # Standard ghost
GHOST_VULNERABLE = 6 # Vulnerable ghost
GHOST_SCATTER = 7 # Ghost in scatter mode (not directly used in grid, but as internal state)

# --- Ghost Modes ---
CHASE = 0
SCATTER = 1
VULNERABLE = 2

class PacManEnvEnhanced(gym.Env):
    metadata = {'render_modes': ['ansi', 'rgb_array'], 'render_fps': 10}

    def __init__(self, 
                 grid_size=(19, 21), # Odd numbers for better maze generation
                 num_ghosts=4, 
                 num_pellets_target_ratio=0.4, # Ratio of free cells to be pellets
                 num_power_pellets=4, 
                 egocentric_view_size=9, # Must be odd
                 max_steps_episode=500,
                 ghost_scatter_time=70, # Timesteps ghosts scatter
                 ghost_chase_time=200, # Timesteps ghosts chase
                 power_pellet_duration=60, # Timesteps ghosts are vulnerable
                 reward_pellet=10,
                 reward_power_pellet=50,
                 reward_eat_ghost=200,
                 penalty_ghost_collision=-500,
                 penalty_step=-1,
                 penalty_wall_bump=-5,
                 reward_clear_level=1000
                 ):
        super(PacManEnvEnhanced, self).__init__()

        assert grid_size[0] % 2 == 1 and grid_size[1] % 2 == 1, "Grid dimensions must be odd for maze generation."
        assert egocentric_view_size % 2 == 1, "Egocentric view size must be odd."

        self.height, self.width = grid_size
        self.num_ghosts = num_ghosts
        self.num_pellets_target_ratio = num_pellets_target_ratio
        self.num_power_pellets_initial = num_power_pellets
        self.egocentric_view_size = egocentric_view_size
        self.max_steps_episode = max_steps_episode
        
        self.reward_pellet = reward_pellet
        self.reward_power_pellet = reward_power_pellet
        self.reward_eat_ghost = reward_eat_ghost
        self.penalty_ghost_collision = penalty_ghost_collision
        self.penalty_step = penalty_step
        self.penalty_wall_bump = penalty_wall_bump
        self.reward_clear_level = reward_clear_level

        self.current_step = 0
        self.action_space = spaces.Discrete(4) # 0:Up, 1:Down, 2:Left, 3:Right
        
        self.observation_channels = 6 # Wall, Pellet, Power Pellet, Ghost (Normal), Ghost (Vulnerable), Agent
        self.observation_space = spaces.Box(low=0, high=1, 
                                            shape=(self.observation_channels, self.egocentric_view_size, self.egocentric_view_size), 
                                            dtype=np.float32)
        
        self.grid = np.zeros(grid_size, dtype=np.int8)
        self.agent_pos = np.array([0, 0], dtype=np.int16)
        
        # Ghost properties
        self.ghost_positions = np.zeros((num_ghosts, 2), dtype=np.int16)
        self.ghost_modes = np.full(num_ghosts, CHASE, dtype=np.int8)
        self.ghost_home_positions = np.zeros((num_ghosts, 2), dtype=np.int16) # Where ghosts respawn
        self.ghost_scatter_targets = np.array([[1,1], [1, self.width-2], [self.height-2, 1], [self.height-2, self.width-2]], dtype=np.int16) # Corner targets
        self.ghost_mode_timer = 0
        self.ghost_scatter_time = ghost_scatter_time
        self.ghost_chase_time = ghost_chase_time
        self.current_ghost_phase = CHASE # CHASE or SCATTER

        self.pellet_positions = set()
        self.power_pellet_positions = set()
        
        self.score = 0
        self.power_pellet_active_timer = 0
        self.power_pellet_duration = power_pellet_duration
        self.total_initial_pellets = 0 # Includes power pellets for win condition
        
        self._generate_maze()
        self.reset()

    def _generate_maze(self):
        """Generates a maze using Randomized DFS. Walls are 1, paths are 0."""
        self.maze_grid = np.ones(self.grid_size, dtype=np.int8) # Start with all walls
        
        # Randomized DFS
        stack = collections.deque()
        start_r, start_c = (random.randrange(1, self.height, 2), random.randrange(1, self.width, 2))
        self.maze_grid[start_r, start_c] = EMPTY
        stack.append((start_r, start_c))

        while stack:
            r, c = stack[-1]
            neighbors = []
            # Check neighbors (2 steps away)
            for dr, dc in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nr, nc = r + dr, c + dc
                if 0 < nr < self.height -1 and 0 < nc < self.width -1 and self.maze_grid[nr, nc] == WALL:
                    neighbors.append((nr, nc, r + dr//2, c + dc//2)) # Neighbor, Wall_between
            
            if neighbors:
                nr, nc, wr, wc = random.choice(neighbors)
                self.maze_grid[nr, nc] = EMPTY # Carve path to neighbor
                self.maze_grid[wr, wc] = EMPTY # Carve path for wall between
                stack.append((nr, nc))
            else:
                stack.pop()
        
        # Ensure border walls, though DFS on odd grid with 1-step border already implies this
        self.maze_grid[0, :] = self.maze_grid[-1, :] = WALL
        self.maze_grid[:, 0] = self.maze_grid[:, -1] = WALL

    def _get_random_empty_cell(self, ensure_not_in=None):
        if ensure_not_in is None:
            ensure_not_in = []
        attempts = 0
        while attempts < self.height * self.width * 2:
            r, c = random.randint(1, self.height - 2), random.randint(1, self.width - 2)
            is_occupied = False
            for occupied_pos in ensure_not_in:
                if r == occupied_pos[0] and c == occupied_pos[1]:
                    is_occupied = True
                    break
            if self.maze_grid[r, c] == EMPTY and not is_occupied:
                return np.array([r, c], dtype=np.int16)
            attempts += 1
        # Fallback if too many attempts (should be rare with good maze)
        for r in range(1, self.height - 1):
            for c in range(1, self.width - 1):
                is_occupied = False
                for occupied_pos in ensure_not_in:
                    if r == occupied_pos[0] and c == occupied_pos[1]:
                        is_occupied = True; break
                if self.maze_grid[r,c] == EMPTY and not is_occupied: return np.array([r,c], dtype=np.int16)
        return np.array([1,1], dtype=np.int16) # Default fallback

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.score = 0
        self.power_pellet_active_timer = 0
        self.ghost_mode_timer = 0
        self.current_ghost_phase = CHASE # Start with chase
        
        self._generate_maze() # Regenerate maze for variability, or comment out for fixed maze
        self.grid = np.copy(self.maze_grid) # Copy maze structure to game grid

        occupied_for_placement = []

        # Place Agent
        self.agent_pos = self._get_random_empty_cell()
        self.grid[self.agent_pos[0], self.agent_pos[1]] = AGENT
        occupied_for_placement.append(self.agent_pos.tolist())

        # Place Ghosts and their homes (make homes distinct for respawn)
        # For simplicity, ghost homes are their starting positions.
        for i in range(self.num_ghosts):
            self.ghost_positions[i] = self._get_random_empty_cell(ensure_not_in=occupied_for_placement)
            self.ghost_home_positions[i] = np.copy(self.ghost_positions[i])
            self.grid[self.ghost_positions[i,0], self.ghost_positions[i,1]] = GHOST_NORMAL
            self.ghost_modes[i] = CHASE # Initial mode
            occupied_for_placement.append(self.ghost_positions[i].tolist())
            
        # Place Pellets and Power Pellets
        self.pellet_positions.clear()
        self.power_pellet_positions.clear()
        
        empty_path_cells = []
        for r in range(1, self.height - 1):
            for c in range(1, self.width - 1):
                if self.grid[r, c] == EMPTY: # Path cells (not agent/ghost start)
                    empty_path_cells.append((r,c))
        
        random.shuffle(empty_path_cells)
        
        num_power_pellets_placed = 0
        for r, c in empty_path_cells:
            if num_power_pellets_placed < self.num_power_pellets_initial:
                self.grid[r,c] = POWER_PELLET
                self.power_pellet_positions.add((r,c))
                num_power_pellets_placed += 1
            else: # Fill remaining with normal pellets up to ratio
                break # Temp break to limit pellets for now
        
        # Count remaining empty path cells for normal pellets
        current_pellet_count = 0
        path_cells_for_pellets = [cell for cell in empty_path_cells if self.grid[cell[0], cell[1]] == EMPTY]
        target_pellets = int(len(path_cells_for_pellets) * self.num_pellets_target_ratio)

        for r, c in path_cells_for_pellets:
            if current_pellet_count < target_pellets:
                self.grid[r,c] = PELLET
                self.pellet_positions.add((r,c))
                current_pellet_count +=1
            else:
                break
        
        self.total_initial_pellets = len(self.pellet_positions) + len(self.power_pellet_positions)

        self.stats = {
            "pellets_eaten": 0, "power_pellets_eaten": 0, "ghosts_eaten": 0,
            "collisions_with_ghost": 0, "wall_bumps": 0
        }
        return self._get_observation(), self._get_info()

    def _get_observation(self):
        obs = np.zeros((self.observation_channels, self.egocentric_view_size, self.egocentric_view_size), dtype=np.float32)
        pad = self.egocentric_view_size // 2
        
        for r_view in range(self.egocentric_view_size):
            for c_view in range(self.egocentric_view_size):
                grid_r, grid_c = self.agent_pos[0] - pad + r_view, self.agent_pos[1] - pad + c_view
                
                if 0 <= grid_r < self.height and 0 <= grid_c < self.width:
                    cell_type = self.grid[grid_r, grid_c]
                    if cell_type == WALL: obs[0, r_view, c_view] = 1.0
                    elif cell_type == PELLET: obs[1, r_view, c_view] = 1.0
                    elif cell_type == POWER_PELLET: obs[2, r_view, c_view] = 1.0
                    elif cell_type == GHOST_NORMAL: obs[3, r_view, c_view] = 1.0
                    elif cell_type == GHOST_VULNERABLE: obs[4, r_view, c_view] = 1.0
                    # Agent channel is handled separately if needed, here it's implicit center
                else: # Out of bounds is like a wall
                    obs[0, r_view, c_view] = 1.0 
        
        obs[5, pad, pad] = 1.0 # Agent's own position in the view center
        return obs

    def _get_info(self):
        # Enhanced Risk Signal
        risk_signal_val = 0.0
        min_dist_to_normal_ghost = float('inf')
        normal_ghosts_in_proximity = 0
        proximity_radius_sq = 5**2 # squared for efficiency

        for i in range(self.num_ghosts):
            if self.ghost_modes[i] != VULNERABLE: # Only consider non-vulnerable ghosts for risk
                dist_sq = (self.agent_pos[0] - self.ghost_positions[i,0])**2 + \
                          (self.agent_pos[1] - self.ghost_positions[i,1])**2
                dist = np.sqrt(dist_sq)
                min_dist_to_normal_ghost = min(min_dist_to_normal_ghost, dist)
                if dist_sq <= proximity_radius_sq:
                    normal_ghosts_in_proximity += 1
                    # Contribution to risk: inverse distance, capped
                    risk_signal_val += 1.0 / (1.0 + dist) 
        
        # Normalize risk signal (e.g. by number of ghosts or a max value)
        if normal_ghosts_in_proximity > 0:
             risk_signal_val = min(risk_signal_val / normal_ghosts_in_proximity, 1.0) if normal_ghosts_in_proximity > 0 else 0.0


        return {
            "score": self.score,
            "step_count": self.current_step,
            "pellets_remaining": len(self.pellet_positions),
            "power_pellets_remaining": len(self.power_pellet_positions),
            "agent_pos_row": self.agent_pos[0], "agent_pos_col": self.agent_pos[1],
            "power_pellet_active_timer": self.power_pellet_active_timer,
            "min_dist_to_normal_ghost": min_dist_to_normal_ghost if min_dist_to_normal_ghost != float('inf') else -1,
            "risk_signal": risk_signal_val, 
            "ghost_modes": [int(m) for m in self.ghost_modes], # list of int for easier logging
            "stats_pellets_eaten": self.stats["pellets_eaten"],
            "stats_power_pellets_eaten": self.stats["power_pellets_eaten"],
            "stats_ghosts_eaten": self.stats["ghosts_eaten"],
            "stats_collisions": self.stats["collisions_with_ghost"],
            "stats_wall_bumps": self.stats["wall_bumps"]
        }

    def _update_ghost_modes_and_timers(self):
        if self.power_pellet_active_timer > 0:
            self.power_pellet_active_timer -= 1
            for i in range(self.num_ghosts):
                self.ghost_modes[i] = VULNERABLE
            if self.power_pellet_active_timer == 0: # Vulnerability ends
                for i in range(self.num_ghosts):
                    self.ghost_modes[i] = self.current_ghost_phase # Revert to CHASE/SCATTER
        else: # Normal CHASE/SCATTER cycle
            self.ghost_mode_timer +=1
            if self.current_ghost_phase == CHASE and self.ghost_mode_timer >= self.ghost_chase_time:
                self.current_ghost_phase = SCATTER
                self.ghost_mode_timer = 0
            elif self.current_ghost_phase == SCATTER and self.ghost_mode_timer >= self.ghost_scatter_time:
                self.current_ghost_phase = CHASE
                self.ghost_mode_timer = 0
            
            for i in range(self.num_ghosts):
                 if self.ghost_modes[i] != VULNERABLE: # Don't override vulnerability
                    self.ghost_modes[i] = self.current_ghost_phase


    def _move_entity(self, current_pos, action):
        """Helper to get new position if valid, else current_pos"""
        dr = [-1, 1, 0, 0]; dc = [0, 0, -1, 1] # Up, Down, Left, Right
        
        new_r, new_c = current_pos[0] + dr[action], current_pos[1] + dc[action]
        
        # Check bounds and wall collision (ghosts can't move into walls)
        if 0 <= new_r < self.height and 0 <= new_c < self.width and \
           self.maze_grid[new_r, new_c] != WALL: # Use maze_grid for static obstacles
            return np.array([new_r, new_c], dtype=np.int16)
        return current_pos # No move if invalid

    def _move_ghosts_ai(self):
        # Clear old ghost positions from grid, handling what was underneath
        for i in range(self.num_ghosts):
            r, c = self.ghost_positions[i,0], self.ghost_positions[i,1]
            if (r,c) in self.pellet_positions: self.grid[r,c] = PELLET
            elif (r,c) in self.power_pellet_positions: self.grid[r,c] = POWER_PELLET
            elif self.grid[r,c] != AGENT : self.grid[r,c] = EMPTY # Avoid clearing agent

        for i in range(self.num_ghosts):
            current_pos = self.ghost_positions[i]
            target_pos = np.copy(self.agent_pos) # Default for CHASE

            if self.ghost_modes[i] == SCATTER:
                target_pos = self.ghost_scatter_targets[i % len(self.ghost_scatter_targets)]
            elif self.ghost_modes[i] == VULNERABLE:
                # Fleeing behavior: try to move away from agent or randomly
                # For simplicity, just random valid move if vulnerable for now
                valid_moves = []
                for act in range(4):
                    next_p = self._move_entity(current_pos, act)
                    if not np.array_equal(next_p, current_pos): # if move is valid
                        valid_moves.append(next_p)
                if valid_moves:
                     self.ghost_positions[i] = random.choice(valid_moves)
                continue # Skip targeted movement for vulnerable ghosts

            # Determine best action to move towards target
            # Simple heuristic: try moves that reduce Manhattan distance, prefer non-reversal
            # More advanced: A* or learned policies
            best_action = -1
            min_dist_sq = float('inf')

            # TODO: Prevent ghosts from occupying the same cell (more complex)
            # For now, they might overlap if their targets align.
            
            # Try to move towards target
            possible_actions = list(range(4))
            random.shuffle(possible_actions) # Add randomness to break ties

            for action_idx in possible_actions:
                next_pos_candidate = self._move_entity(current_pos, action_idx)
                if not np.array_equal(next_pos_candidate, current_pos): # If it's a valid move
                    dist_sq = (next_pos_candidate[0] - target_pos[0])**2 + \
                              (next_pos_candidate[1] - target_pos[1])**2
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        best_action = action_idx
            
            if best_action != -1:
                self.ghost_positions[i] = self._move_entity(current_pos, best_action)
        
        # Update grid with new ghost positions
        for i in range(self.num_ghosts):
            r, c = self.ghost_positions[i,0], self.ghost_positions[i,1]
            if self.grid[r,c] != AGENT: # Don't overwrite agent if ghost lands on it (collision handled later)
                self.grid[r,c] = GHOST_VULNERABLE if self.ghost_modes[i] == VULNERABLE else GHOST_NORMAL


    def step(self, action):
        self.current_step += 1
        reward = self.penalty_step 
        terminated = False
        truncated = False

        # 1. Agent Movement
        old_agent_pos = np.copy(self.agent_pos)
        next_agent_pos = self._move_entity(self.agent_pos, action)

        if np.array_equal(next_agent_pos, self.agent_pos): # Agent bumped wall
            reward += self.penalty_wall_bump
            self.stats["wall_bumps"] += 1
        else: # Agent moved
            self.grid[self.agent_pos[0], self.agent_pos[1]] = EMPTY # Clear old agent spot
            self.agent_pos = next_agent_pos
        
        # 2. Agent Interactions at New Position (Pellets, Power Pellets)
        current_cell_r, current_cell_c = self.agent_pos[0], self.agent_pos[1]
        
        if (current_cell_r, current_cell_c) in self.pellet_positions:
            reward += self.reward_pellet
            self.score += self.reward_pellet
            self.pellet_positions.remove((current_cell_r, current_cell_c))
            self.stats["pellets_eaten"] += 1
            # Grid cell will be AGENT
        elif (current_cell_r, current_cell_c) in self.power_pellet_positions:
            reward += self.reward_power_pellet
            self.score += self.reward_power_pellet
            self.power_pellet_positions.remove((current_cell_r, current_cell_c))
            self.stats["power_pellets_eaten"] += 1
            self.power_pellet_active_timer = self.power_pellet_duration
            # Ghost modes will be updated to VULNERABLE by _update_ghost_modes_and_timers
        
        self.grid[current_cell_r, current_cell_c] = AGENT # Mark agent new position

        # 3. Update Ghost Modes (Chase/Scatter/Vulnerable) and Timers
        self._update_ghost_modes_and_timers()

        # 4. Ghost Movement AI
        self._move_ghosts_ai() # This also updates ghost positions on the grid

        # 5. Check for Agent-Ghost Collisions (Post-Ghost Movement)
        for i in range(self.num_ghosts):
            if np.array_equal(self.agent_pos, self.ghost_positions[i]):
                if self.ghost_modes[i] == VULNERABLE:
                    reward += self.reward_eat_ghost
                    self.score += self.reward_eat_ghost
                    self.stats["ghosts_eaten"] += 1
                    # Respawn ghost at its home position, reset its mode
                    self.ghost_positions[i] = np.copy(self.ghost_home_positions[i])
                    self.ghost_modes[i] = CHASE # Or a brief "eaten" state then home
                    # Update grid for respawned ghost
                    gr, gc = self.ghost_positions[i,0], self.ghost_positions[i,1]
                    if self.grid[gr,gc] != AGENT: self.grid[gr,gc] = GHOST_NORMAL
                else: # Collision with normal/chasing/scattering ghost
                    reward += self.penalty_ghost_collision
                    self.score += self.penalty_ghost_collision # score can go negative
                    self.stats["collisions_with_ghost"] += 1
                    terminated = True
                    break # Episode ends on collision
        if terminated:
             return self._get_observation(), reward, terminated, truncated, self._get_info()


        # 6. Check for Win/Loss Conditions
        if not self.pellet_positions and not self.power_pellet_positions:
            reward += self.reward_clear_level
            self.score += self.reward_clear_level
            terminated = True
        
        if self.current_step >= self.max_steps_episode:
            truncated = True
            terminated = True # Consider truncation as termination for most RL algos

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def render(self, mode='ansi'):
        if mode == 'ansi':
            # Create a temporary grid for rendering to show AGENT on top if overlap
            render_grid_display = np.copy(self.grid)
            render_grid_display[self.agent_pos[0], self.agent_pos[1]] = AGENT # Ensure agent is shown

            char_map = {EMPTY: ' ', WALL: '#', AGENT: 'A', PELLET: '.', 
                        POWER_PELLET: 'O', GHOST_NORMAL: 'G', GHOST_VULNERABLE: 'g'}
            
            output = ""
            for r_idx in range(self.height):
                row_str = ""
                for c_idx in range(self.width):
                    row_str += char_map.get(render_grid_display[r_idx, c_idx], '?')
                output += row_str + "\n"
            
            info = self._get_info()
            output += f"Score: {self.score}, Step: {self.current_step}/{self.max_steps_episode}\n"
            output += f"PwrTimer: {self.power_pellet_active_timer}, RiskSignal: {info['risk_signal']:.2f}\n"
            output += f"GhostModes: {info['ghost_modes']}\n"
            print(output)
        elif mode == 'rgb_array':
            # Basic RGB rendering
            scale = 10
            rgb_img = np.zeros((self.height * scale, self.width * scale, 3), dtype=np.uint8)
            colors = {
                EMPTY: [0,0,0], WALL: [100,100,100], AGENT: [255,255,0], 
                PELLET: [200,200,200], POWER_PELLET: [255,255,255], 
                GHOST_NORMAL: [255,0,0], GHOST_VULNERABLE: [0,0,255]
            }
            render_grid_display = np.copy(self.grid)
            render_grid_display[self.agent_pos[0], self.agent_pos[1]] = AGENT

            for r in range(self.height):
                for c in range(self.width):
                    cell_val = render_grid_display[r,c]
                    color = colors.get(cell_val, [128,0,128]) # Magenta for unknown
                    rgb_img[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = color
            return rgb_img
        else:
            super(PacManEnvEnhanced, self).render(mode=mode)

    def close(self):
        pass

if __name__ == '__main__':
    env = PacManEnvEnhanced(grid_size=(15,17), egocentric_view_size=7, num_ghosts=2, max_steps_episode=300)
    obs, info = env.reset()
    print("Initial Observation Shape:", obs.shape)
    print("Initial Info:", info)
    
    for episode in range(1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_ep_reward = 0
        env.render()
        for step in range(env.max_steps_episode):
            action = env.action_space.sample() 
            # action = int(input("Enter action (0:U, 1:D, 2:L, 3:R): ")) # Manual play
            obs, reward, terminated, truncated, info = env.step(action)
            total_ep_reward += reward
            env.render()
            print(f"Action: {action}, Reward: {reward:.2f}, Term: {terminated}, Trunc: {truncated}")
            # print(f"Info: {info}")
            if terminated or truncated:
                print(f"Episode {episode+1} finished after {step+1} steps. Score: {info['score']}, Total Reward: {total_ep_reward:.2f}")
                break
    env.close()
