import pygame # Pygame is a set of Python modules designed for writing video games. We use it here to draw our dots and simulation!
import random # This module helps us make random choices, like picking random directions for dots or deciding if a mutation happens.
import math # Math functions are needed for calculations, like distances and angles for movement.
import pickle # Pickle allows us to save our simulation's progress (like the dots' brains) to a file and load it later.
import os # The OS module helps us interact with the operating system, like creating directories for saving files.
import logging # Logging helps us print messages (like info or warnings) to the console to see what the program is doing.
import datetime # Used to create unique filenames for saved simulations based on the current date and time.
import numpy as np # NumPy is a powerful library for numerical computing, especially for working with arrays. We use it for the Q-table in Reinforcement Learning.

# --- 🎮 GENERAL SIMULATION CONFIGURATION ---
# Welcome to the control panel for our Dot AI Simulation!
# These are global settings that you can tweak to change how the simulation runs.

# SIMULATION_TYPE: Choose which AI or control mode to run.
# - "Genetic": Dots evolve over generations to find the goal (like natural selection).
# - "Reinforcement": A dot learns by trial and error, getting rewards or penalties (like training a pet).
# - "Player": You control the dot (not implemented in this version, but could be an extension!).
SIMULATION_TYPE = SIMULATION_TEXT = "Genetic" # Current options: "Genetic", "Reinforcement". (Player mode is a fun idea for later!)

POPULATION_SIZE = 500 # For Genetic Algorithm: How many dots are in each generation. More dots mean more variety but slower simulation.
MAX_VELOCITY = 3 # How fast a dot can move in any direction (pixels per frame).

# --- 🧬 GENETIC ALGORITHM CONFIGURATION ---
# These settings are specific to the "Genetic" AI mode.

MUTATION_RATE = 0.02 # Chance (from 0.0 to 1.0) that a dot's "brain" (its movement instructions) will have a random change.
                     # A small mutation rate helps explore new solutions without disrupting good ones too much.
BRAIN_SIZE = 1500 # How many steps/instructions are in a dot's "brain" for the Genetic Algorithm.
                 # This is the maximum number of moves a dot can make in its lifetime.

# --- 🤖 REINFORCEMENT LEARNING (Q-LEARNING) CONFIGURATION ---
# These settings are specific to the "Reinforcement" AI mode.

LEARNING_RATE = 0.1 # (Alpha) How quickly the AI updates its knowledge (Q-table) based on new experiences.
                    # A higher rate means faster learning but can be unstable. (0.0 to 1.0)
DISCOUNT_FACTOR = 0.95 # (Gamma) How much the AI values future rewards compared to immediate rewards.
                       # A value closer to 1 makes the AI more farsighted. (0.0 to 1.0)
EPSILON = 0.5 # Exploration Rate: Chance (from 0.0 to 1.0) that the AI will try a random action instead of the "best" known action.
              # This helps the AI discover new paths and avoid getting stuck in a rut. Decreases over time in some setups.
GRID_SIZE = 10 # Divides the game screen into a coarser grid for the Q-table.
               # A smaller grid means more states (more memory/slower learning), a larger grid means fewer states (faster learning but less precision).

# --- 🎨 COLOR DEFINITIONS ---
# Let's define some colors to make our simulation look good!
# Colors are represented as (R, G, B) tuples, where R, G, B are values from 0 to 255.

# Standard Colors (will be overridden by accessibility-friendly versions)
WHITE_STD = (255, 255, 255)
BLACK_STD = (0, 0, 0)
RED_STD = (225, 0, 0) # A bright red
GREEN_STD = (0, 250, 0) # A bright green
BLUE_STD = (0, 0, 250) # A bright blue

# Accessibility-Focused Colors:
# Using pure white/black or very saturated colors can be hard on the eyes.
# These adjusted colors aim for better readability and comfort.

# Instead of pure white, use a slightly off-white color to reduce eye strain.
WHITE_ACC = (224, 224, 224)
# Pure black can be harsh; a dark gray is often preferred.
BLACK_ACC = (18, 18, 18)
# Use a less saturated red to avoid harshness, while still being clearly red.
RED_ACC = (255, 107, 107)
# Opt for a softer green that maintains good contrast.
GREEN_ACC = (76, 175, 80)
# Choose a blue that is not too vibrant to ensure readability.
BLUE_ACC = (66, 165, 245)

# Override the standard colors with the accessibility-friendly versions.
# This makes our simulation more comfortable to look at!
WHITE = WHITE_ACC
BLACK = BLACK_ACC
RED = RED_ACC
GREEN = GREEN_ACC
BLUE = BLUE_ACC

# --- 🖥️ GAME UI (USER INTERFACE) CONFIGURATION ---
WIDTH, HEIGHT = 1800, 980 # Size of the game window in pixels (width, height).
SCREEN_COLOR = BLACK # Background color of the game window.
FONT_SIZE = 36 # Size of the text used for displaying information.
FONT_COLOR = GREEN # Color of the information text.

# --- 🔵 DOT CONFIGURATION ---
DOT_SIZE = 3 # Radius of the regular dots in pixels.
DOT_COLOR = BLUE # Color of the regular dots.
BEST_DOT_SIZE = 5 # Radius of the "best" dot (e.g., in Genetic Algorithm) to make it stand out.
BEST_DOT_COLOR = RED # Color of the "best" dot.

# --- 🎯 GOAL CONFIGURATION ---
GOAL_X = WIDTH // 3 # X-coordinate of the goal (center of the screen horizontally).
GOAL_Y = 25         # Y-coordinate of the goal (near the top of the screen).
GOAL = (GOAL_X, GOAL_Y) # The (x, y) position of the goal.
GOAL_SIZE = 12 # Radius of the goal circle in pixels.
GOAL_COLOR = GREEN # Color of the goal.

# --- 🚧 OBSTACLE CONFIGURATION ---
OBSTACLE_COUNT = 200 # How many obstacles will be randomly generated on the map.
OBSTACLE_COLOR = WHITE # Color of the obstacles.
OBSTACLE_MIN_WIDTH = 20 # Minimum width of an obstacle.
OBSTACLE_MAX_WIDTH = 40 # Maximum width of an obstacle.
OBSTACLE_MIN_HEIGHT = 10 # Minimum height of an obstacle.
OBSTACLE_MAX_HEIGHT = 10 # Maximum height of an obstacle.

# --- ⚙️ MISCELLANEOUS CONFIGURATION (Internal Setup) ---
# These settings are mostly for the program's internal logic. You usually don't need to change these.

# Convert SIMULATION_TYPE string to a number for easier checking in code.
# This is a common trick to make `if` statements cleaner later on.
SIM_TYPE_PLAYER = 0
SIM_TYPE_GENETIC = 1
SIM_TYPE_REINFORCEMENT = 2

if SIMULATION_TEXT == "Genetic":
    SIMULATION_TYPE = SIM_TYPE_GENETIC
elif SIMULATION_TEXT == "Reinforcement":
    SIMULATION_TYPE = SIM_TYPE_REINFORCEMENT
else: # Default or if "Player" is typed (though not fully implemented)
    SIMULATION_TYPE = SIM_TYPE_PLAYER
    SIMULATION_TEXT = "Player" # Ensure text matches if type is defaulted

LOG_LEVEL = logging.INFO # How detailed the log messages should be. INFO shows general progress. DEBUG shows much more detail.
# Use a relative path for the save directory. This means it will be created inside the folder where your script is.
SAVE_DIR = "simulation_saves" # Changed from ".\\simulation_saves" for better cross-platform compatibility and to avoid escape sequence warnings.

# Create a base filename for saved simulations, including the AI type and current timestamp.
# This ensures each saved file has a unique name.
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if SIMULATION_TYPE == SIM_TYPE_GENETIC:
    SAVE_FILE_PREFIX = f"genetic_dot_ai_results_{timestamp}"
elif SIMULATION_TYPE == SIM_TYPE_REINFORCEMENT:
    SAVE_FILE_PREFIX = f"reinforcement_dot_ai_results_{timestamp}"
else: # Player or other modes
    SAVE_FILE_PREFIX = f"player_dot_ai_results_{timestamp}"

# Helper function to get the string name of a variable (e.g., to print "RED" instead of "(225,0,0)")
# This is a bit advanced but useful for making UI messages more readable.
def get_variable_name(var_to_find):
    # globals() gives a dictionary of all global variables.
    # We look through it to find which variable name matches the color value we passed in.
    for name, value in globals().items():
        if value is var_to_find: # 'is' checks if they are the exact same object
            return name
    return None # If not found

# --- 📝 LOGGING SETUP ---
# Configure how log messages are formatted and displayed.
log = logging.getLogger('dot_logger') # Create a logger instance.
log.setLevel(logging.DEBUG) # Set the minimum level of messages this logger will handle.

# Console Handler: This will print log messages to your terminal/console.
console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL) # Set the level for what actually gets printed to console.

# Formatter: Define how each log message line should look.
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # Timestamp - Level - Message
console_handler.setFormatter(formatter)
log.addHandler(console_handler) # Add the configured handler to our logger.

# Ensure the directory for saving simulation files exists. If not, create it.
# os.makedirs() can create parent directories if needed (like mkdir -p).
# exist_ok=True means it won't raise an error if the directory already exists.
os.makedirs(SAVE_DIR, exist_ok=True)
log.info(f"Save directory '{SAVE_DIR}' is ready.")

# --- 🧠 BRAIN CLASS (For Genetic Algorithm) ---
# The Brain class defines the "DNA" or instruction set for dots in the Genetic Algorithm.
# Each dot gets a Brain, which tells it how to move at each step.
class Brain:
    def __init__(self, size: int):
        # 'size' is the number of instructions (moves) this brain will have.
        self.size = size
        # 'directions' is a list of movement vectors. Each vector is an (x_change, y_change) pair.
        # We initialize it with 'size' number of random direction vectors.
        self.directions: list[tuple[float, float]] = [self.random_vector() for _ in range(size)]
        # 'step' keeps track of which instruction/direction the dot is currently on.
        self.step: int = 0

    def random_vector(self) -> tuple[float, float]:
        """Generates a random 2D direction vector with a length of 1 (a unit vector)."""
        # Pick a random angle between 0 and 2*PI radians (a full circle).
        angle = random.uniform(0, 2 * math.pi)
        # Convert the angle to an (x, y) vector using trigonometry.
        # math.cos and math.sin give the x and y components for a unit circle.
        return (math.cos(angle), math.sin(angle))

    def clone(self):
        """Creates an exact copy of this brain.
        This is crucial for the Genetic Algorithm when creating offspring.
        The child inherits the parent's brain (instructions).
        """
        # Create a new Brain instance with the same size.
        cloned_brain = Brain(self.size)
        # Copy the list of directions. Using [:] creates a shallow copy, which is fine here
        # as the tuples (vectors) themselves are immutable.
        cloned_brain.directions = self.directions[:]
        return cloned_brain

    def mutate(self):
        """Applies random changes (mutations) to the brain's directions.
        Mutation is a key part of Genetic Algorithms. It introduces new variations
        into the population, potentially leading to better solutions.
        """
        # Go through each direction instruction in the brain.
        for i in range(len(self.directions)):
            # There's a small chance (MUTATION_RATE) that this instruction will be changed.
            if random.random() < MUTATION_RATE:
                # If mutation occurs, replace the current direction with a new random one.
                self.directions[i] = self.random_vector()

# --- 🔵 DOT CLASS ---
# The Dot class represents a single agent (our little dot) in the simulation.
# It handles its own position, movement, and status (alive, dead, reached goal).
class Dot:
    def __init__(self, width: int, height: int, goal: tuple[int, int], obstacles: list[pygame.Rect], is_rl_copy: bool = False):
        # --- Dot's Physical Properties & Environment Awareness ---
        self.width: int = width # Width of the simulation area (screen width)
        self.height: int = height # Height of the simulation area (screen height)
        self.goal: tuple[int, int] = goal # Target (x,y) coordinates the dot tries to reach
        self.obstacles: list[pygame.Rect] = obstacles # List of rectangular obstacles to avoid

        # --- Dot's State ---
        # Position: Where the dot is. Represented as [x, y]. Using list[float] for precise movement.
        self.pos: list[float] = [float(self.width / 2), float(self.height - 20)] # Start near bottom-center
        # Velocity: How fast and in what direction the dot is moving [vx, vy].
        self.vel: list[float] = [0.0, 0.0] # Starts stationary
        # Acceleration: Change in velocity [ax, ay]. This is what the "brain" or AI controls.
        self.acc: list[float] = [0.0, 0.0] # Starts with no acceleration

        self.dead: bool = False # Is the dot out of moves, hit a wall, or an obstacle?
        self.reached_goal: bool = False # Did the dot successfully reach the goal?
        self.fitness: float = 0.0 # A score indicating how well the dot performed (higher is better).
        self.is_best: bool = False # Is this dot currently the "best" performing one in the population? (Used for display)

        # --- AI-Specific Initialization ---
        if SIMULATION_TYPE == SIM_TYPE_GENETIC: # Genetic Algorithm Dot
            self.brain: Brain = Brain(BRAIN_SIZE) # Each genetic dot gets its own "brain" with instructions.
            # self.is_rl_copy is not used for genetic dots in this setup.
        
        elif SIMULATION_TYPE == SIM_TYPE_REINFORCEMENT: # Reinforcement Learning (Q-learning) Dot
            # Q-table: This is the "brain" for an RL dot. It's a big table that stores the expected future reward
            # for taking each possible action in each possible state.
            # State: Defined by the dot's position on a grid (pos_x // GRID_SIZE, pos_y // GRID_SIZE).
            # Actions: 8 possible directions of movement.
            # Dimensions: (grid_width, grid_height, num_actions)
            # We initialize it with all zeros, meaning the AI initially knows nothing.
            if not hasattr(self, 'q_table') or self.q_table is None or is_rl_copy: # Initialize Q-table if it's a new RL dot or a copy needing one
                self.q_table: np.ndarray = np.zeros((self.width // GRID_SIZE, self.height // GRID_SIZE, 8))
            
            # Current state of the dot on the discrete grid used by the Q-table.
            self.state: tuple[int, int] = (int(self.pos[0]) // GRID_SIZE, int(self.pos[1]) // GRID_SIZE)
            # The action chosen by the RL agent. None initially.
            self.action: int | None = None # Action is an integer from 0 to 7, representing a direction.
            # self.previous_positions = [] # Could be used for debugging or more complex reward logic.

    def move(self):
        """Updates the dot's position based on its acceleration and velocity.
        This is where the physics of the dot's movement happens.
        """
        # --- Basic Physics Update ---
        # Velocity changes based on acceleration: v = u + at (here, t=1 frame)
        self.vel[0] += self.acc[0]
        self.vel[1] += self.acc[1]

        # Keep the dot from moving too fast.
        self.limit_velocity(MAX_VELOCITY)

        # Position changes based on velocity: s = ut (here, t=1 frame)
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        
        # --- AI-Specific Action/Acceleration Update ---
        if SIMULATION_TYPE == SIM_TYPE_GENETIC: # Genetic Algorithm
            # The dot follows the instructions in its brain, one step at a time.
            if self.brain.step < len(self.brain.directions):
                # Get the current instruction (acceleration vector) from the brain.
                self.acc = list(self.brain.directions[self.brain.step]) # Convert tuple to list for acc
                self.brain.step += 1 # Move to the next instruction for the next frame.
            else:
                # If the brain runs out of instructions, the dot is considered "dead" (can't move anymore).
                self.dead = True
        
        elif SIMULATION_TYPE == SIM_TYPE_REINFORCEMENT: # Reinforcement Learning
            # For RL, the dot needs to choose an action (which determines its acceleration).
            if self.action is None: # If no action chosen for this state yet
                self.action = self.choose_action() # AI decides what to do
            
            # Convert the chosen action (0-7) into an acceleration vector.
            self.acc = list(self.action_to_vector(self.action)) # Convert tuple to list for acc

            # Update the dot's current discrete state for the Q-table.
            # Ensure state indices are within the Q-table bounds.
            new_grid_x = max(0, min(int(self.pos[0]) // GRID_SIZE, (self.width // GRID_SIZE) - 1))
            new_grid_y = max(0, min(int(self.pos[1]) // GRID_SIZE, (self.height // GRID_SIZE) - 1))
            self.state = (new_grid_x, new_grid_y)
            # self.previous_positions.append((int(self.pos[0]), int(self.pos[1]))) # Optional: track path

    def limit_velocity(self, max_v: float):
        """Ensures the dot's speed does not exceed max_v."""
        # Calculate the current speed (magnitude of the velocity vector).
        current_speed_sq = self.vel[0]**2 + self.vel[1]**2
        if current_speed_sq > max_v**2: # Compare squared speeds to avoid costly sqrt
            current_speed = math.sqrt(current_speed_sq)
            # If too fast, scale down the velocity vector to the max_v.
            # This keeps the direction the same but reduces the speed.
            self.vel[0] = (self.vel[0] / current_speed) * max_v
            self.vel[1] = (self.vel[1] / current_speed) * max_v

    def update(self):
        """Updates the dot's state for one frame: moves it, checks for collisions, and handles AI logic."""
        if not self.dead and not self.reached_goal: # Only update if the dot is still active.
            self.move() # Perform the movement logic.
            
            # --- Reinforcement Learning: Learning Step ---
            if SIMULATION_TYPE == SIM_TYPE_REINFORCEMENT:
                # After moving, the RL agent is in a new state and receives a reward.
                reward = self.calculate_reward() # How good was that last move?
                
                # Get the new state after the move.
                next_grid_x = max(0, min(int(self.pos[0]) // GRID_SIZE, (self.width // GRID_SIZE) - 1))
                next_grid_y = max(0, min(int(self.pos[1]) // GRID_SIZE, (self.height // GRID_SIZE) - 1))
                next_state = (next_grid_x, next_grid_y)

                # Update the Q-table using the Bellman equation (core of Q-learning).
                # This is where the "learning" happens: adjusting the value of the previous state-action pair.
                # Ensure previous state and action are valid before updating Q-table
                if self.state is not None and self.action is not None and \
                   0 <= self.state[0] < self.q_table.shape[0] and \
                   0 <= self.state[1] < self.q_table.shape[1] and \
                   0 <= next_state[0] < self.q_table.shape[0] and \
                   0 <= next_state[1] < self.q_table.shape[1]:
                    self.update_q_table(self.state, self.action, reward, next_state)
                
                # The new state becomes the current state for the next decision.
                self.state = next_state
                # The dot needs to choose a new action for this new state.
                self.action = self.choose_action()

            # --- Collision Detection & Goal Check ---
            # Check for collision with screen boundaries (walls).
            # A small buffer (e.g., DOT_SIZE) is used so the dot dies if its edge hits the wall.
            if self.pos[0] < DOT_SIZE or self.pos[1] < DOT_SIZE or \
               self.pos[0] > self.width - DOT_SIZE or self.pos[1] > self.height - DOT_SIZE:
                self.dead = True
            # Check if the dot reached the goal.
            elif self.distance(self.pos, self.goal) < (GOAL_SIZE + DOT_SIZE): # Consider radii for collision
                self.reached_goal = True
            # Check for collision with any of the obstacles.
            else:
                for obstacle in self.obstacles:
                    # Pygame's Rect.collidepoint checks if a point is inside a rectangle.
                    if obstacle.collidepoint(self.pos[0], self.pos[1]):
                        self.dead = True
                        break # No need to check other obstacles if one is hit.

    def calculate_fitness(self):
        """Calculates the "fitness" score of the dot.
        Fitness determines how well the dot performed its task.
        This is crucial for the Genetic Algorithm to select parents.
        For Reinforcement Learning, this can be a measure of overall episode performance.
        """
        if self.reached_goal:
            # If the goal is reached, fitness is high!
            # For Genetic Algorithm, we also reward dots that reach the goal in fewer steps.
            # The (1.0 / 16.0) is a base high score, and the division by step squared heavily rewards speed.
            if SIMULATION_TYPE == SIM_TYPE_GENETIC:
                # Ensure brain.step is not zero to avoid division by zero error
                # (though it's unlikely to be zero if goal is reached)
                steps_taken = self.brain.step if self.brain.step > 0 else 1
                self.fitness = (1.0 / 16.0) + (10000.0 / (steps_taken * steps_taken))
            else: # For RL or other types, a simpler high score for reaching goal.
                self.fitness = 1000.0 
        else:
            # If the goal is not reached, fitness is based on how close the dot got to the goal.
            # Closer dots get higher fitness. We use the inverse of the squared distance.
            # Squaring the distance penalizes dots that are far away more heavily.
            # Adding 1 to distance avoids division by zero if distance is 0 (though unlikely if not at goal).
            distance_to_goal = self.distance(self.pos, self.goal)
            self.fitness = 1.0 / ((distance_to_goal * distance_to_goal) + 1e-6) # Add small epsilon to avoid division by zero

            # Optional: Penalize for being dead if not at goal
            if self.dead:
                self.fitness *= 0.5 # For example, halve the fitness if it died

    def calculate_reward(self) -> float: # Specific to Reinforcement Learning
        """Calculates the immediate reward or punishment for the dot's last action in RL."""
        if self.reached_goal:
            return 100.0 # Big reward for reaching the goal!
        elif self.dead:
            return -100.0 # Big penalty for dying (hitting wall/obstacle).
        else:
            # Small negative reward based on distance to goal (encourages getting closer).
            # The closer, the less negative the reward.
            distance_to_goal = self.distance(self.pos, self.goal)
            # Normalize distance or scale it to be a reasonable reward value.
            # Example: reward = -distance_to_goal / 100
            # Or, a reward for reducing distance:
            # if hasattr(self, 'prev_dist_to_goal') and self.prev_dist_to_goal is not None:
            #     reward = (self.prev_dist_to_goal - distance_to_goal) # Positive if got closer
            # else:
            #     reward = -distance_to_goal / (WIDTH) # Small penalty for existing
            # self.prev_dist_to_goal = distance_to_goal
            return - (distance_to_goal / 10.0) # Simple distance-based penalty

    def distance(self, pos1: list[float], pos2: tuple[int,int] | list[float]) -> float:
        """Calculates the Euclidean distance between two points (pos1 and pos2)."""
        # Uses the Pythagorean theorem: sqrt((x2-x1)^2 + (y2-y1)^2)
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx*dx + dy*dy)

    def clone(self): # Primarily used by Genetic Algorithm
        """Creates a copy of this dot."""
        # Create a new Dot instance with the same environmental parameters.
        # The 'is_rl_copy=False' means if this clone method is somehow used on an RL dot,
        # it would get a fresh Q-table, not a copy of the parent's.
        # This is generally fine as RL uses `reset_population` for its "cloning" logic.
        cloned_dot = Dot(self.width, self.height, self.goal, self.obstacles, is_rl_copy=False)
        
        if SIMULATION_TYPE == SIM_TYPE_GENETIC:
            # For Genetic Algorithm, the crucial part is cloning the brain.
            # The new dot gets a copy of the parent's brain (instructions).
            if hasattr(self, 'brain') and self.brain is not None:
                cloned_dot.brain = self.brain.clone()
            else:
                # This case should ideally not be hit if it's a genetic dot.
                log.warning("Cloning a genetic dot that appears to be missing its brain. Giving it a new random brain.")
                cloned_dot.brain = Brain(BRAIN_SIZE) 
        # Note: Q-table is not cloned here. RL dots handle Q-table propagation in `reset_population`.
        return cloned_dot

    # --- Reinforcement Learning Specific Methods ---
    def choose_action(self) -> int: # Specific to Reinforcement Learning
        """Selects an action (0-7 for 8 directions) using an epsilon-greedy strategy.
        Epsilon-greedy means:
        - With probability EPSILON: choose a random action (explore).
        - With probability 1-EPSILON: choose the action with the highest Q-value for the current state (exploit).
        """
        # Ensure current state is valid for Q-table lookup
        current_grid_x = max(0, min(self.state[0], self.q_table.shape[0] - 1))
        current_grid_y = max(0, min(self.state[1], self.q_table.shape[1] - 1))
        valid_state = (current_grid_x, current_grid_y)

        if random.uniform(0, 1) < EPSILON:
            # Explore: Choose a random action (0 to 7 for 8 directions).
            return random.randint(0, 7)
        else:
            # Exploit: Choose the best known action from the Q-table for the current state.
            # np.argmax finds the index (action) with the maximum Q-value.
            # Explicitly cast to int to ensure a Python integer type is returned.
            return int(np.argmax(self.q_table[valid_state[0], valid_state[1]]))

    def action_to_vector(self, action: int) -> tuple[float, float]: # Specific to Reinforcement Learning
        """Converts an integer action (0-7) into a 2D movement vector."""
        # Each action corresponds to a direction (e.g., 0=East, 1=Northeast, etc.).
        # We divide the circle (2*PI radians) into 8 segments.
        angle = action * (math.pi / 4) # angle for each of the 8 directions
        return (math.cos(angle), math.sin(angle)) # Returns a unit vector

    def update_q_table(self, state: tuple[int,int], action: int, reward: float, next_state: tuple[int,int]): # Specific to RL
        """Updates the Q-value in the Q-table using the Q-learning formula.
        Q(s,a) = Q(s,a) + LEARNING_RATE * [reward + DISCOUNT_FACTOR * max_a'(Q(s',a')) - Q(s,a)]
        s: current state, a: current action, s': next state, a': next action
        """
        # Find the best possible Q-value for the next_state (max_a' Q(s',a')).
        best_next_action_q_value = np.max(self.q_table[next_state[0], next_state[1]])
        
        # Calculate the "TD target": the reward plus the discounted value of the best next state.
        # This is what we think the Q-value for Q(state, action) *should* be.
        td_target = reward + DISCOUNT_FACTOR * best_next_action_q_value
        
        # Calculate the "TD error" (Temporal Difference error): the difference between the TD target and current Q-value.
        # This error tells us how much to adjust our current Q-value.
        td_error = td_target - self.q_table[state[0], state[1], action]
        
        # Update the Q-value for the original state-action pair.
        # We move it closer to the td_target by a step size of LEARNING_RATE.
        self.q_table[state[0], state[1], action] += LEARNING_RATE * td_error

    def reset(self): # Used by Reinforcement Learning to reset dot for a new episode
        """Resets the dot's state to its initial conditions."""
        self.pos = [float(self.width / 2), float(self.height - 20)] # Back to start
        self.vel = [0.0, 0.0]
        self.acc = [0.0, 0.0]
        self.dead = False
        self.reached_goal = False
        # self.fitness = 0.0 # Fitness might be recalculated or accumulated differently in RL

        if SIMULATION_TYPE == SIM_TYPE_REINFORCEMENT:
            # Reset RL specific state
            self.state = (int(self.pos[0]) // GRID_SIZE, int(self.pos[1]) // GRID_SIZE)
            self.action = None # Needs to choose a new action for the new state
            # self.previous_positions = [] # Clear path history

        elif SIMULATION_TYPE == SIM_TYPE_GENETIC:
            # For genetic dots, if they are reset (e.g. the champion carrying over), reset its brain step
            if hasattr(self, 'brain') and self.brain is not None:
                self.brain.step = 0
        
        self.is_best = False # No longer the "best" of a completed generation/episode until proven otherwise

# --- 👨‍👩‍👧‍👦 POPULATION CLASS ---
# The Population class manages a collection of Dot objects.
# This is central to the Genetic Algorithm, where a population evolves over generations.
# It can also be used in Reinforcement Learning to manage multiple agents learning simultaneously or sequentially.
class Population:
    def __init__(self, size: int, width: int, height: int, goal: tuple[int, int], obstacles: list[pygame.Rect]):
        # Store the parameters needed to create new dots.
        self.width = width
        self.height = height
        self.goal = goal
        self.obstacles = obstacles
        self.population_size = size

        # Create the initial population of dots.
        # Each dot is a new instance of the Dot class.
        self.dots: list[Dot] = [Dot(width, height, goal, obstacles, is_rl_copy=(SIMULATION_TYPE == SIM_TYPE_REINFORCEMENT)) for _ in range(size)]
        
        self.gen: int = 1 # Current generation number (or episode number for RL).
        
        # Tracking performance metrics:
        # For Genetic Algorithm, min_step is the fewest brain steps taken by a champion dot to reach the goal.
        self.min_step: int | float = BRAIN_SIZE if SIMULATION_TYPE == SIM_TYPE_GENETIC else float('inf')
        
        # Stores the actual Dot object that has the highest fitness in the current generation/overall.
        self.best_dot_object: Dot | None = None
        
        # Highest fitness score achieved by any dot in the current generation/overall.
        self.max_fitness: float = -float('inf') # Initialize to a very small number.
        
        # Average fitness of all dots in the current population.
        self.average_fitness: float = 0.0

    def update(self):
        """Updates all dots in the population for one frame."""
        # Simply loop through each dot and call its own update method.
        for dot in self.dots:
            dot.update() # Each dot handles its own movement, collisions, AI logic.

    def calculate_fitness(self):
        """Calculates the fitness for every dot in the population and updates overall population statistics."""
        # This function is called after all dots in the current generation/episode have finished their run
        # (e.g., all are dead or reached the goal).
        if not self.dots: # Safety check: if there are no dots, nothing to calculate!
            self.max_fitness = -float('inf')
            self.average_fitness = 0.0
            self.best_dot_object = None
            return

        total_fitness_for_generation = 0.0
        current_generation_max_fitness = -float('inf') # Reset for current generation's max
        current_generation_best_dot = None

        for dot in self.dots:
            dot.calculate_fitness() # Tell each dot to calculate its own fitness score.
            total_fitness_for_generation += dot.fitness

            # Check if this dot is the best one we've seen *in this current generation*.
            if dot.fitness > current_generation_max_fitness:
                current_generation_max_fitness = dot.fitness
                current_generation_best_dot = dot
        
        # After checking all dots, update population-level stats for this generation.
        self.max_fitness = current_generation_max_fitness
        self.best_dot_object = current_generation_best_dot 
        
        if len(self.dots) > 0:
            self.average_fitness = total_fitness_for_generation / len(self.dots)
        else:
            self.average_fitness = 0.0

        # If a best dot was found for this generation AND it reached the goal...
        if self.best_dot_object and self.best_dot_object.reached_goal:
            if SIMULATION_TYPE == SIM_TYPE_GENETIC: # Specific to Genetic Algorithm
                # Update min_step if this champion dot reached the goal faster (fewer brain steps).
                if self.best_dot_object.brain.step < self.min_step:
                     self.min_step = self.best_dot_object.brain.step
            # For RL, min_step might track something else, or not be used.

    def natural_selection(self): # Core of the Genetic Algorithm's "evolution"
        """Creates a new generation of dots based on the fitness of the current generation.
        This involves:
        1. Elitism: The best dot(s) from the current generation survive directly.
        2. Selection: Parents are chosen from the current generation (fitter dots have higher chance).
        3. Crossover (not explicitly here, but cloning is a form of it): Offspring inherit traits (brains).
        4. Mutation: Offspring might have small random changes to their traits (brains).
        """
        if not self.dots:
            log.warning("Attempted natural selection on an empty population.")
            return
        
        # Crucial: calculate_fitness() must have been called before this,
        # so self.best_dot_object is the champion of the *current* generation.
        if self.best_dot_object is None:
            log.warning("No best dot determined before natural selection. Trying to pick one or creating new population.")
            # Fallback: if no best dot, population might be all bad or it's an edge case.
            # If there are dots, pick the first one as a placeholder champion.
            if self.dots:
                self.best_dot_object = self.dots[0] # Not ideal, but keeps the process going.
                self.best_dot_object.calculate_fitness() # Ensure its fitness is known
                self.max_fitness = self.best_dot_object.fitness
                log.info(f"Fallback: selected first dot as best for natural selection. Fitness: {self.max_fitness}")
            else: # No dots at all, cannot proceed.
                return

        new_generation_dots: list[Dot] = [] # Start with an empty list for the new generation.

        # 1. Elitism: The very best dot from the current generation gets to survive directly.
        # This ensures that our best solution so far is never lost.
        champion_clone = self.best_dot_object.clone() # Make a copy of the best dot.
        champion_clone.reset() # Reset its state (like brain step) if it's carried over.
        champion_clone.is_best = True # Mark this one specially for display.
        new_generation_dots.append(champion_clone)

        # 2. Fill the rest of the new generation by selecting parents and creating offspring.
        for _ in range(1, self.population_size): # Start from 1 because champion is already added.
            parent_dot = self.select_parent() # "Roulette Wheel" selection based on fitness.
            
            if parent_dot:
                child_dot = parent_dot.clone() # Offspring is a clone of the selected parent.
                new_generation_dots.append(child_dot)
            else:
                # This might happen if all dots have zero fitness.
                log.warning("Parent selection returned None. Creating a new random dot as fallback.")
                # Create a brand new dot with default properties.
                fallback_dot = Dot(self.width, self.height, self.goal, self.obstacles)
                new_generation_dots.append(fallback_dot)
        
        self.dots = new_generation_dots # The new generation replaces the old one.
        self.gen += 1 # Increment the generation counter.
            
    def reset_population(self): # Used for Reinforcement Learning to start a new "episode"
        """Resets all dots for a new learning episode in Reinforcement Learning.
        Often, learned knowledge (like Q-tables) from the best dot is propagated.
        """
        if not self.dots and self.population_size > 0: # If dots list is empty but should not be
             log.warning("Dots list empty in reset_population, re-initializing.")
             self.dots = [Dot(self.width, self.height, self.goal, self.obstacles, is_rl_copy=True) for _ in range(self.population_size)]

        if not self.dots: # Still no dots (e.g. population_size is 0)
            log.error("Dot list is empty in reset_population and population_size is 0. Cannot proceed.")
            return

        # Determine the template dot for creating new dots.
        # This template will provide environmental parameters and, for RL, the Q-table to copy.
        template_dot_for_episode: Dot
        if self.best_dot_object and SIMULATION_TYPE == SIM_TYPE_REINFORCEMENT:
            template_dot_for_episode = self.best_dot_object
            log.info(f"Resetting RL population. Using Q-table from previous best dot (Fitness: {template_dot_for_episode.fitness:.2f}).")
        else:
            # If no best_dot_object (e.g., first episode) or not RL, create a fresh template.
            # For RL, this means starting with a fresh Q-table.
            log.info("Resetting RL population. No previous best dot or not RL mode. Creating fresh dots.")
            template_dot_for_episode = Dot(self.width, self.height, self.goal, self.obstacles, is_rl_copy=True)

        new_dots_for_episode: list[Dot] = []
        for i in range(self.population_size):
            new_single_dot = Dot(self.width, self.height, self.goal, self.obstacles, is_rl_copy=True)

            if SIMULATION_TYPE == SIM_TYPE_REINFORCEMENT:
                # All new dots in the episode inherit the Q-table from the template (often the previous best).
                # This allows knowledge to be shared and built upon.
                new_single_dot.q_table = template_dot_for_episode.q_table.copy() # Crucial: .copy() for independent Q-tables.
            
            new_dots_for_episode.append(new_single_dot)
        
        self.dots = new_dots_for_episode

        # The actual best dot from the previous episode (if one existed and it's RL)
        # can be preserved and reset to continue learning. This is a form of elitism.
        if self.best_dot_object and SIMULATION_TYPE == SIM_TYPE_REINFORCEMENT:
            self.dots[0] = self.best_dot_object # Put the actual best dot object back.
            self.dots[0].is_best = True      # Mark it as the current "champion".
            self.dots[0].reset()             # Reset its state for the new episode.
        elif self.dots: # Fallback if no prior best_dot_object, mark the first new one as 'best'.
            self.dots[0].is_best = True
            self.dots[0].reset() # Ensure it's reset.

        self.gen += 1 # Increment generation/episode counter.
            
    def select_parent(self) -> Dot | None: # Type hint indicates it returns a Dot or None
        """Selects a parent dot from the population using fitness-proportional selection (Roulette Wheel).
        Dots with higher fitness have a higher probability of being selected.
        """
        if not self.dots: # Safety check: no dots, no parent.
            return None

        # Calculate the sum of all positive fitness scores in the population.
        # We only want to select from dots that performed reasonably well.
        fitness_sum = sum(dot.fitness for dot in self.dots if dot.fitness > 0)

        if fitness_sum <= 0:
            # If all dots have zero or negative fitness (e.g., none moved or all died immediately),
            # we can't use fitness-proportional selection.
            # In this scenario, pick a random dot to keep the process going and hope for improvement.
            log.info("All dots have zero/negative fitness. Selecting a random parent.")
            return random.choice(self.dots) if self.dots else None

        # Pick a random number (the "pointer" on our roulette wheel) between 0 and the total fitness sum.
        pick = random.uniform(0, fitness_sum)
        
        running_sum = 0
        for dot in self.dots:
            if dot.fitness > 0: # Only consider dots that contributed to the positive fitness_sum.
                running_sum += dot.fitness
                if running_sum >= pick: # If our running sum has "crossed" the random pick value...
                    return dot # ...this dot is the chosen parent!
        
        # Fallback: This should ideally not be reached if fitness_sum > 0 and self.dots is not empty.
        # If it is, it might indicate an issue with fitness values or the selection logic.
        # As a safeguard, return a random dot from those with positive fitness, or any if none.
        log.warning("select_parent reached end without selection, returning a random positive-fitness dot or any dot.")
        positive_fitness_dots = [d for d in self.dots if d.fitness > 0]
        if positive_fitness_dots:
            return random.choice(positive_fitness_dots)
        return random.choice(self.dots) if self.dots else None


    def mutate(self): # Specific to Genetic Algorithm
        """Applies mutation to the brains of all dots in the population, except for the champion (elitism)."""
        # Mutation introduces small random changes, helping to explore new solutions.
        # We usually don't mutate the "best" dot (the first one, due to elitism in natural_selection)
        # to preserve the best solution found so far.
        for i in range(len(self.dots)):
            if self.dots[i].is_best: # Don't mutate the champion that was carried over.
                continue
            if hasattr(self.dots[i], 'brain') and self.dots[i].brain is not None:
                 self.dots[i].brain.mutate() # Tell each dot's brain to try mutating.

# --- 🗺️ OBSTACLE GENERATION ---
def generate_obstacles() -> list[pygame.Rect]:
    """Creates a list of randomly positioned and sized rectangular obstacles."""
    obstacles = []
    for _ in range(OBSTACLE_COUNT):
        # Ensure obstacles don't overlap too much with goal or start areas.
        # Obstacles are placed somewhat in the middle vertical section of the screen.
        # Random width and height within configured limits.
        w = random.randint(OBSTACLE_MIN_WIDTH, OBSTACLE_MAX_WIDTH)
        h = random.randint(OBSTACLE_MIN_HEIGHT, OBSTACLE_MAX_HEIGHT)
        
        # Random x position, ensuring obstacle fits on screen.
        x = random.randint(0, WIDTH - w)
        # Random y position, avoiding very top (goal area) and very bottom (start area).
        y_buffer = 50 # Space from top/bottom edges
        y = random.randint(y_buffer + GOAL_SIZE*2, HEIGHT - h - y_buffer) 
        
        obstacles.append(pygame.Rect(x, y, w, h)) # Pygame's Rect is handy for collision detection.
    log.info(f"Generated {len(obstacles)} obstacles.")
    return obstacles

# --- 💾 SIMULATION SAVE/LOAD ---
def save_simulation_state(population: Population, generation: int):
    """Saves the current state of the population (especially dot brains or Q-tables) to a file using pickle.
    This allows resuming the simulation or analyzing results later.
    """
    # Filename includes the AI type, timestamp, and current generation number.
    filename = os.path.join(SAVE_DIR, f"{SAVE_FILE_PREFIX}_G-{generation}.pkl")
    try:
        with open(filename, 'wb') as f: # 'wb' means write in binary mode (for pickle).
            # We are pickling the entire population object.
            # This works if all components of Population and Dot (like Brain, np.ndarray for Q-table) are picklable.
            pickle.dump(population, f)
        log.info(f"Simulation state saved to {filename}")
    except Exception as e:
        log.error(f"Error saving simulation state to {filename}: {e}")

# (Load function would be the counterpart, using pickle.load(f))

# --- 🎉 MAIN GAME FUNCTION ---
# This is where the simulation comes to life!
# It sets up Pygame, creates the population, and runs the main game loop.
def main():
    pygame.init() # Initialize all the Pygame modules (like display, font, time).
    
    # Set up the game window (screen).
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Dot AI Simulation - {SIMULATION_TEXT} Mode") # Window title.
    
    # Clock is used to control the frame rate (how fast the game runs).
    clock = pygame.time.Clock()
    
    # Set up fonts for displaying text information.
    # Using a default system font. None means Pygame picks one.
    try:
        font = pygame.font.SysFont(None, FONT_SIZE)
        font_note = pygame.font.SysFont(None, FONT_SIZE - 4) # Slightly smaller for notes
    except Exception as e: # Fallback if system font fails
        log.warning(f"SysFont not found, using default pygame font: {e}")
        font = pygame.font.Font(None, FONT_SIZE) # Pygame's default font
        font_note = pygame.font.Font(None, FONT_SIZE -4)


    # Create the obstacles for the dots to navigate around.
    obstacles = generate_obstacles()
    
    # Create the population of dots.
    population = Population(POPULATION_SIZE, WIDTH, HEIGHT, GOAL, obstacles)
    log.info(f"Initialized population with {POPULATION_SIZE} dots for {SIMULATION_TEXT} AI.")

    running = True # This variable controls the main game loop. When False, the game ends.

    # --- Main Game Loop ---
    # This loop runs continuously, once per frame, until the game is quit.
    # Each iteration handles events, updates game state, and draws everything.
    while running:
        # --- Event Handling ---
        # Check for any events that have happened (like keyboard presses, mouse clicks, window close).
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # If the user clicks the window's close button.
                log.info("Quit event received. Saving simulation...")
                save_simulation_state(population, population.gen)
                running = False # Exit the loop, ending the game.
            elif event.type == pygame.KEYDOWN: # If a key is pressed.
                if event.key == pygame.K_ESCAPE: # If the Escape key is pressed.
                    log.info("Escape key pressed. Saving simulation...")
                    save_simulation_state(population, population.gen)
                    running = False # Exit the loop.
                elif event.key == pygame.K_s: # If 'S' key is pressed.
                    log.info("'S' key pressed. Saving current simulation state...")
                    save_simulation_state(population, population.gen) # Save a snapshot.
            elif event.type == pygame.MOUSEBUTTONUP: # If a mouse button is released.
                # Could be used for interaction, e.g., placing obstacles or saving.
                # For now, let's make it save a snapshot too.
                log.info("Mouse click detected. Saving current simulation state...")
                save_simulation_state(population, population.gen)


        # --- Game Logic Update ---
        # Update the state of the population (all dots move, check collisions, etc.).
        population.update()

        # Check if all dots in the current generation/episode are "done"
        # (i.e., dead or have reached the goal).
        if all(dot.dead or dot.reached_goal for dot in population.dots):
            log.info(f"All dots finished for generation/episode {population.gen}.")
            # If all done, it's time to evaluate them and prepare for the next round.
            population.calculate_fitness() # Calculate fitness for every dot.
            
            log.info(f"Gen {population.gen}: Max Fitness: {population.max_fitness:.4f}, Avg Fitness: {population.average_fitness:.4f}")
            if population.best_dot_object:
                 log.info(f"Best dot reached goal: {population.best_dot_object.reached_goal}, Steps (GA): {population.best_dot_object.brain.step if hasattr(population.best_dot_object, 'brain') and population.best_dot_object.brain else 'N/A'}")

            # Save state at the end of each generation/episode.
            save_simulation_state(population, population.gen)

            if SIMULATION_TYPE == SIM_TYPE_GENETIC:
                # Genetic Algorithm: Perform natural selection and mutation to create the next generation.
                population.natural_selection()
                population.mutate() # Mutate the new generation (except the champion).
                log.info(f"Genetic Algorithm: Advanced to generation {population.gen}. Population mutated.")
            elif SIMULATION_TYPE == SIM_TYPE_REINFORCEMENT:
                # Reinforcement Learning: Reset the population for a new learning episode.
                # Learned Q-tables are often propagated from the best dot.
                population.reset_population()
                log.info(f"Reinforcement Learning: Population reset for episode {population.gen}.")
                
        # --- Drawing / Rendering ---
        # First, fill the screen with the background color to clear the previous frame.
        screen.fill(SCREEN_COLOR)

        # Draw the goal.
        pygame.draw.circle(screen, GOAL_COLOR, GOAL, GOAL_SIZE)

        # Draw all the obstacles.
        for obstacle in obstacles:
            pygame.draw.rect(screen, OBSTACLE_COLOR, obstacle)

        # Draw all the dots in the population.
        for dot in population.dots:
            # Use different color/size for the "best" dot to make it stand out.
            color_to_draw = BEST_DOT_COLOR if dot.is_best else DOT_COLOR
            size_to_draw = BEST_DOT_SIZE if dot.is_best else DOT_SIZE
            # Pygame needs integer coordinates for drawing.
            pygame.draw.circle(screen, color_to_draw, (int(dot.pos[0]), int(dot.pos[1])), size_to_draw)

        # --- Display Information (Metrics) ---
        # Calculate current metrics for display.
        # These are updated each frame, not just at generation end, for live view.
        current_max_fitness = population.max_fitness
        current_avg_fitness = population.average_fitness # This is from last completed gen
        
        # Live count of dots that reached goal in current run (might not be full population yet)
        dots_reached_goal_count = sum(1 for dot in population.dots if dot.reached_goal)
        percent_reached_goal = (dots_reached_goal_count / population.population_size * 100) if population.population_size > 0 else 0

        metrics_to_display = [
            f"Generation/Episode: {population.gen}",
            f"Max Fitness (last gen): {current_max_fitness:.4f}",
            f"Avg Fitness (last gen): {current_avg_fitness:.4f}",
            f"Dots Reached Goal (live): {dots_reached_goal_count}/{population.population_size} ({percent_reached_goal:.1f}%)"
        ]
        if SIMULATION_TYPE == SIM_TYPE_GENETIC:
            metrics_to_display.append(f"Min Steps (Champ): {population.min_step if population.min_step != BRAIN_SIZE else 'N/A'}")


        # Render each metric line at the bottom of the screen.
        for i, metric_text in enumerate(metrics_to_display):
            text_surface = font.render(metric_text, True, FONT_COLOR) # True for anti-aliasing (smoother text).
            # Position text lines from the bottom up.
            screen.blit(text_surface, (10, HEIGHT - ((len(metrics_to_display) - i) * (FONT_SIZE + 2)) - 10 ))
            
        # Display current AI mode and other notes at the top.
        info_texts = [
            f"Mode: {SIMULATION_TEXT} AI",
            f"S-Key or Click: Save Snapshot | Esc: Save & Exit",
            f"Champion Dot: {get_variable_name(BEST_DOT_COLOR)}",
            f"Goal: {get_variable_name(GOAL_COLOR)}"
        ]
        for i, note_text in enumerate(info_texts):
            note_surface = font_note.render(note_text, True, FONT_COLOR if i < 2 else (BEST_DOT_COLOR if "Champion" in note_text else GOAL_COLOR) )
            screen.blit(note_surface, (10, 10 + i * (FONT_SIZE -2)))
        
        # --- Update Display ---
        # After drawing everything, flip the display to show the new frame.
        pygame.display.flip()

        # --- Control Frame Rate ---
        # clock.tick(60) limits the game to 60 frames per second (FPS).
        # This makes the simulation run at a consistent speed on different computers.
        clock.tick(60)

    # --- End of Game ---
    # This code runs after the main loop (running = False).
    pygame.quit() # Uninitialize Pygame modules.
    log.info("Pygame quit. Simulation ended.")

# --- Script Execution ---
# This is a standard Python construct:
# If this script is run directly (not imported as a module into another script),
# then call the main() function to start the game.
if __name__ == "__main__":
    main()
