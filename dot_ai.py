import pygame # Requires pip install
import random
import math
import pickle
import os
import logging
import datetime
import numpy as np

# GENERAL CONFIGURATION
SIMULATION_TYPE = "Reinforcement" # Player, Genetic, or Reinforcement
POPULATION_SIZE = 200
MAX_VELOCITY = 5

# GENETIC CONFIGURATION
MUTATION_RATE = 0.01
BRAIN_SIZE = 400

# REINFORCEMENT CONFIGURATION
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 0.5 # Exploration rate
GRID_SIZE = 10 # Coarser grid to reduce Q-table size

# Color Definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (225, 0, 0)
GREEN = (0, 250, 0)
BLUE = (0, 0, 250)

# Accessibility colors
#White: Instead of pure white, use a slightly off-white color to reduce eye strain.
WHITE_ACC = (224, 224, 224)
WHITE = WHITE_ACC # Override
#Black: Pure black can be harsh on the eyes, so a dark gray is often preferred.
BLACK_ACC = (18, 18, 18)
BLACK = BLACK_ACC # Override
#Red: Use a less saturated red to avoid harshness.
RED_ACC = (255, 107, 107)
RED = RED_ACC # Override
#Green: Opt for a softer green that maintains good contrast.
GREEN_ACC = (76, 175, 80)
GREEN = GREEN_ACC # Override
#Blue: Choose a blue that is not too vibrant to ensure readability.
BLUE_ACC = (66, 165, 245)
BLUE = BLUE_ACC # Override

# GAME UI CONFIGURATION
WIDTH, HEIGHT = 800, 600 # Size of screen
SCREEN_COLOR = BLACK
FONT_SIZE = 20
FONT_COLOR = GREEN

# DOT CONFIGURATION
DOT_SIZE = 2
DOT_COLOR = BLUE
BEST_DOT_SIZE = 4 # Larger size so its more visible
BEST_DOT_COLOR = RED # Highlight color for best dot

# GOAL CONFIGURATION
GOAL = (WIDTH // 2, 25) # Goal placement x, y
GOAL_SIZE = 4
GOAL_COLOR = GREEN

# OBSTACLE CONFIGURATION
OBSTACLE_COUNT = 5
OBSTACLE_COLOR = WHITE
OBSTACLE_MIN_WIDTH = 0
OBSTACLE_MAX_WIDTH = 200
OBSTACLE_MIN_HEIGHT = 0
OBSTACLE_MAX_HEIGHT = 250

# Misc Config - Shouldn't need to update this stuff or below
# Make the sim type easier to code
if SIMULATION_TYPE == "Genetic":
    SIMULATION_TYPE = 1
elif SIMULATION_TYPE == "Reinforcement":
    SIMULATION_TYPE = 2
else:
    SIMULATION_TYPE = 0 # Player Control

LOG_LEVEL = logging.INFO
SAVE_DIR = ".\simulation_saves"
if SIMULATION_TYPE == 1: # Genetic
    SAVE_FILE = "genetic_dot_ai_results_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
elif SIMULATION_TYPE == 2: # Reinforcement
    SAVE_FILE = "reinforcement_dot_ai_results_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
# Function needed to get variable color names
def get_variable_name(var):
    # Get the global symbol table
    global_vars = globals()
    # Iterate over the global variables
    for name, value in global_vars.items():
        if value is var:
            return name
    return None

# Set up logs
log = logging.getLogger('dot_logger')
log.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

# Ensure the save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

class Brain:
    def __init__(self, size):
        self.directions = [self.random_vector() for _ in range(size)]
        self.step = 0

    def random_vector(self):
        """Generate a random direction vector."""
        angle = random.uniform(0, 2 * math.pi)
        return (math.cos(angle), math.sin(angle))

    def clone(self):
        """Clone the brain."""
        clone = Brain(len(self.directions))
        clone.directions = self.directions[:]
        return clone

    def mutate(self):
        """Mutate the brain's directions."""
        for i in range(len(self.directions)):
            if random.random() < MUTATION_RATE:
                self.directions[i] = self.random_vector()

class Dot:
    def __init__(self, width, height, goal, obstacles, is_copy=False):
        self.pos = [width / 2, height - 10]
        self.vel = [0, 0]
        self.acc = [0, 0]
        self.dead = False
        self.reached_goal = False
        self.fitness = 0
        self.width = width
        self.height = height
        self.goal = goal
        self.obstacles = obstacles
        self.is_best = False
        if SIMULATION_TYPE == 1: # Genetic
            self.brain = Brain(BRAIN_SIZE)
            self.is_copy = False
        elif SIMULATION_TYPE == 2: # Reinforcement
            self.is_copy = is_copy
            self.q_table = np.zeros((width // GRID_SIZE, height // GRID_SIZE, 8))  # Q-table for Q-learning
            self.state = (int(self.pos[0]) // GRID_SIZE, int(self.pos[1]) // GRID_SIZE)
            self.action = None
            self.previous_positions = []

    def move(self):
        """Move the dot according to its directions."""
        self.vel[0] += self.acc[0]
        self.vel[1] += self.acc[1]
        self.limit_velocity(MAX_VELOCITY)
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        
        if SIMULATION_TYPE == 1: # Genetic
            if len(self.brain.directions) > self.brain.step:
                self.acc = self.brain.directions[self.brain.step]
                self.brain.step += 1
            else:
                self.dead = True
        elif SIMULATION_TYPE == 2: # Reinforcement
            if self.action is None:
                self.action = self.choose_action()
                
            self.acc = self.action_to_vector(self.action)
            self.state = (int(self.pos[0]) // GRID_SIZE, int(self.pos[1]) // GRID_SIZE)
            self.previous_positions.append((int(self.pos[0]), int(self.pos[1])))

    def limit_velocity(self, max_velocity):
        """Limit the velocity of the dot."""
        speed = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
        if speed > max_velocity:
            self.vel[0] = (self.vel[0] / speed) * max_velocity
            self.vel[1] = (self.vel[1] / speed) * max_velocity

    def update(self):
        """Update the dot's position and check for collisions."""
        if not self.dead and not self.reached_goal:
            self.move()
            
            if SIMULATION_TYPE == 2: # Reinforcement
                reward = self.calculate_reward()
                next_state = (int(self.pos[0]) // GRID_SIZE, int(self.pos[1]) // GRID_SIZE)
                if 0 <= next_state[0] < self.q_table.shape[0] and 0 <= next_state[1] < self.q_table.shape[1]:
                    self.update_q_table(self.state, self.action, reward, next_state)
                    self.state = next_state
                    self.action = self.choose_action()

            # Check for collisions with walls
            if self.pos[0] < 2 or self.pos[1] < 2 or self.pos[0] > self.width - 2 or self.pos[1] > self.height - 2:
                self.dead = True
            # Check if goal is reached
            elif self.distance(self.pos, self.goal) < 5:
                self.reached_goal = True
            # Check for collisions with obstacles
            else:
                for obstacle in self.obstacles:
                    if obstacle.collidepoint(self.pos):
                        self.dead = True
                        break

    def calculate_fitness(self):
        """Calculate the fitness of the dot."""
        if SIMULATION_TYPE == 1: # Genetic
            if self.reached_goal:
                self.fitness = 1.0 / 16.0 + 10000.0 / (self.brain.step * self.brain.step)
            else:
                distance_to_goal = self.distance(self.pos, self.goal)
                self.fitness = 1.0 / (distance_to_goal * distance_to_goal)
        elif SIMULATION_TYPE == 2: # Reinforcement    
            self.fitness = 1 / (self.distance(self.pos, self.goal) + 1)

    def calculate_reward(self):
        """Calculate the reward or punishment for the dot."""
        if self.reached_goal:
            return 100
        elif self.dead:
            return -100
        else:
            distance_to_goal = self.distance(self.pos, self.goal)
            return -distance_to_goal
        
    def distance(self, pos1, pos2):
        """Calculate the distance between two points."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def clone(self):
        """Clone the dot."""
        clone = Dot(self.width, self.height, self.goal, self.obstacles)
        clone.brain = self.brain.clone()
        return clone

    def choose_action(self):
        """Select an action based on reinforced learning rewards."""
        if random.uniform(0, 1) < EPSILON:
            return random.randint(0, 7)
        else:
            return np.argmax(self.q_table[self.state[0], self.state[1]])

    def action_to_vector(self, action):
        angle = action * (math.pi / 4)
        return (math.cos(angle), math.sin(angle))

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
        td_target = reward + DISCOUNT_FACTOR * self.q_table[next_state[0], next_state[1], best_next_action]
        td_error = td_target - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += LEARNING_RATE * td_error

    def reset(self):
        self.pos = [self.width / 2, self.height - 10]
        self.vel = [0, 0]
        self.acc = [0, 0]
        self.dead = False
        self.reached_goal = False
        self.state = (int(self.pos[0]) // GRID_SIZE, int(self.pos[1]) // GRID_SIZE)
        self.action = None
        self.previous_positions = []
        
class Population:
    def __init__(self, size, width, height, goal, obstacles):
        self.dots = [Dot(width, height, goal, obstacles) for _ in range(size)]
        self.gen = 1
        self.min_step = 400
        self.best_dot = 0
        self.max_fitness = 0
        self.average_fitness = 0
        self.best_dot = None

    def update(self):
        """Update all dots in the population."""
        for dot in self.dots:
            dot.update()

    def calculate_fitness(self):
        """Calculate fitness for all dots."""
        total_fitness = 0
        for dot in self.dots:
            dot.calculate_fitness()
            total_fitness += dot.fitness
            if dot.fitness > self.max_fitness:
                self.max_fitness = dot.fitness
                self.best_dot = dot
        self.average_fitness = total_fitness / len(self.dots)
        
    def natural_selection(self):
        """Perform natural selection to create a new generation."""
        new_dots = [None] * len(self.dots)
        self.set_best_dot()
        new_dots[0] = self.dots[self.best_dot].clone()
        new_dots[0].is_best = True
        for i in range(1, len(new_dots)):
            parent = self.select_parent()
            new_dots[i] = parent.clone()
        self.dots = new_dots
        self.gen += 1
        
    def reset_population(self):
        best_dot = self.best_dot
        self.dots = [Dot(best_dot.width, best_dot.height, best_dot.goal, best_dot.obstacles) for _ in range(POPULATION_SIZE)]
        self.dots[0] = best_dot
        self.dots[0].is_best = True
        self.dots[0].reset()
        for dot in self.dots[1:]:
            dot.q_table = best_dot.q_table.copy()
        self.gen += 1
            
    def select_parent(self):
        """Select a parent based on fitness."""
        fitness_sum = sum(dot.fitness for dot in self.dots)
        rand = random.uniform(0, fitness_sum)
        running_sum = 0
        for dot in self.dots:
            running_sum += dot.fitness
            if running_sum > rand:
                return dot
        return None

    def mutate(self):
        """Mutate all dots except the best ones clone."""
        for i in range(1, len(self.dots)):
            self.dots[i].brain.mutate()

    def set_best_dot(self):
        """Set the best dot based on fitness."""
        max_fitness = 0
        average_fitness = 0
        dot_count = 0
        max_index = 0
        for i, dot in enumerate(self.dots):
            average_fitness += dot.fitness
            dot_count += 1
            if dot.fitness > max_fitness:
                max_fitness = dot.fitness
                max_index = i
        self.best_dot = max_index
        self.max_fitness = max_fitness
        self.average_fitness = average_fitness / dot_count
        if self.dots[self.best_dot].reached_goal:
            self.min_step = self.dots[self.best_dot].brain.step

def generate_obstacles():
    """Generate random obstacles for the map."""
    obstacles = []
    for _ in range(OBSTACLE_COUNT):
        x = random.randint(0, WIDTH - OBSTACLE_MAX_WIDTH)
        y = random.randint(100, HEIGHT - OBSTACLE_MAX_HEIGHT - 100)
        w = random.randint(OBSTACLE_MIN_WIDTH, OBSTACLE_MAX_WIDTH)
        h = random.randint(OBSTACLE_MIN_HEIGHT, OBSTACLE_MAX_HEIGHT)
        obstacles.append(pygame.Rect(x, y, w, h))
    return obstacles

def save_simulation(population, generation):
    """Save the current state of the simulation."""
    filename = os.path.join(SAVE_DIR, f"{SAVE_FILE}_G-{generation}.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(population, f)
    log.info(f"Simulation state saved to {filename}")

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, FONT_SIZE)
    font_note = pygame.font.SysFont(None, FONT_SIZE-2)
    obstacles = generate_obstacles()
    population = Population(POPULATION_SIZE, WIDTH, HEIGHT, GOAL, obstacles)
    running = True

    # Initialize metrics
    best_fitness = 0
    average_fitness = 0
    dots_reached_goal = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_simulation(population, population.gen)
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                save_simulation(population, population.gen)
                running = False
            elif event.type == pygame.MOUSEBUTTONUP:
                save_simulation(population, population.gen)

        screen.fill(SCREEN_COLOR)
        pygame.draw.circle(screen, GOAL_COLOR, GOAL, GOAL_SIZE)

        # Draw obstacles
        for obstacle in obstacles:
            pygame.draw.rect(screen, OBSTACLE_COLOR, obstacle)

        # Update population and draw dots
        population.update()

        for dot in population.dots:
            color = BEST_DOT_COLOR if dot.is_best else DOT_COLOR
            size = BEST_DOT_SIZE if dot.is_best else DOT_SIZE
            pygame.draw.circle(screen, color, (int(dot.pos[0]), int(dot.pos[1])), size)

        # Check if all dots are dead or have reached the goal
        if all(dot.dead or dot.reached_goal for dot in population.dots):
            population.calculate_fitness()
            if SIMULATION_TYPE == 1: # Genetic
                population.natural_selection()
                population.mutate()
            elif SIMULATION_TYPE == 2: # Reinforcement
                population.reset_population()
                
        # Calculate metrics
        best_fitness = population.max_fitness
        average_fitness = population.average_fitness
        dots_reached_goal = sum(dot.reached_goal for dot in population.dots)

        log.debug(f'Generation: {population.gen}')
        log.debug(f'Best Fitness: {best_fitness:.4f}')
        log.debug(f'Average Fitness: {average_fitness:.4f}')
        log.debug(f'Dots Reached Goal: {dots_reached_goal}')

        # Render metrics
        metrics = [
            f"Generation: {population.gen}",
            f"Best Fitness: {best_fitness:.4f}",
            f"Average Fitness: {average_fitness:.4f}",
            f"Dots Reached Goal: {round(dots_reached_goal/POPULATION_SIZE*100)}%"
        ]

        text_location = 0
        for i, metric in enumerate(metrics):
            text_location += i
            text_surface = font.render(metric, True, FONT_COLOR)
            screen.blit(text_surface, (10, HEIGHT - (20 + i * FONT_SIZE)))
        text_surface = font_note.render(f"(Click for Snapshot. Esc to Exit)", True, FONT_COLOR)
        screen.blit(text_surface, (5, FONT_SIZE))
        text_surface = font_note.render(f"(Previous best is {get_variable_name(BEST_DOT_COLOR)})", True, BEST_DOT_COLOR)
        screen.blit(text_surface, (5, FONT_SIZE * 2))
        text_surface = font_note.render(f"(Goal is {get_variable_name(GOAL_COLOR)})", True, GOAL_COLOR)
        screen.blit(text_surface, (5, FONT_SIZE * 3))
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
