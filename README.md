# AI Dot Simulation: Genetic vs. Reinforcement Learning

This project showcases the fascinating world of Artificial Intelligence through a visual simulation built with Pygame. It allows you to observe a "dot" learning to navigate an environment, reach a goal, and avoid obstacles using two distinct AI paradigms: **Genetic Algorithms** and **Reinforcement Learning**.

## 🚀 Features

* **Interactive Simulation:** Watch dots learn and adapt in real-time.
* **Genetic Algorithm Mode:** Observe an evolutionary process where dots "mutate" and "naturally select" the best strategies to reach the goal.
* **Reinforcement Learning Mode:** See a single dot learn optimal actions through trial and error, leveraging Q-learning to navigate the environment and maximize rewards.
* **Configurable Environment:** Easily adjust simulation parameters like population size, obstacle count, and learning rates.
* **Performance Metrics:** Track generations, fitness scores, and goal completion rates.

## 🧠 How it Works

The core of this simulation lies in its intelligent "dots." Depending on the `SIMULATION_TYPE` configured:

* **Genetic Mode:** A population of dots attempts to reach the goal. Dots that perform better (e.g., get closer or reach the goal faster) are more likely to pass on their "brains" (a sequence of movement directions) to the next generation, with slight mutations to encourage exploration.
* **Reinforcement Mode:** A single dot learns by interacting with the environment. It uses a Q-table to store the "value" of taking specific actions in different states. Through rewards (for reaching the goal) and penalties (for hitting obstacles or walls), the dot iteratively refines its Q-table, leading to optimized pathfinding.

## ⚙️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/ai-dot-simulation.git](https://github.com/your-username/ai-dot-simulation.git)
    cd ai-dot-simulation
    ```
2.  **Install dependencies:**
    ```bash
    pip install pygame numpy
    ```
3.  **Run the simulation:**
    ```bash
    python your_main_script_name.py
    ```
    *(Note: Replace `your_main_script_name.py` with the actual name of your Python file, e.g., `main.py` or `simulation.py`)*

## 🛠️ Configuration

You can easily switch between simulation types and adjust parameters at the top of the main script:

```python
# GENERAL CONFIGURATION
SIMULATION_TYPE = "Reinforcement" # Player, Genetic, or Reinforcement
POPULATION_SIZE = 200
MAX_VELOCITY = 5

# GENETIC CONFIGURATION (relevant when SIMULATION_TYPE = "Genetic")
MUTATION_RATE = 0.01
BRAIN_SIZE = 400

# REINFORCEMENT CONFIGURATION (relevant when SIMULATION_TYPE = "Reinforcement")
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 0.5 # Exploration rate
GRID_SIZE = 10 # Coarser grid to reduce Q-table size
