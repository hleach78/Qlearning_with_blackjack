# Qlearning_with_blackjack
A Python program that uses Q-learning to learn a policy for blackjack. Investigate how the algorithm behaves under different hyperparameter settings.

## Setup
1. Clone the repo to a local directory
2. Ensure python is installed
3. Run pip install "gymnasium[toy-text]" numpy matplotlib
4. Run pip install gymnasium[toy-text] numpy imageio imageio-ffmpeg pillow

## Train_agent.py
Implements a tabular Q-learning agent that learns a Blackjack strategy by choosing when to hit or stick. It trains and evaluates agents using different hyperparameter configurations to investigate how the learning rate, discount factor, and exploration schedule affect performance. The script trains and evaluates the agents, prints a performance comparison, and saves the output files in the current working directory.

### Output files
- blackjack_learning_curves.png — compares the smoothed training rewards across hyperparameter configurations.
- blackjack_baseline_rewards.png — plots the baseline agent's average reward against training episode.
- blackjack_q_table.npy — contains the Q-table for the best-performing agent.
  
## Record_best_agent.py
Loads the saved Q-table from the best performing Blackjack Q-learning agent and records the agent playing Blackjack. It uses the Gymnasium Blackjack environment to demonstrate how the trained policy chooses when to hit or stick.

### Output files
- blackjack_agent.mp4 — video recording of the trained agent playing Blackjack.
