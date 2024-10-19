# Deep Q-Network Agent for Connect Four

The code implements a reinforcement learning approach where the AI Agent learns to play Connect Four by interacting with the game environment. It uses a neural network to approximate the Q-values for each possible action, allowing it to make informed decisions based on the current state of the game.

Here's a simplified explanation of the code:

The ConnectFourEnv class represents the game environment of Connect Four. It defines the game board, rules and how the game progresses.

The ReplayBuffer class stores past experiences for the Agent to learn from during training.

The DQNAgent class defines the AI Agent that learns to play Connect Four using a neural network.

The train function trains the AI Agent by playing multiple episodes of the game against a random opponent.

The play function allows a human player to play against the trained AI Agent.

### Dependencies

* Python 3.10

```python
pip install numpy tensorflow matplotlib
```
