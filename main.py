import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import os

# Define constants for reward values
WIN_REWARD = 1
LOSS_REWARD = -1
INVALID_MOVE_REWARD = -10
DRAW_REWARD = 0


class ConnectFourEnv:
    def __init__(self):
        self.rows = 6
        self.columns = 7
        self.board = np.zeros((self.rows, self.columns), dtype=int)

    def reset(self):
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        return self.board.copy()

    def step(self, action, player):
        if not self.is_valid_action(action):
            return self.board.copy(), INVALID_MOVE_REWARD, True, {}
        row = self.get_next_open_row(action)
        self.board[row][action] = player

        if self.check_win(player):
            reward = WIN_REWARD
            done = True
        elif self.is_board_full():
            reward = DRAW_REWARD
            done = True
        else:
            reward = 0
            done = False

        return self.board.copy(), reward, done, {}

    def is_valid_action(self, action):
        return self.board[0][action] == 0

    def get_next_open_row(self, action):
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][action] == 0:
                return r

    def is_board_full(self):
        return np.all(self.board[0] != 0)

    def check_win(self, player):
        for c in range(self.columns - 3):
            for r in range(self.rows):
                if all(self.board[r, c + i] == player for i in range(4)):
                    return True
        for c in range(self.columns):
            for r in range(self.rows - 3):
                if all(self.board[r + i, c] == player for i in range(4)):
                    return True
        for c in range(self.columns - 3):
            for r in range(self.rows - 3):
                if all(self.board[r + i, c + i] == player for i in range(4)):
                    return True
        for c in range(self.columns - 3):
            for r in range(3, self.rows):
                if all(self.board[r - i, c + i] == player for i in range(4)):
                    return True
        return False

    def get_valid_actions(self):
        return [c for c in range(self.columns) if self.is_valid_action(c)]

    def render(self):
        print(np.flip(self.board, 0))


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

    def size(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995  # Decay rate
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = ReplayBuffer()
        self.train_start = 1000  # Start training after 1000 experiences
        self.update_target_freq = 200  # Updated target network frequency

        # Build networks
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        # Internal step counter
        self.step_count = 0

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=self.state_shape))
        model.add(layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'))
        model.add(layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, valid_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        masked_q_values = np.full(self.action_size, -np.inf)
        masked_q_values[valid_actions] = q_values[valid_actions]
        return np.argmax(masked_q_values)

    def remember(self, state, action, reward, next_state, done):
        if action is not None and 0 <= action < self.action_size:
            self.memory.add((state, action, reward, next_state, done))

    def replay(self):
        if self.memory.size() < self.train_start:
            return

        minibatch = self.memory.sample(self.batch_size)
        states = np.array([sample[0] for sample in minibatch]).astype('float32')
        actions = np.array([sample[1] for sample in minibatch])
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.array([sample[3] for sample in minibatch]).astype('float32')
        dones = np.array([sample[4] for sample in minibatch])

        target = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.max(target_next[i])

        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.update_target_model()

    def save(self, name):
        self.model.save(name)
        print(f"Model saved to {name}")

    def load(self, name):
        self.model = models.load_model(name)
        self.update_target_model()
        print(f"Model loaded from {name}")


def get_user_action(valid_actions, max_column):
    while True:
        try:
            action = int(input(f"Enter your move (column 0-{max_column}): "))
            if action in valid_actions:
                return action
            else:
                print("Invalid action. Please choose from available columns:", valid_actions)
        except ValueError:
            print("Invalid input. Please enter an integer.")


def train(agent, env, num_episodes, model_filename):
    win_rates = []
    wins = 0
    for episode in range(num_episodes):
        state = env.reset()
        state = np.expand_dims(state, axis=2).astype('float32')
        done = False

        while not done:
            # Agent's move
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions)
            next_state_agent, reward_agent, done_agent, _ = env.step(action, player=1)
            next_state_agent_expanded = np.expand_dims(next_state_agent, axis=2).astype('float32')

            # Check if agent wins
            if done_agent:
                agent.remember(state, action, reward_agent, next_state_agent_expanded, done_agent)
                agent.replay()
                if reward_agent == WIN_REWARD:
                    wins += 1
                break

            # Opponent's move
            opponent_valid_actions = env.get_valid_actions()
            opponent_action = random.choice(opponent_valid_actions)
            next_state_opponent, reward_opponent, done_opponent, _ = env.step(opponent_action, player=-1)
            next_state_opponent_expanded = np.expand_dims(next_state_opponent, axis=2).astype('float32')

            # If opponent wins
            if done_opponent:
                agent.remember(state, action, LOSS_REWARD, next_state_opponent_expanded, done_opponent)
                agent.replay()
                break

            # Check for draw
            if env.is_board_full():
                agent.remember(state, action, DRAW_REWARD, next_state_opponent_expanded, True)
                agent.replay()
                break

            # Store experience with cumulative reward
            cumulative_reward = reward_agent + reward_opponent
            agent.remember(state, action, cumulative_reward, next_state_opponent_expanded, False)
            agent.replay()

            # Update state
            state = next_state_opponent_expanded

        # Calculate win rate
        win_rate = wins / (episode + 1)
        win_rates.append(win_rate)

        if episode % 100 == 0:
            print(f"Episode {episode}, Win Rate: {win_rate:.2f}, Epsilon: {agent.epsilon:.2f}")

    # Save the trained model
    agent.save(model_filename)

    # Plot the win rates
    plt.plot(win_rates)
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title('Training Progress')
    plt.show()


def play(agent, env):
    agent.epsilon = 0.0  # Disable exploration
    print("Starting a game of Connect Four!")
    print("You are player -1 (Your pieces are represented by -1).")

    state = env.reset()
    state = np.expand_dims(state, axis=2).astype('float32')
    done = False
    env.render()
    print("\n")

    while not done:
        # User's turn
        valid_actions = env.get_valid_actions()
        user_action = get_user_action(valid_actions, env.columns - 1)
        next_state, reward, done, _ = env.step(user_action, player=-1)
        next_state_expanded = np.expand_dims(next_state, axis=2).astype('float32')
        env.render()
        print("\n")

        if done:
            if reward == WIN_REWARD:
                print("You win!")
            elif reward == INVALID_MOVE_REWARD:
                print("Invalid move. You lose.")
            elif reward == DRAW_REWARD:
                print("It's a draw.")
            else:
                print("Game over.")
            break

        # Agent's turn
        state = next_state_expanded
        valid_actions = env.get_valid_actions()
        action = agent.act(state, valid_actions)
        next_state, reward, done, _ = env.step(action, player=1)
        next_state_expanded = np.expand_dims(next_state, axis=2).astype('float32')
        print("Agent's move:")
        env.render()
        print("\n")

        if done:
            if reward == WIN_REWARD:
                print("Agent wins!")
            elif reward == INVALID_MOVE_REWARD:
                print("Agent made an invalid move. You win!")
            elif reward == DRAW_REWARD:
                print("It's a draw.")
            else:
                print("Game over.")
            break

        state = next_state_expanded


def main():
    env = ConnectFourEnv()
    state_shape = (env.rows, env.columns, 1)
    action_size = env.columns
    agent = DQNAgent(state_shape, action_size)
    model_filename = "dqn_connect4.keras"

    print("Select an option:")
    print("1. Train the model")
    print("2. Play against the model")
    choice = input("Enter 1 or 2: ")

    if choice == '1':
        num_episodes = int(input("Enter the number of training episodes: "))
        train(agent, env, num_episodes, model_filename)
    elif choice == '2':
        if os.path.exists(model_filename):
            agent.load(model_filename)
            play(agent, env)
        else:
            print("No trained model found. Please train the model first.")
    else:
        print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
