import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("Agg")

def train_agent(
    episodes=200_000,
    alpha=0.01,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.99995,
    seed=42,
):

    env = gym.make("Blackjack-v1", sab=True)
    env.action_space.seed(seed)
    rng = np.random.default_rng(seed)

    q_table = np.zeros((32, 11, 2, 2))

    epsilon = epsilon_start
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            player, dealer, usable_ace = state
            state_index = (player, dealer, int(usable_ace))

            if rng.random() < epsilon:
                action = int(rng.integers(2))
            else:
                action = int(np.argmax(q_table[state_index]))

            next_state, reward, terminated, truncated, _ = (
                env.step(action)
            )

            next_player, next_dealer, next_ace = next_state
            next_index = (
                next_player,
                next_dealer,
                int(next_ace),
            )

            if terminated or truncated:
                target = reward
            else:
                target = reward + gamma * np.max(
                    q_table[next_index]
                )

            old_value = q_table[state_index + (action,)]
            q_table[state_index + (action,)] = (
                old_value + alpha * (target - old_value)
            )

            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    env.close()
    return q_table, np.array(rewards)


def evaluate_agent(q_table, games=50_000, seed=100_000):
    env = gym.make("Blackjack-v1", sab=True)
    wins = losses = draws = 0
    total_reward = 0.0

    for episode in range(games):
        state, _ = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        reward = 0.0

        while not (terminated or truncated):
            player, dealer, usable_ace = state
            index = (player, dealer, int(usable_ace))
            action = int(np.argmax(q_table[index]))

            state, reward, terminated, truncated, _ = (
                env.step(action)
            )

        total_reward += reward

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    env.close()

    return {
        "win_rate": wins / games,
        "loss_rate": losses / games,
        "draw_rate": draws / games,
        "avg_reward": total_reward / games,
    }


def run_experiment(label, **params):
    q_table, rewards = train_agent(**params)
    metrics = evaluate_agent(q_table)

    window = 5_000
    smoothed = np.convolve(
        rewards,
        np.ones(window) / window,
        mode="valid",
    )

    return {
        "label": label,
        "q_table": q_table,
        "rewards": rewards,
        "smoothed": smoothed,
        **metrics,
    }


if __name__ == "__main__":
    experiments = [
        {
            "label": "Low alpha",
            "alpha": 0.001,
            "gamma": 0.95,
            "epsilon_decay": 0.99995,
        },
        {
            "label": "Medium alpha",
            "alpha": 0.01,
            "gamma": 0.95,
            "epsilon_decay": 0.99995,
        },
        {
            "label": "High alpha",
            "alpha": 0.1,
            "gamma": 0.95,
            "epsilon_decay": 0.99995,
        },
        {
            "label": "Low gamma",
            "alpha": 0.01,
            "gamma": 0.5,
            "epsilon_decay": 0.99995,
        },
        {
            "label": "Medium gamma",
            "alpha": 0.01,
            "gamma": 0.75,
            "epsilon_decay": 0.99995,
        },
        {
            "label": "High gamma",
            "alpha": 0.01,
            "gamma": 0.99,
            "epsilon_decay": 0.99995,
        },
        {
            "label": "Fast decay",
            "alpha": 0.01,
            "gamma": 0.95,
            "epsilon_decay": 0.999,
        },
        {
            "label": "Medium decay",
            "alpha": 0.01,
            "gamma": 0.95,
            "epsilon_decay": 0.9999,
        },
        {
            "label": "Slow decay",
            "alpha": 0.01,
            "gamma": 0.95,
            "epsilon_decay": 0.99999,
        },
    ]

    results = []

    for experiment in experiments:
        experiment = experiment.copy()
        label = experiment.pop("label")
        print(f"Training: {label}", flush=True)

        result = run_experiment(
            label,
            episodes=200_000,
            **experiment,
        )
        results.append(result)

        print(
            f"  Win rate: {result['win_rate']:.2%} | "
            f"Loss rate: {result['loss_rate']:.2%} | "
            f"Draw rate: {result['draw_rate']:.2%} | "
            f"Average reward: {result['avg_reward']:.4f}",
            flush=True,
        )

    plt.figure(figsize=(12, 6))

    for result in results:
        plt.plot(
            result["smoothed"],
            label=result["label"],
        )

    plt.title("Blackjack Q-learning: Hyperparameter Comparison")
    plt.xlabel("Training episode")
    plt.ylabel("Average reward per game")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("blackjack_learning_curves.png", dpi=150)
    plt.close()

    print("\nFinal evaluation results")
    print("-" * 75)
    print(
        f"{'Experiment':<25} {'Win %':>8} "
        f"{'Loss %':>8} {'Draw %':>8} {'Avg reward':>12}"
    )

    for result in sorted(
        results,
        key=lambda x: x["avg_reward"],
        reverse=True,
    ):
        print(
            f"{result['label']:<25} "
            f"{result['win_rate']:>7.2%} "
            f"{result['loss_rate']:>7.2%} "
            f"{result['draw_rate']:>7.2%} "
            f"{result['avg_reward']:>12.4f}"
        )

    baseline_result = next(
        result for result in results
        if result["label"] == "Medium alpha"
    )

    baseline_alpha = 0.01
    baseline_gamma = 0.95
    baseline_epsilon_start = 1.0
    baseline_epsilon_end = 0.05
    baseline_epsilon_decay = 0.99995
    baseline_episodes = 200_000

    print("\nBASELINE CONFIGURATION")
    print("-" * 45)
    print(f"Learning rate (alpha):       {baseline_alpha}")
    print(f"Discount factor (gamma):     {baseline_gamma}")
    print(f"Initial epsilon:             {baseline_epsilon_start}")
    print(f"Minimum epsilon:              {baseline_epsilon_end}")
    print(f"Epsilon decay per episode:   {baseline_epsilon_decay}")
    print(f"Training episodes:           {baseline_episodes}")
    print(f"Random seed:                 42")

    print("\nBASELINE EVALUATION")
    print("-" * 45)
    print(f"Win rate:      {baseline_result['win_rate']:.2%}")
    print(f"Loss rate:     {baseline_result['loss_rate']:.2%}")
    print(f"Draw rate:     {baseline_result['draw_rate']:.2%}")
    print(f"Average reward: {baseline_result['avg_reward']:.4f}")

    window = 5_000
    baseline_rewards = baseline_result["rewards"]
    smoothed_rewards = baseline_result["smoothed"]
    episodes = np.arange(
        window,
        len(baseline_rewards) + 1
    )

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, smoothed_rewards, label="Baseline (5,000-episode average)")
    plt.axhline(y=0, linestyle="--", linewidth=1, label="Zero reward")
    plt.title("Baseline Blackjack Q-learning: Reward vs. Training Episode")
    plt.xlabel("Training episode")
    plt.ylabel("Average reward per episode")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("blackjack_baseline_rewards.png", dpi=150)
    plt.close()

    print("\nBaseline reward plot saved to blackjack_baseline_rewards.png")

    best_result = max(
        results,
        key=lambda r: r["avg_reward"],
    )

    np.save("blackjack_q_table.npy", best_result["q_table"])

    print(f"\nSaved best agent: {best_result['label']}")
    print(f"Average reward: {best_result['avg_reward']:.4f}")
    print("Learning curves saved to blackjack_learning_curves.png")
    print("Q-table saved to blackjack_q_table.npy")