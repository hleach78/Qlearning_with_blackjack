import gymnasium as gym
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

q_table = np.load("blackjack_q_table.npy")

print("Using saved Q-learning agent")

NUM_GAMES = 30
FPS = 4
VIDEO_PATH = "blackjack_agent.mp4"

env = gym.make(
    "Blackjack-v1",
    render_mode="rgb_array",
    sab=True
)

frames = []
font = ImageFont.load_default()
rng = np.random.default_rng()

def capture_frame(lines, repeats=8):
    image = Image.fromarray(env.render()).convert("RGB")
    draw = ImageDraw.Draw(image)

    panel_height = 8 + 20 * len(lines)
    draw.rectangle(
        [(0, 0), (image.width, panel_height)],
        fill=(25, 30, 40)
    )

    for i, line in enumerate(lines):
        draw.text(
            (8, 5 + i * 20),
            line,
            fill=(255, 255, 255),
            font=font
        )

    frame = np.array(image)
    frames.extend([frame] * repeats)


game_results = []

for game_number in range(1, NUM_GAMES + 1):
    state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    terminated = truncated = False
    total_reward = 0

    capture_frame([
        "BLACKJACK Q-LEARNING AGENT",
        f"GAME {game_number} OF {NUM_GAMES}",
        "New hand starting..."
    ], repeats=12)

    while not (terminated or truncated):
        player, dealer, usable_ace = state

        q_values = q_table[player, dealer, int(usable_ace)]
        action = int(np.argmax(q_values))
        action_name = "STICK" if action == 0 else "HIT"

        capture_frame([
            f"GAME {game_number} OF {NUM_GAMES}",
            f"Player total: {player} | Dealer card: {dealer}",
            f"Usable ace: {usable_ace}",
            f"Decision: {action_name}",
            f"Q-values (Stick, Hit): {q_values.round(2)}"
        ])

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    if total_reward > 0:
        outcome = "WIN"
    elif total_reward < 0:
        outcome = "LOSS"
    else:
        outcome = "DRAW"

    game_results.append(total_reward)

    capture_frame([
        f"GAME {game_number} COMPLETE",
        f"Result: {outcome}",
        f"Reward: {total_reward:+.0f}",
        f"Score: {sum(r > 0 for r in game_results)} wins, "
        f"{sum(r < 0 for r in game_results)} losses"
    ], repeats=16)

env.close()

imageio.mimsave(
    VIDEO_PATH,
    frames,
    fps=FPS,
    macro_block_size=1
)

print(f"\nVideo saved: {VIDEO_PATH}")
print(f"Games recorded: {NUM_GAMES}")
print(f"Total wins: {sum(r > 0 for r in game_results)}")
print(f"Total losses: {sum(r < 0 for r in game_results)}")
print(f"Total draws: {sum(r == 0 for r in game_results)}")
print(f"Average reward: {np.mean(game_results):.3f}")
