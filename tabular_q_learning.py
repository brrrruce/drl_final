# -*- coding: utf-8 -*-
# Tabular-Based Q-Learning for 3×3, 3 色消除遊戲
# 訓練環境：Google Colab
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from env import TurnEnv, count_combos

grid_size = 3
num_colors = 3
env = TurnEnv(size=grid_size, colors=num_colors, max_steps=100, min_combos=2)


def serialize_state(env):
    # 將 board（numpy array）轉為 tuple
    board_tuple = tuple(env.board.flatten().tolist())
    # phase（1 或 2）
    phase = env.phase
    # held 位置，如果為 None，則用 (-1,-1)
    if env.held is None:
        held_tuple = (-1, -1)
    else:
        held_tuple = env.held
    # 組合成最終鍵
    return (board_tuple, phase, held_tuple)

# 初始化 Q-table，使用 defaultdict，預設 Q 值為 0
Q = defaultdict(lambda: np.zeros(env.action_dim, dtype=np.float32))
state_action_counts = defaultdict(lambda: np.zeros(env.action_dim, dtype=np.int32))
state_counts = defaultdict(int)

# ε-greedy 策略
def choose_action(state_key, valid_mask, epsilon):
    if random.random() < epsilon:
        # 隨機從合法動作中選擇
        end = env.size * env.size + 4
        valid_actions = list(np.where(valid_mask > 0)[0])
        """
        if end in valid_actions:
            valid_actions.remove(end)
        """
        return int(np.random.choice(valid_actions))
    else:
        # 選擇 Q 值最大的合法動作
        q_values = Q[state_key]
        # 對不合法動作設為極小值，避免被選中
        masked_q = np.copy(q_values)
        masked_q[valid_mask == 0] = -np.inf
        return int(np.argmax(masked_q))

def choose_action_exp(state_key, valid_mask, step, exp_step):
    if step < exp_step:
        # 隨機從合法動作中選擇
        end = env.size * env.size + 4
        valid_actions = list(np.where(valid_mask > 0)[0])
        return int(np.random.choice(valid_actions))
    else:
        # 選擇 Q 值最大的合法動作
        q_values = Q[state_key]
        # 對不合法動作設為極小值，避免被選中
        masked_q = np.copy(q_values)
        masked_q[valid_mask == 0] = -np.inf
        return int(np.argmax(masked_q))

def choose_action_ucb1(state_key, valid_mask, c=2.5):
    """
    UCB1:
        a* = argmax_a [ Q(s,a) + c * sqrt( ln(N(s)+1) / (1 + N(s,a)) ) ]
    其中
        N(s)      : 該 state 造訪次數
        N(s,a)    : 在該 state 選擇動作 a 的次數
        valid_mask: 長度 = env.action_dim，合法動作為 1，其餘 0
    """
    # -------- 基本查表 --------
    N_s     = state_counts[state_key]          # int
    q_values = Q[state_key]                    # ndarray shape = (action_dim,)
    counts   = state_action_counts[state_key]  # ndarray 同上

    # -------- 首次造訪：先更新計數，再隨機探索 --------
    if N_s == 0:
        state_counts[state_key] = 1            # 立即 +1，避免下一步 ln(1)=0
        valid_actions = np.flatnonzero(valid_mask)
        a = int(np.random.choice(valid_actions))
        counts[a] += 1                         # 這一步也順便記 1 次
        return a

    # -------- 計算 UCB 值 --------
    log_N = np.log(N_s + 1)                    # +1 防止 ln(1)=0
    ucb_values = np.full_like(q_values, -np.inf, dtype=np.float32)

    for a in range(env.action_dim):
        if valid_mask[a] == 0:
            continue                           # 非法動作保持 -inf
        if counts[a] == 0:
            ucb_values[a] = np.inf             # 強制先探索每個動作一次
        else:
            bonus = c * np.sqrt(log_N / (1.0 + counts[a]))
            ucb_values[a] = q_values[a] + bonus

    # -------- 決策 --------
    return int(np.argmax(ucb_values))

# Q-Learning 超參數
num_episodes = 4000000           # 訓練回合數
max_steps_per_episode = 100    # 每回合最多步數 (同 env.max_steps)
alpha = 0.1                    # 學習率
gamma = 0.999                   # 折扣因子
exp_step = 1500000              # 探索的回合數

# 用於追蹤訓練過程的指標
episode_rewards = []
episode_com = []
ep_len = []
mv_avg_c = []
mv_avg_r = []
epsilon_start = 1.0            # ε 初始值
epsilon_end = 0.01          # ε 最小值
epsilon_decay = 0.0000002      # ε 衰減速率

stepp = 0
for episode in range(1, num_episodes + 1):
    stepp += 1
    # 每回合開始：重置環境
    state_np = env.reset()
    state_key = serialize_state(env)
    total_reward = 0.0
    epsilon = 1 #max(epsilon_end, epsilon_start - stepp * epsilon_decay)

    for step in range(max_steps_per_episode):
        # 取得當前合法動作遮罩
        valid_mask = env._valid_mask()
        # 選擇動作
        action = choose_action_ucb1(state_key, valid_mask)

        # 執行動作
        next_state_np, reward, done, _ = env.step(action)
        next_state_key = serialize_state(env)
        total_reward += reward

        # 更新 Q-table
        old_value = Q[state_key][action]
        # 選擇 next_state 最大 Q 值（僅考慮合法動作）
        next_valid_mask = env._valid_mask() if not done else np.zeros(env.action_dim, dtype=np.float32)
        next_q = Q[next_state_key]
        next_q_masked = np.copy(next_q)
        next_q_masked[next_valid_mask == 0] = -np.inf
        max_next_q = np.max(next_q_masked) if not done else 0.0

        # Q-Learning 更新公式：Q(s,a) ← Q(s,a) + α [r + γ max Q(s',·) − Q(s,a)]
        Q[state_key][action] = old_value + alpha * (reward + gamma * max_next_q - old_value)

        # 轉換至下一個狀態
        state_key = next_state_key

        if done:
            break
    comb = count_combos(env.board)

    ep_len.append(step)
    episode_com.append(comb)
    episode_rewards.append(total_reward)

    # 每 1000 回合印出一次訓練狀況
    if episode % 1000 == 0:
        #print(env.board)
        avg_reward = np.mean(episode_rewards[-1000:])
        avg_combo = np.mean(episode_com[-1000:])
        mv_avg_r.append(avg_reward)
        mv_avg_c.append(avg_combo)

        print(f"Episode {episode:5d} | Avg Reward (last 1000): {avg_reward:.3f} | Epsilon: {epsilon:.3f}, {avg_combo}, step: {np.mean(ep_len[-1000:])}")
        #print(f"Episode {episode:5d} | Avg Reward (last 1000): {avg_reward:.3f} | Epsilon: {epsilon:.3f}, {np.mean(episode_com

plt.figure()
plt.plot(mv_avg_c, linestyle='-')
plt.xlabel('Episode (×1000)')
plt.ylabel('Combo')
plt.title('Avg Combo (last 1000)')
plt.grid(True)
plt.tight_layout()

# ─────────────────────────────────────────────
# 畫 reward 的走勢圖
plt.figure()
plt.plot(mv_avg_r, linestyle='-')
plt.xlabel('Episode (×1000)')
plt.ylabel('Reward')
plt.title('Avg Reward (last 1000)')
plt.grid(True)
plt.tight_layout()

plt.show()

