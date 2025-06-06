import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, defaultdict
import os
import matplotlib.pyplot as plt
from env import TurnEnv, count_combos

log_dir  = "/content/drive/MyDrive/drl_final/alpha_zero"
os.makedirs(log_dir, exist_ok=True)
log_path = f"{log_dir}/puct_training_log.txt"

# --- 1. Policy-Value Network ---
class PVNet(nn.Module):
    def __init__(self, size, colors, action_dim):
        super().__init__()
        self.size, self.colors = size, colors

        # ----- Conv Backbone -----
        self.conv = nn.Sequential(
            nn.Conv2d(colors, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),    nn.ReLU(),
        )

        # ----- Board feature -----
        self.fc_board = nn.Sequential(
            nn.Linear(128 * size * size, 256), nn.ReLU()
        )

        # ----- Extra info (phase, t, held) -----
        extra_dim = 1 + 1 + size * size
        self.fc_info = nn.Sequential(
            nn.Linear(extra_dim, 128), nn.ReLU()
        )

        # ----- Merge -----
        merged_dim = 256 + 128

        # ===== Policy head =====
        self.pi = nn.Linear(merged_dim, action_dim)

        # ===== Value head =====
        self.v_hidden = nn.Sequential(
            nn.Linear(merged_dim, 128), nn.ReLU()
        )
        self.v_out = nn.Linear(128, 1)

        # ----- (optional) 初始化 -----
        nn.init.constant_(self.pi.weight, 0.0)
        nn.init.constant_(self.pi.bias,   0.0)
        nn.init.constant_(self.v_out.weight, 0.0)
        nn.init.constant_(self.v_out.bias,   0.0)

    def forward(self, board_oh, phase, t, held):
        """
        board_oh: [B, colors, size, size]
        phase   : [B, 1]
        t       : [B, 1]
        held    : [B, size*size]
        """
        # Conv features
        h = self.conv(board_oh)                         # [B,128,H,W]
        h = h.view(h.size(0), -1)                       # [B,128*size*size]
        h = self.fc_board(h)                            # [B,256]

        # Extra info
        info = torch.cat([phase, t, held], dim=1)       # [B,extra_dim]
        h_info = self.fc_info(info)                     # [B,128]

        # Merge
        h_all = torch.cat([h, h_info], dim=1)           # [B,merged_dim]

        # Outputs
        logits = self.pi(h_all)                         # [B,action_dim]

        v_hidden = self.v_hidden(h_all)                 # [B,128]
        value = torch.tanh(self.v_out(v_hidden)).squeeze(-1)  # [B]

        return logits, value

"""
額外資訊如何輸入net"""
def split_state(state_vec, size, colors):
    board_len = colors * size * size
    board = state_vec[:board_len] \
                .reshape(colors, size, size)
    phase = state_vec[board_len]          # 0/1
    t     = state_vec[board_len+1]        # [0,1]
    held  = state_vec[-(size*size):]      # one-hot
    return board, phase, t, held
# --- 2. MCTS Node ---
class PUCTNode:
    def __init__(self, state_vec, parent=None):
        self.state_vec = state_vec
        self.parent = parent
        self.N = defaultdict(int)
        self.W = defaultdict(float)
        self.P = {}
        self.children = {}
        self.expanded = False

    def Q(self, a):
        return 0.0 if self.N[a]==0 else self.W[a]/self.N[a]

    def U(self, a, c):
        total = sum(self.N.values())
        return c * self.P[a] * math.sqrt(total+1e-8)/(1+self.N[a])


# --- 3. PUCT Search with rollout-weighting ---
class PUCTSearch:
    def __init__(
        self, env, net:PVNet,
        iterations=100, c=1.5, gamma=0.99,
        rollout_depth=5, alpha_steps=1000, device="cpu"
    ):
        self.base_env = env
        self.net = net.to(device)
        self.iter = iterations
        self.c = c
        self.gamma = gamma
        self.device = device
        self.size, self.colors = env.size, env.colors
        self.act_dim = env.action_dim
        self.rollout_depth = rollout_depth
        self.alpha_steps = alpha_steps
        self.alpha = 0.0

    def _softmax(self, x):
        x = x - np.max(x)
        exp = np.exp(x)
        return exp/np.sum(exp)

    def rollout(self, env):
        total = 0.0
        γ = self.gamma
        base = env.size * env.size
        for d in range(self.rollout_depth):
            mask = env._valid_mask()
            legal = list(np.where(mask==1)[0])
            if len(legal)==0 or env.done:
                break

            a = int(random.choice(legal))
            _, r, done, _ = env.step(a)
            total += (γ**d)*r
            if done:
                break
        #print("rollout total reward", d, env.steps, total, a)
        return total
    """
    1. 逐步降低random rollout 的比例，改為使用value net (current)
    2. 先用監督式訓練value net，可靠時再替換掉random rollout
    3. always 完整rollout
    4. 先random rollout, 用value net評估剩餘
    """
    def search(self, root_state_vec):
        # update alpha
        # assume external code sets self.alpha appropriately
        root = PUCTNode(root_state_vec)
        # initial expand
        self._expand(root, self.base_env)
        for _ in range(self.iter):
            node, sim_env, path, done = root, self.base_env.clone(), [], False
            total_r = 0
            # Selection
            while node.expanded:
                best_a, best_ucb = None, -float('inf')
                for a in node.P:
                    ucb = node.Q(a) + node.U(a, self.c)
                    if ucb > best_ucb:
                        best_ucb, best_a = ucb, a

                path.append((node, best_a))
                _, rew, done, _ = sim_env.step(best_a)
                total_r += rew
                if best_a not in node.children:
                    nxt = sim_env._get_state()
                    node.children[best_a] = PUCTNode(nxt, parent=node)
                node = node.children[best_a]
                if done:
                    break
            # Expansion + Evaluation

            if not done:
                # network value
                v_net = self._expand(node, sim_env)
                # rollout estimate
                """
                ro = self.rollout(copy.deepcopy(sim_env))
                # combine
                α = self.alpha
                total_r += α * v_net + (1-α) * ro
                """
                total_r += v_net
            # Backprop
            for nd, a in path:
                nd.N[a] += 1
                nd.W[a] += total_r
        # return pi
        pi = np.zeros(self.act_dim)
        tot = sum(root.N.values())
        for a, n in root.N.items():
            pi[a] = n/tot if tot>0 else 0
        return pi

    def _expand(self, node, env):
        if node.expanded:
            # return net value only
            board, ph, tt, held = split_state(node.state_vec,
                                  self.size, self.colors)
            img   = torch.from_numpy(board).unsqueeze(0).float().to(self.device)
            phase = torch.tensor([[ph]], dtype=torch.float32).to(self.device)
            time  = torch.tensor([[tt]], dtype=torch.float32).to(self.device)
            held  = torch.from_numpy(held).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                logits, v = self.net(img, phase, time, held)
            return v.item()
        # first expansion
        board, ph, tt, held = split_state(node.state_vec,
                                  self.size, self.colors)
        img   = torch.from_numpy(board).unsqueeze(0).float().to(self.device)
        phase = torch.tensor([[ph]], dtype=torch.float32).to(self.device)
        time  = torch.tensor([[tt]], dtype=torch.float32).to(self.device)
        held  = torch.from_numpy(held).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits, v = self.net(img, phase, time, held)
        probs = self._softmax(logits.cpu().numpy()[0])
        mask = env._valid_mask()
        for a in np.where(mask==1)[0]:
            node.P[a] = probs[a]
            node.N[a] = 0
            node.W[a] = 0.0
        node.expanded = True
        return v.item()
def compute_grad_vector(loss, net):
    """
    回傳 (梯度向量, 參數 shape 清單, 是否有梯度 mask)
    """
    net.zero_grad()
    loss.backward(retain_graph=True)
    grads = []
    masks = []
    for p in net.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
            masks.append(torch.ones_like(p.grad, dtype=torch.bool).view(-1))
        else:
            grads.append(torch.zeros(p.numel(), device=next(net.parameters()).device))
            masks.append(torch.zeros(p.numel(), dtype=torch.bool, device=next(net.parameters()).device))
    grad_vector = torch.cat(grads)
    mask_vector = torch.cat(masks)
    return grad_vector, mask_vector

# --- 4. Trainer with alpha schedule & MSE logging ---
class AZTrainer:
    def __init__(
        self, env, net:PVNet,
        selfplay_eps=20, iterations=96, c=1.5,
        rollout_depth=5, alpha_steps=1000,
        batch_size=64, lr=1e-3, buffer_size=20000,
        gamma=0.99, value_weight=1.0,
        eval_interval=10, device="cpu"
    ):
        self.env = env
        self.net = net
        self.searcher = PUCTSearch(
            env, net, iterations, c, gamma,
            rollout_depth, alpha_steps, device
        )
        self.buffer = deque(maxlen=buffer_size)
        self.selfplay_eps = selfplay_eps
        self.batch = batch_size
        self.gamma = gamma
        self.value_w = value_weight
        self.opt = torch.optim.Adam(net.parameters(), lr=lr)
        self.device = device
        self.size, self.colors = env.size, env.colors
        self.eval_interval = eval_interval
        self.train_steps = 0
        self.alpha_steps = alpha_steps

    def self_play(self):
        comb_list = []
        rew_list = []
        for i in range(self.selfplay_eps):
            s = self.env.reset()
            traj, rewards, done = [], [], False
            while not done:

                pi = self.searcher.search(s)
                a = np.random.choice(len(pi), p=pi)
                #a = int(np.argmax(pi))
                traj.append((s.copy(), pi.copy()))
                s, r, done, _ = self.env.step(a)
                rewards.append(r)
            z = sum(self.gamma**i * rewards[i] for i in range(len(rewards)))
            comb_list.append(count_combos(self.env.board))
            rew_list.append(sum(rewards))
            """可以更精細的計算reward
            """
            for st, p in traj: self.buffer.append((st, p, z))
        return np.mean(comb_list), np.mean(rew_list)

    def train_step(self):
        """
        1. 隨機抽樣 replay buffer
        2. 前向 → policy、value
        3. loss = policy CE + value_weight × MSE
        4. 反向更新
        5. 動態調 α，並定期列印 value-head MSE
        """
        # ----------- 抽樣 ----------
        if len(self.buffer) < self.batch:
            return 0.0

        batch = random.sample(self.buffer, self.batch)
        boards, phases, times, helds = [], [], [], []
        pis, zs = [], []

        for s_vec, pi, z in batch:
            b, ph, tt, h = split_state(s_vec, self.size, self.colors)
            boards.append(b)           # (colors, size, size)
            phases.append([ph])        # shape (1,)
            times.append([tt])         # shape (1,)
            helds.append(h)            # (size*size,)
            pis.append(pi)             # 行動分布
            zs.append(z)               # 折扣回報

        # ----------- 轉成 numpy.ndarray 再轉 tensor ----------
        # 把 boards: list of (colors, size, size) 轉成一個 array: (B, colors, size, size)
        boards_np = np.array(boards, dtype=np.float32)
        boards_t  = torch.from_numpy(boards_np).to(self.device)

        phase_np = np.array(phases, dtype=np.float32)   # shape: (B, 1)
        phase_t  = torch.from_numpy(phase_np).to(self.device)

        time_np = np.array(times, dtype=np.float32)     # shape: (B, 1)
        time_t  = torch.from_numpy(time_np).to(self.device)

        held_np = np.array(helds, dtype=np.float32)     # shape: (B, size*size)
        held_t  = torch.from_numpy(held_np).to(self.device)

        pis_np = np.array(pis, dtype=np.float32)        # shape: (B, action_dim)
        pis_t  = torch.from_numpy(pis_np).to(self.device)

        zs_np = np.array(zs, dtype=np.float32)          # shape: (B,)
        zs_t  = torch.from_numpy(zs_np).to(self.device)

        # ----------- 前向 ----------
        logits, values = self.net(boards_t, phase_t, time_t, held_t)   # logits:[B,A]  values:[B]

        # ----------- 損失 ----------
        logp         = F.log_softmax(logits, dim=1)
        policy_loss  = -torch.mean(torch.sum(pis_t * logp, dim=1))
        value_loss   = F.mse_loss(values, zs_t)
        loss         = policy_loss + self.value_w * value_loss

        # 1. policy_loss 梯度
        policy_grad, policy_mask = compute_grad_vector(policy_loss, self.net)

        # 2. value_loss 梯度
        value_grad, value_mask  = compute_grad_vector(value_loss, self.net)

        # 3. cosine similarity
        """
        common_mask = policy_mask & value_mask
        if common_mask.sum() == 0:
            cos_sim = float('nan')
        else:
            pg = policy_grad[common_mask]
            vg = value_grad[common_mask]
            cos_sim = F.cosine_similarity(pg.unsqueeze(0), vg.unsqueeze(0)).item()
        print(f"Policy-Value Gradient Cosine Similarity: {cos_sim:.4f}")
        """
        # ----------- 反向 ----------
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # ----------- alpha schedule ----------
        self.train_steps += 1
        #self.searcher.alpha = min(1.0, self.train_steps / self.alpha_steps)

        # ----------- 週期性 MSE 監控 ----------
        if self.train_steps % self.eval_interval == 0:
            eval_batch = random.sample(self.buffer, min(len(self.buffer), self.batch))
            eb_b, eb_p, eb_t, eb_h, eb_z = [], [], [], [], []
            for s_vec, _, z in eval_batch:
                b, ph, tt, h = split_state(s_vec, self.size, self.colors)
                eb_b.append(b);  eb_p.append([ph]);  eb_t.append([tt]);  eb_h.append(h);  eb_z.append(z)
            eb_b  = torch.tensor(eb_b, dtype=torch.float32, device=self.device)
            eb_p  = torch.tensor(eb_p, dtype=torch.float32, device=self.device)
            eb_t  = torch.tensor(eb_t, dtype=torch.float32, device=self.device)
            eb_h  = torch.tensor(eb_h, dtype=torch.float32, device=self.device)
            eb_z  = torch.tensor(eb_z, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                _, eb_val = self.net(eb_b, eb_p, eb_t, eb_h)
            mse = F.mse_loss(eb_val, eb_z)
            print(f"[Eval] step {self.train_steps:4d} | α={self.searcher.alpha:.3f} | Value-MSE={mse.item():.4f}")

        return loss.item()

# --- 5. Main Function ---
if __name__ == "__main__":
    # Hyperparameters
    env = TurnEnv(size=4, colors=4, max_steps=100, min_combos=2)
    net = PVNet(env.size, env.colors, env.action_dim)
    trainer = AZTrainer(
        env, net,
        selfplay_eps=25,
        iterations=25,
        c=2.3,
        rollout_depth=5,
        alpha_steps=300,   # 漸增到 alpha=1 所需 train steps
        batch_size=256,
        lr=2e-4,
        buffer_size=1000000,
        gamma=0.995,
        value_weight=1,
        eval_interval=50,
        device="cuda"
    )

    # 完整訓練循環
    total_epochs = 3000
    avg_comb_list = []
    avg_rew_list = []
    loss_list = []
    for ep in range(1, total_epochs+1):
        avg_comb, avg_rew = trainer.self_play()
        avg_comb_list.append(avg_comb)
        avg_rew_list.append(avg_rew)
        loss = trainer.train_step()
        loss_list.append(loss)
        print("ep:", ep)
        if ep % 50 == 0:
            print(f"Epoch {ep}/{total_epochs}, loss={loss:.4f}")
            plt.figure()
            plt.plot(avg_rew_list, linestyle='-')
            plt.xlabel('Episode')
            plt.ylabel('reward')
            plt.title('avg reward')
            plt.grid(True)
            plt.tight_layout()
            save_path = log_dir + '/avg_reward.png'
            #plt.savefig(save_path)
            plt.show()

            plt.figure()
            plt.plot(avg_comb_list, linestyle='-')
            plt.xlabel('Episode')
            plt.ylabel('Combo')
            plt.title('avg Combo')
            plt.grid(True)
            plt.tight_layout()
            save_path = log_dir + '/avg_combo.png'
            #plt.savefig(save_path)
            plt.show()

            plt.figure()
            plt.plot(loss_list, linestyle='-')
            plt.xlabel('Episode')
            plt.ylabel('loss')
            plt.title('Training loss')
            plt.grid(True)
            plt.tight_layout()
            save_path = log_dir + '/t_loss.png'
            #plt.savefig(save_path)
            plt.show()

            checkpoint = {
                'epoch': ep,
                'model_state_dict': trainer.net.state_dict(),
                'optimizer_state_dict': trainer.opt.state_dict(),
                'train_steps': trainer.train_steps,
                'alpha': trainer.searcher.alpha,
                # 如果想要保存 Replay Buffer，可將 deque 轉為 list
                # 'buffer': list(trainer.buffer),
                # 同時也可保存目前的統計資料
                'avg_comb_list': avg_comb_list,
                'avg_rew_list': avg_rew_list,
                'loss_list': loss_list
            }
            ckpt_path = os.path.join(log_dir, f'checkpoint_ep.pth')
            torch.save(checkpoint, ckpt_path)
            print(f"Checkpoint 已儲存到 {ckpt_path}")

    @torch.no_grad()
    def validate_policy_value(env, net, searcher_cls, episodes=100, device="cpu"):
        """
        使用訓練好的 net，跑 `episodes` 回合自我對弈。
        回傳平均 reward 與平均步數，可依需求擴充。
        """
        net.eval()
        total_reward, total_steps = 0.0, 0
        # 以 evaluation 用的 searcher，alpha 直接設為 1（僅用 value，不再 random rollout）
        searcher = searcher_cls(
            env, net,
            iterations=50,        # 可自行調整
            c=2.3,
            gamma=0.995,
            rollout_depth=0,      # 不再 random rollout
            alpha_steps=1,        # 讓 alpha 一開始就是 1
            device=device,
        )
        searcher.alpha = 1.0

        max_combo = []
        agent_combo = []
        for _ in range(episodes):
            s = env.reset()
            done, ep_reward, steps = False, 0.0, 0
            while not done:
                pi = searcher.search(s)
                a = int(np.argmax(pi))  # 評估用：貪婪選擇
                s, r, done, _ = env.step(a)
                ep_reward += r
                steps += 1
            total_reward += ep_reward
            total_steps  += steps

            max_combo.append(env.max_combo)
            agent_combo.append(count_combos(env.board))

        avg_reward = total_reward / episodes
        avg_steps  = total_steps  / episodes

        plt.plot(max_combo, label='Max combo', linestyle='-', linewidth=2)
        plt.plot(agent_combo, label='Agent combo', linestyle='--', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Combo')
        plt.title('Agent v.s. max combo ')
        #plt.xticks(episodes)  # 顯示所有 Episode 序號為 x 軸刻度
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return avg_reward, avg_steps

    print("Training completed, running 100-episode validation …")
    avg_r, avg_s = validate_policy_value(
        env, net, PUCTSearch,
        episodes=100,
        device="cuda"
    )
    print(f"[Validation] Avg Reward = {avg_r:.4f} | Avg Steps = {avg_s:.1f}")



