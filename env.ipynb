# 1. 定義環境與消除邏輯
import random, numpy as np, torch, torch.nn.functional as F
from collections import deque
from gym import Env
import numpy as np
from collections import deque

def get_components(board):
    """
    使用 BFS，回傳一個 list，其中每個元素都是一個 component，
    component 本身是一個 list，裡面放該 component 中所有 (r, c) 座標。
    """
    rows, cols = board.shape
    visited = np.zeros((rows, cols), dtype=bool)
    components = []

    for r in range(rows):
        for c in range(cols):
            if board[r, c] < 0 or visited[r, c]:
                continue

            color = board[r, c]
            queue = deque([(r, c)])
            visited[r, c] = True
            comp = [(r, c)]

            while queue:
                cr, cc = queue.popleft()
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if not visited[nr, nc] and board[nr, nc] == color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                            comp.append((nr, nc))

            components.append(comp)

    return components

def find_matches_per_component(component):
    """
    給定單一 component (list of (r,c))，檢查其中是否有「水平或垂直 ≥3 連線」：
      - 水平：同一 row 上，col 索引連續 ≥3。
      - 垂直：同一 col 上，row 索引連續 ≥3。
    如果有，就將該 component 裡，屬於那條連線的所有座標回傳 (set of (r,c))；
    如果同一 component 內有兩條或更多不相交的直線，也都一併加到同一個 set 裡。
    若該 component 根本沒有任何可消直線，就回傳空 set。
    """
    # 先把 component 裡頭的座標按 row / col 分組
    rows_dict = {}  # key = row 編號，value = list of col
    cols_dict = {}  # key = col 編號，value = list of row

    for (r, c) in component:
        rows_dict.setdefault(r, []).append(c)
        cols_dict.setdefault(c, []).append(r)

    to_clear = set()

    # 檢查「水平連線」：對每個 row 看它的 col_list 裡是否有連續 >= 3
    for r, col_list in rows_dict.items():
        if len(col_list) < 3:
            continue
        col_list.sort()
        count = 1
        for i in range(1, len(col_list)):
            if col_list[i] == col_list[i-1] + 1:
                count += 1
            else:
                count = 1
            if count >= 3:
                # 把連續段 col_list[i-count+1 ... i] 的所有 (r, c) 加到 to_clear
                for j in range(i-count+1, i+1):
                    to_clear.add((r, col_list[j]))
                # 不 break，因為可能有更長連續段，或一次 component 有多條橫線。

    # 檢查「垂直連線」：對每個 col 看它的 row_list 裡是否有連續 >= 3
    for c, row_list in cols_dict.items():
        if len(row_list) < 3:
            continue
        row_list.sort()
        count = 1
        for i in range(1, len(row_list)):
            if row_list[i] == row_list[i-1] + 1:
                count += 1
            else:
                count = 1
            if count >= 3:
                # 把連續段 row_list[i-count+1 ... i] 的所有 (r, c) 加到 to_clear
                for j in range(i-count+1, i+1):
                    to_clear.add((row_list[j], c))
                # 不 break，同一 component 可能有多條直線。

    return to_clear

def clear_and_drop(board, matches):
    rows, cols = board.shape
    # 清除：標為 -1
    for r, c in matches:
        board[r, c] = -1
    # 下落：把非 -1 的收集起來
    for c in range(cols):
        stack = [board[r, c] for r in range(rows) if board[r, c] >= 0]
        for r in range(rows-1, -1, -1):
            board[r, c] = stack.pop() if stack else -1
    return board

def count_combos(initial_board):
    """
    改良版 count_combos：每次迴圈遍歷所有 component，
    如果 component 內有可消直線，就算一個 combo，並把那條直線座標加入待消 set；
    直到再也沒有任何 component 可消為止。
    """
    board = initial_board.copy()
    total_combos = 0

    while True:
        # 1. 先把所有 component 拆好
        components = get_components(board)

        # 2. 針對每個 component 檢查「≥3 直線」
        to_clear_all = set()
        combos_this_round = 0

        for comp in components:
            # comp 是一個 list of (r,c)，缺少任何可消直線就得回空 set
            this_clear = find_matches_per_component(comp)
            if this_clear:
                # 只要 comp 裡有任一條直線，就算 comp 貢獻 1 combo
                combos_this_round += 1
                # 把這條直線上所有座標加到整輪要清的集合裡
                to_clear_all |= this_clear

        # 如果這一輪任何 component 都沒有可消直線，就跳出
        if combos_this_round == 0:
            break

        # 3. 累加 combo（同一輪按 component 數計）
        total_combos += combos_this_round

        # 4. 清除所有 to_clear_all 座標，並下落
        clear_and_drop(board, to_clear_all)

    return total_combos

#─── 環境定義 ──────────────────────────────────────────
class TurnEnv:
    def __init__(self, size=4, colors=6, max_steps=100, min_combos=1):
        self.size        = size
        self.colors      = colors
        self.max_steps   = max_steps
        self.min_combos  = min_combos
        self.action_dim  = size*size + 5
        self.max_combo   = 0
        self.prev_combo  = 0    # for reward shaping
        self.reset()
    def reset(self):
        # 產生符合條件的盤面
        while True:
            board = np.random.randint(self.colors, size=(self.size, self.size))
            if count_combos(board) != 0:
                continue
            counts    = np.bincount(board.flatten(), minlength=self.colors)
            potential = sum(min(cnt//3, 2) for cnt in counts)
            if potential >= self.min_combos:
                break

        self.board       = board
        self.max_combo   = potential
        self.rs_list = [3]*potential*2
        self.prev_combo  = 0    # reset shaping baseline
        self.phase       = 1
        self.steps       = 0
        self.held        = None
        self.done        = False
        return self._get_state()

    def _get_state(self):
        b     = np.eye(self.colors)[self.board].flatten()
        phase = np.array([self.phase], dtype=np.float32)
        t     = np.array([self.steps/self.max_steps], dtype=np.float32)
        held  = np.zeros(self.size*self.size, dtype=np.float32)
        if self.held is not None:
            idx      = self.held[0]*self.size + self.held[1]
            held[idx] = 1.0
        return np.concatenate([b, phase, t, held]).astype(np.float32)

    def _valid_mask(self):
        mask = np.zeros(self.action_dim, dtype=np.float32)
        if self.phase == 1:
            mask[:self.size*self.size] = 1
        else:
            r, c = self.held
            dirs = [(-1,0),(1,0),(0,-1),(0,1)]
            for i,(dr,dc) in enumerate(dirs):
                nr, nc = r+dr, c+dc
                if 0<=nr<self.size and 0<=nc<self.size:
                    mask[self.size*self.size + i] = 1
            mask[-1] = 1
        return mask

    def step(self, action):
        mask = self._valid_mask()
        # 無效動作
        if mask[action] == 0:
            self.done = True
            return self._get_state(), -1.0, True, {}

        reward = 0.0

        # 起手階段
        if self.phase == 1:
            r, c      = divmod(action, self.size)
            self.held = (r, c)
            self.phase = 2

        else:
            base = self.size*self.size

            # 最後一步：放下並結算，不加 middle shaping
            if action == base + 4:
                combos   = count_combos(self.board)
                baseline = 0#self.max_combo // 2
                reward   = (combos - baseline) * 0.2

                if combos > 0:
                    reward += (self.max_steps - self.steps) / 2500.0

                self.done = True
                return self._get_state(), reward, True, {}

            # 中途交換：先 swap
            r, c    = self.held
            dirs    = [(-1,0),(1,0),(0,-1),(0,1)]
            dr, dc  = dirs[action - base]
            nr, nc  = r+dr, c+dc
            self.board[r,c], self.board[nr,nc] = self.board[nr,nc], self.board[r,c]
            self.held = (nr, nc)

            # Reward shaping：計算當前潛在 combo 並與前一狀態差值
            """
            curr_combo = count_combos(self.board)

            if curr_combo > self.prev_combo:
                reward    += (curr_combo - self.prev_combo) * 0.2
                if self.rs_list[curr_combo] > 0:
                    reward += (curr_combo - self.prev_combo) * 0.2
                    self.rs_list[curr_combo] -= 1

            else:
               reward    -= (self.prev_combo - curr_combo) * 0.1

            self.prev_combo = curr_combo
            """
        # 計步與超時自動放下
        self.steps += 1
        if self.steps >= self.max_steps:
            combos   = count_combos(self.board)
            baseline = 0#self.max_combo // 2
            reward   = (combos - baseline) * 0.2

            if combos > 0:
                reward += (self.max_steps - self.steps) / 2500.0

            self.done = True
            return self._get_state(), reward, True, {}

        return self._get_state(), reward, self.done, {}
    def clone(self):
        """
        回傳一個「與當前 env 狀態完全一致」的獨立副本，
        只複製必要屬性與 np.ndarray，避免 __init__() 的額外開銷。
        """
        # 1. 建立未初始化的空物件，跳過 __init__()
        new_env = TurnEnv.__new__(TurnEnv)

        # 2. ----------- 基本（不可變）屬性 -----------
        new_env.size       = self.size
        new_env.colors     = self.colors
        new_env.max_steps  = self.max_steps
        new_env.min_combos = self.min_combos
        new_env.action_dim = self.action_dim

        # 3. ----------- 動態（可變）狀態 -----------
        new_env.board      = self.board.copy()             # 深複製棋盤
        new_env.max_combo  = self.max_combo
        new_env.prev_combo = self.prev_combo
        new_env.phase      = self.phase
        new_env.steps      = self.steps
        new_env.done       = self.done
        new_env.held       = None if self.held is None else tuple(self.held)

        # 4. 若日後需要複製 RNG，可視情況加入：
        # new_env._np_random_state = np.random.get_state()

        return new_env
def evaluate(policy_net, env, num_episodes=10, device="cpu"):
    policy_net.eval()
    initial_boards = []
    combos = []

    for ep in range(num_episodes):
        # 1. 重置環境，並記錄初始盤面
        state_np = env.reset()
        initial_board = env.board.copy()   # 記下 numpy 盤面
        initial_boards.append(initial_board)

        state = torch.from_numpy(state_np).float()  # flat state tensor
        done = False

        # 2. 用 greedy policy 完整跑完一集
        while not done:
            mask = env._valid_mask().astype(bool)
            # 把 flat 前一段轉 image
            img = state[:colors*grid_size*grid_size] \
                      .view(1, colors, grid_size, grid_size) \
                      .to(device)
            with torch.no_grad():
                q = policy_net(img)[0].cpu().numpy()
            q[~mask] = -1e9
            action = int(np.argmax(q))

            next_np, _, done, _ = env.step(action)
            state = torch.from_numpy(next_np).float()

        # 3. 結算 combo
        c = count_combos(env.board)
        combos.append(c)

    # 4. 印出結果
    print(f"Evaluation over {num_episodes} episodes:")
    for i, (board, c) in enumerate(zip(initial_boards, combos), 1):
        print(f"\nEpisode {i}:")
        print(board)
        print(f"→ combo = {c}")

    policy_net.train()
    return combos, initial_boards
