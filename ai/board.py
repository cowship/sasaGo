# board.py (필수 메서드 보강)

import numpy as np

class Board:
    def __init__(self, width=9, height=9):
        self.w, self.h = width, height
        self.board = np.zeros((self.h, self.w), dtype=int)  # 0 empty, 1 black, -1 white
        self.current_player = 1
        self.moves_played = 0

    def set_state(self, board_list, current_player=1):
        arr = np.array(board_list, dtype=int)
        assert arr.shape == (self.h, self.w)
        self.board[:] = arr
        self.current_player = int(current_player)
        self.moves_played = int((arr != 0).sum())

    # ---- 좌표/인덱스 변환 ----
    def location_to_move(self, loc):
        x, y = loc
        return x * self.w + y

    def move_to_location(self, move):
        return (move // self.w, move % self.w)

    # ---- 합법수 ----
    @property
    def availables(self):
        xs, ys = np.where(self.board == 0)
        return [int(x)*self.w + int(y) for x, y in zip(xs, ys)]

    def legal_mask(self):
        mask = np.zeros(self.w * self.h, dtype=bool)
        for m in self.availables:
            mask[m] = True
        return mask

    # ---- 착수 ----
    def do_move(self, move):
        """move는 인덱스(int) 또는 (x,y) tuple 둘 다 허용"""
        if isinstance(move, tuple):
            x, y = move
        else:
            x, y = self.move_to_location(int(move))
        if self.board[x, y] != 0:
            raise ValueError("잘못된 수입니다.")
        self.board[x, y] = self.current_player
        self.moves_played += 1
        self.current_player *= -1

    # ---- 상태 인코딩(정책/가치망용) ----
    def encode_two_channels(self):
        cur = (self.board == self.current_player).astype(np.float32)
        opp = (self.board == -self.current_player).astype(np.float32)
        return np.stack([cur, opp], axis=0)

    # ---- 승패 판정 ----
    def _check_winner(self):
        s = self.w
        b = self.board
        dirs = [(1,0),(0,1),(1,1),(1,-1)]
        for x in range(s):
            for y in range(s):
                p = b[x, y]
                if p == 0: 
                    continue
                for dx, dy in dirs:
                    cnt = 1
                    for k in range(1, 5):
                        nx, ny = x + dx*k, y + dy*k
                        if 0 <= nx < s and 0 <= ny < s and b[nx, ny] == p:
                            cnt += 1
                        else:
                            break
                    if cnt >= 5:
                        return True, int(p)
        return False, 0

    def game_end(self):
        """MCTS가 기대하는 인터페이스: (end: bool, winner: int)"""
        done, winner = self._check_winner()
        if done:
            return True, winner
        # 꽉 차면 무승부
        if self.moves_played == self.w * self.h or not self.availables:
            return True, 0
        return False, 0

    # (라이브러리에 따라 필요한 경우)
    def get_current_player(self):
        return self.current_player
