# ai/views.py
import os, json, traceback, random, logging
import numpy as np
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

logger = logging.getLogger(__name__)

# ===== 설정 =====
BOARD_SIZE = 9
MODEL_FILENAME = 'alphazero_gomoku9.pth'  # 필요시 교체
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ai', MODEL_FILENAME)

# ===== Board 인터페이스 =====
try:
    from .board import Board  # 프로젝트에 이미 있다면 사용
except Exception:
    class Board:
        def __init__(self, width=9, height=9):
            self.w, self.h = width, height
            self.board = np.zeros((self.h, self.w), dtype=int)  # 0,1,-1
            self.current_player = 1
            self.moves_played = 0

        def set_state(self, board_list, current_player=1):
            arr = np.array(board_list, dtype=int)
            assert arr.shape == (self.h, self.w)
            self.board[:] = arr
            self.current_player = int(current_player)
            self.moves_played = int((arr != 0).sum())

        @property
        def availables(self):
            xs, ys = np.where(self.board == 0)
            return [int(x)*self.w + int(y) for x, y in zip(xs, ys)]

        def legal_mask(self):
            mask = np.zeros(self.w*self.h, dtype=bool)
            for m in self.availables:
                mask[m] = True
            return mask

        def move_to_location(self, move):
            return (move // self.w, move % self.w)

        def location_to_move(self, loc):
            x, y = loc
            return x*self.w + y

        def do_move(self, move):
            if isinstance(move, tuple):
                x, y = move
            else:
                x, y = self.move_to_location(int(move))
            if self.board[x, y] != 0:
                raise ValueError("illegal move")
            self.board[x, y] = self.current_player
            self.current_player *= -1
            self.moves_played += 1

        def _check_winner(self):
            s = self.w
            b = self.board
            dirs = [(1,0),(0,1),(1,1),(1,-1)]
            for x in range(s):
                for y in range(s):
                    p = b[x,y]
                    if p == 0: continue
                    for dx, dy in dirs:
                        cnt = 1
                        for k in range(1,5):
                            nx, ny = x+dx*k, y+dy*k
                            if 0 <= nx < s and 0 <= ny < s and b[nx,ny] == p: cnt += 1
                            else: break
                        if cnt >= 5:
                            return True, int(p)
            return False, 0

        def game_end(self):
            done, winner = self._check_winner()
            if done:
                return True, winner
            if self.moves_played == self.w*self.h or not self.availables:
                return True, 0
            return False, 0

        def encode_two_channels(self):
            cur = (self.board == self.current_player).astype(np.float32)
            opp = (self.board == -self.current_player).astype(np.float32)
            return np.stack([cur, opp], axis=0)

# ===== MCTS / 모델 로딩 =====
device = 'cpu'
mcts_player = None
torch_ok = False
mcts_ok = False

try:
    import torch
    from .torch_policy_value_net import AlphaZeroNet
    torch_ok = True

    net = AlphaZeroNet(board_size=BOARD_SIZE).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    def policy_value_fn(b: Board):
        obs = b.encode_two_channels()          # (2, S, S)
        legal = b.legal_mask()                 # (S*S,)
        with torch.no_grad():
            x = torch.tensor(obs).unsqueeze(0).to(device)
            mask = torch.tensor(legal).unsqueeze(0).to(device)
            logp, v = net(x, legal_mask=mask)
            p = torch.exp(logp).squeeze(0).cpu().numpy()  # (S*S,)
        action_probs = [(i, float(p[i])) for i in np.where(legal)[0]]
        return action_probs, float(v.item())

    from .mcts_alphaZero import MCTSPlayer
    mcts_player = MCTSPlayer(policy_value_function=policy_value_fn, c_puct=5, n_playout=400)
    mcts_ok = True

except Exception as e:
    logger.warning("AI init fallback: %s", e, exc_info=True)
    # mcts_player = None -> 폴백(랜덤) 사용

def _safe_get_action(board: Board):
    """MCTS+Net이 가능하면 사용, 아니면 랜덤 합법수."""
    end, _ = board.game_end()
    if end:
        return None
    if mcts_player is not None:
        try:
            mv = mcts_player.get_action(board)
            if isinstance(mv, int):
                return mv
            if isinstance(mv, (tuple, list)) and len(mv) == 2:
                return board.location_to_move((int(mv[0]), int(mv[1])))
        except Exception as e:
            logger.error("MCTS get_action failed: %s", e, exc_info=True)
    # fallback
    avail = board.availables
    return random.choice(avail) if avail else None

# ===== Views =====
def index(request):
    # 템플릿이 있으면 render 사용: 
    return render(request, 'index.html')
    # return JsonResponse({"message": "Welcome to the AI service index.", "path": request.path})

def api_health(request):
    return JsonResponse({
        "status": "AI service is running.",
        "mcts": mcts_ok and (mcts_player is not None),
        "torch": torch_ok,
        "path": request.path
    })

@csrf_exempt
def api_move(request):
    if request.method != "POST":
        return JsonResponse({'error': 'POST only'}, status=405)
    try:
        # JSON 파싱
        if request.headers.get("Content-Type", "").startswith("application/json"):
            data = json.loads(request.body or b"{}")
        else:
            return JsonResponse({'error': 'Content-Type must be application/json'}, status=400)

        board_data = data.get("board")
        current = data.get("current_player", 1)
        if board_data is None:
            return JsonResponse({'error': 'Missing "board"'}, status=400)

        board = Board(width=BOARD_SIZE, height=BOARD_SIZE)
        board.set_state(board_data, current_player=current)

        move = _safe_get_action(board)
        if move is None:
            return JsonResponse({'move': None, 'note': 'game ended or no legal moves'})

        x, y = board.move_to_location(move)
        return JsonResponse({'move': [x, y]})

    except Exception as e:
        logger.error("api_move failed: %s", e, exc_info=True)
        return JsonResponse({'error': str(e)}, status=500)
