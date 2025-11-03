# app.py
# -*- coding: utf-8 -*-
import os, pickle, numpy as np
from flask import Flask, request, jsonify, send_from_directory
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy

app = Flask(__name__, static_url_path='', static_folder='static')

# --- 전역 게임 상태 (단일 게임) ---
board = None
game = None
mcts_player = None
order = 0  # 0: 사람 흑, 1: AI 흑
width, height, n = 9, 9, 5

def load_model(hard=10000):
    model_file = f'./model/policy_9_{hard}.model'
    if not os.path.exists(model_file):
        alt = f'./omok-ai-master/model/policy_9_{hard}.model'
        if os.path.exists(alt): model_file = alt
        else: raise FileNotFoundError(model_file)
    params = pickle.load(open(model_file, 'rb'), encoding='bytes')
    return params

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.post('/api/new')
def new_game():
    global board, game, mcts_player, order
    data = request.get_json(force=True) or {}
    hard = int(data.get('hard', 10000))
    order = int(data.get('order', 0))
    c_puct = float(data.get('c_puct', 3.0))
    n_playout = int(data.get('n_playout', 1600))

    # 보드 초기화
    board = Board(width=width, height=height, n_in_row=n)
    game = Game(board)

    # 정책망 & MCTS
    params = load_model(hard)
    best_policy = PolicyValueNetNumpy(width, height, params)
    mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=c_puct, n_playout=n_playout)

    # 시작
    board.init_board(order)
    if board.is_you_black():  # 흑 차례면 금수 갱신
        board.set_forbidden()

    return jsonify(state=_pack_state())

@app.post('/api/human')
def human_move():
    """ {h, w} 입력 → 유효시 수 반영 """
    global board, game
    data = request.get_json(force=True) or {}
    h = int(data['h']); w = int(data['w'])
    if board is None: return jsonify(error="no game"), 400

    move = board.location_to_move([h, w])
    if move == -1: return jsonify(error="out"), 400
    if move in board.states.keys(): return jsonify(error="occupied"), 400
    if board.is_you_black() and (h, w) in getattr(board, "forbidden_locations", []):
        return jsonify(error="forbidden"), 400

    board.do_move(move)
    end, winner = board.game_end()
    if end: return jsonify(state=_pack_state(), end=True, winner=_winner_name(winner))

    if board.is_you_black(): board.set_forbidden()
    return jsonify(state=_pack_state())

@app.post('/api/ai')
def ai_move():
    """ AI가 한 수 둠 """
    global board, mcts_player
    if board is None: return jsonify(error="no game"), 400

    # 강수 결정 (결정적으로)
    acts, probs = mcts_player.mcts.get_move_probs(board, temp=1e-3)
    move = int(acts[int(np.argmax(probs))])
    mcts_player.mcts.update_with_move(-1)

    board.do_move(move)
    end, winner = board.game_end()
    if end: return jsonify(state=_pack_state(), end=True, winner=_winner_name(winner))

    if board.is_you_black(): board.set_forbidden()
    return jsonify(state=_pack_state())

def _winner_name(winner):
    if winner == -1: return "Tie"
    # winner == 1(사람) or 2(AI)
    return "Player" if winner == 1 else "AI"

def _pack_state():
    # 판(행렬), 금수, 마지막 수, 차례, 선후공 등 전달
    return {
        "width": board.width,
        "height": board.height,
        "order": order,
        "current_player": board.current_player,  # 1=사람, 2=AI
        "stones": board.states,                  # {move_idx: player_id}
        "last_loc": board.last_loc if board.last_loc != -1 else None,
        "forbidden": getattr(board, "forbidden_locations", []),
        "you_black": board.is_you_black()
    }

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
