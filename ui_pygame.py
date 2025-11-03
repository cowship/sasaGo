# -*- coding: utf-8 -*-
import os
import sys
import pickle
import pygame
import numpy as np

# === 프로젝트 내부 모듈 임포트 (경로 맞춰주세요) ===
from game import Board, Game  # 당신 zip의 game.py
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy

# -----------------------------
# 설정값 (원하면 여기만 바꿔도 됩니다)
# -----------------------------
BOARD_SIZE_PX = 720         # 보드 캔버스 한 변 픽셀
MARGIN_PX     = 40          # 여백
STONE_RADIUS  = 14          # 돌 반지름
LINE_COLOR    = (50, 50, 50)
BG_COLOR      = (240, 220, 160)  # 바둑판 느낌
BLACK_COLOR   = (25, 25, 25)
WHITE_COLOR   = (240, 240, 240)
LAST_MOVE_CLR = (255, 120, 0)    # 마지막 수 테두리
FORBID_CLR    = (180, 0, 0)      # 금수 'X'
TEXT_CLR      = (15, 15, 15)
FPS           = 60

# 글꼴 이름(없으면 pygame 기본 폰트)
FONT_NAME     = None

# -----------------------------
# Pygame 기반 Human 입력 클래스
# -----------------------------
class HumanPygame(object):
    """
    기존 Human.get_action(board)가 input()으로 좌표 받던 것을
    pygame 클릭으로 받아주는 버전.
    """
    def __init__(self, ui):
        self.player = None
        self.ui = ui  # UIBoardRenderer 인스턴스

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        """
        사람 차례에서 마우스 클릭을 기다려 유효한 수를 반환.
        금수/중복/범위 밖이면 클릭 무시하고 계속 대기.
        ESC / 창 닫기시 안전 종료.
        """
        clock = pygame.time.Clock()

        # 금수 좌표 최신화(흑 차례라면 상위 루프에서 이미 호출됨)
        # 여기서는 반복 렌더링만 수행
        while True:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # 격자 교차점으로 좌표 스냅
                    mx, my = event.pos
                    grid = self.ui.pixel_to_grid(mx, my)
                    if grid is None:
                        # 보드 밖 클릭: 무시
                        continue
                    h, w = grid  # (행, 열)
                    move = board.location_to_move([h, w])
                    # 유효성 검사 (이미 둔 곳/OUT/금수)
                    if move == -1:  # out
                        continue
                    if move in board.states.keys():  # 중복 클릭
                        continue
                    # 금수: 흑 차례일 때만 금수 목록이 의미 있음
                    if board.is_you_black() and tuple([h, w]) in getattr(board, "forbidden_locations", []):
                        # 살짝 깜박이는 이펙트
                        self.ui.flash_forbidden(h, w)
                        continue
                    return move

            # 보드 렌더(사람 차례 메시지 포함)
            self.ui.draw(board, status_text="당신의 차례입니다 (마우스를 클릭하세요)")

    def __str__(self):
        return f"Human {self.player}"


# -----------------------------
# 그리기/좌표계 담당
# -----------------------------
class UIBoardRenderer:
    def __init__(self, width, height):
        pygame.init()
        # 전체 창: 좌측에 보드, 우측/상단에 상태 텍스트 영역 조금 확보
        ui_w = BOARD_SIZE_PX + MARGIN_PX*2
        ui_h = BOARD_SIZE_PX + MARGIN_PX*2 + 60
        self.surface = pygame.display.set_mode((ui_w, ui_h))
        pygame.display.set_caption("Gomoku (Renju) - Pygame UI")
        self.font = pygame.font.SysFont(FONT_NAME, 18)
        self.font_big = pygame.font.SysFont(FONT_NAME, 22, bold=True)

        self.board_w = width
        self.board_h = height
        self.grid_w  = self.board_w - 1
        self.grid_h  = self.board_h - 1

        # 교차점 간 간격
        self.step_x = (BOARD_SIZE_PX - 2*MARGIN_PX) / self.grid_w if self.grid_w > 0 else 0
        self.step_y = (BOARD_SIZE_PX - 2*MARGIN_PX) / self.grid_h if self.grid_h > 0 else 0

        # 보드 영역 좌상단 기준점
        self.board_x0 = MARGIN_PX
        self.board_y0 = MARGIN_PX + 40  # 위쪽에 텍스트 한 줄 공간

        self.clock = pygame.time.Clock()

    # ===== 좌표 변환 =====
    def grid_to_pixel(self, h, w):
        """
        (행h, 열w) -> 픽셀 좌표(중심)
        """
        x = self.board_x0 + w * self.step_x
        y = self.board_y0 + h * self.step_y
        return int(round(x)), int(round(y))

    def pixel_to_grid(self, mx, my):
        # 보드 영역 경계(조금 여유를 둠)
        x_min = self.board_x0 - self.step_x*0.6
        y_min = self.board_y0 - self.step_y*0.6
        x_max = self.board_x0 + self.step_x*(self.grid_w + 0.6)
        y_max = self.board_y0 + self.step_y*(self.grid_h + 0.6)
        if not (x_min <= mx <= x_max and y_min <= my <= y_max):
            return None

        # 가장 가까운 격자 인덱스
        w = int(round((mx - self.board_x0)/self.step_x))
        h = int(round((my - self.board_y0)/self.step_y))
        if not (0 <= h < self.board_h and 0 <= w < self.board_w):
            return None

        # 히트박스(허용 반경): 교차점 중심에서 반경 tol 이내면 클릭 인정
        cx, cy = self.grid_to_pixel(h, w)
        dx, dy = mx - cx, my - cy
        # 교차점당 간격이 클수록 허용 반경도 커지도록
        tol = max(self.step_x, self.step_y) * 0.45   # 0.4~0.5 사이가 체감 좋아요
        if (dx*dx + dy*dy) ** 2 <= (tol*tol) ** 2:
            return (h, w)
        # 허용 반경 밖이면 무시
        return None


    # ===== 그리기 =====
    def draw_board(self):
        # 배경
        self.surface.fill(BG_COLOR)
        # 바둑판 그리드
        for i in range(self.board_h):
            x1, y = self.grid_to_pixel(i, 0)
            x2, _ = self.grid_to_pixel(i, self.board_w-1)
            pygame.draw.line(self.surface, LINE_COLOR, (x1, y), (x2, y), 2)

        for j in range(self.board_w):
            x, y1 = self.grid_to_pixel(0, j)
            _, y2 = self.grid_to_pixel(self.board_h-1, j)
            pygame.draw.line(self.surface, LINE_COLOR, (x, y1), (x, y2), 2)

        # 별점(핵심 교차점) – 15x15/13x13일 때만 몇 개 찍어준다
        if self.board_w == self.board_h and self.board_w in (15, 13, 19):
            stars = []
            n = self.board_w
            if n == 15:
                # 15x15 표준 별점 위치
                stars = [(3,3),(3,7),(3,11),(7,3),(7,7),(7,11),(11,3),(11,7),(11,11)]
            elif n == 13:
                stars = [(3,3),(3,6),(3,9),(6,3),(6,6),(6,9),(9,3),(9,6),(9,9)]
            elif n == 19:
                stars = [(3,3),(3,9),(3,15),(9,3),(9,9),(9,15),(15,3),(15,9),(15,15)]
            for (h,w) in stars:
                x,y = self.grid_to_pixel(h,w)
                pygame.draw.circle(self.surface, LINE_COLOR, (x,y), 4)

    def draw_stones(self, board):
        # board.states: {move_idx: player_id}
        # board.states_loc: 2D 색 기록(흑=1, 백=2)
        for move, player in board.states.items():
            h = move // board.width
            w = move % board.width
            x,y = self.grid_to_pixel(h,w)
            color = BLACK_COLOR if board.states_loc[h][w] == 1 else WHITE_COLOR
            pygame.draw.circle(self.surface, color, (x,y), STONE_RADIUS)
            # 돌 테두리
            pygame.draw.circle(self.surface, (0,0,0), (x,y), STONE_RADIUS, 2)

        # 마지막 수 하이라이트
        if getattr(board, "last_loc", -1) != -1:
            h,w = board.last_loc
            x,y = self.grid_to_pixel(h,w)
            pygame.draw.circle(self.surface, LAST_MOVE_CLR, (x,y), STONE_RADIUS+4, 2)

    def draw_forbidden(self, board):
        # 현재 차례가 흑일 때만 금수 표시
        if board.is_you_black():
            for (h,w) in getattr(board, "forbidden_locations", []):
                x,y = self.grid_to_pixel(h,w)
                s = STONE_RADIUS
                pygame.draw.line(self.surface, FORBID_CLR, (x-s, y-s), (x+s, y+s), 2)
                pygame.draw.line(self.surface, FORBID_CLR, (x+s, y-s), (x-s, y+s), 2)

    def draw_text(self, upper, lower=None):
        # 상단 상태 라벨
        if upper:
            surf = self.font_big.render(upper, True, TEXT_CLR)
            self.surface.blit(surf, (MARGIN_PX, 8))
        if lower:
            surf2 = self.font.render(lower, True, TEXT_CLR)
            self.surface.blit(surf2, (MARGIN_PX, 8 + 26))

    def draw(self, board, status_text=""):
        self.clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit(0)

        self.draw_board()
        self.draw_stones(board)
        self.draw_forbidden(board)

        # 턴/난이도/승패 등 상태표시는 status_text로 외부에서 전달
        self.draw_text(status_text)

        pygame.display.flip()

    def flash_forbidden(self, h, w):
        # 금수 자리를 빨간 박동처럼 잠깐 깜박
        x,y = self.grid_to_pixel(h,w)
        for _ in range(6):
            pygame.draw.circle(self.surface, (255, 180, 180), (x,y), STONE_RADIUS+8, 0)
            pygame.display.flip()
            pygame.time.delay(60)
            self.draw_board()
            pygame.display.flip()
            pygame.time.delay(60)


# -----------------------------
# 메인 실행 (사람 vs AI)
# -----------------------------
def run_pygame():
    # === 사용자 입력(콘솔) ===
    # 보드는 기존 코드와 동일 기본값(9x9 또는 15x15) 사용 가능
    width, height, n = 9, 9, 5
    print("이 오목 인공지능은 기본 9x9 환경에서 동작합니다. (원하면 코드 상단 width/height 수정)")
    print("현재 가능한 난이도(정책망의 학습 횟수) 목록 : [ 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000 ]")
    print("난이도를 입력하세요.")
    hard = int(input().strip())
    model_file = f'./model/policy_9_{hard}.model'
    if not os.path.exists(model_file):
        # colab/zip 경로 호환
        alt = f'./omok-ai-master/model/policy_9_{hard}.model'
        if os.path.exists(alt):
            model_file = alt
        else:
            print(f"모델 파일을 찾을 수 없습니다: {model_file} 또는 {alt}")
            sys.exit(1)

    print("자신이 선공(흑)인 경우에 0, 후공(백)인 경우에 1을 입력하세요.")
    order = int(input().strip())
    if order not in [0,1]:
        print("잘못된 입력. 종료합니다.")
        sys.exit(0)

    # === 보드/게임/정책망/MCTS ===
    board = Board(width=width, height=height, n_in_row=n)
    game = Game(board)

    policy_param = pickle.load(open(model_file, 'rb'), encoding='bytes')
    best_policy = PolicyValueNetNumpy(width, height, policy_param)
    mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=3, n_playout=1600)

    # === UI 초기화 ===
    ui = UIBoardRenderer(width, height)
    human = HumanPygame(ui)

    # === 대국 루프(Game.start_play를 내부에서 돌리되, pygame 렌더링을 끼워 넣음) ===
    # Game.start_play를 그대로 쓰면 블로킹 input 기반이라 화면 갱신이 느려질 수 있어서
    # 여기서 동일 로직을 살짝 풀어쓴다(렌더를 매 프레임 호출).
    board.init_board(order)
    p1, p2 = board.players
    human.set_player_ind(p1)
    mcts_player.set_player_ind(p2)
    players = {p1: human, p2: mcts_player}

    running = True
    winner_announced = False
    while running:
        # 흑 차례면 금수 갱신
        if board.is_you_black():
            board.set_forbidden()

        # 상태 텍스트
        if board.current_player == 1:
            upper = "흑: 플레이어" if order == 0 else "백: 플레이어"
            turn_txt = "당신의 차례입니다. (마우스로 착수)"
        else:
            upper = "백: AI" if order == 0 else "흑: AI"
            turn_txt = "AI가 수를 두는 중..."

        ui.draw(board, status_text=f"{upper} | {turn_txt}")

        current = board.get_current_player()
        pl = players[current]

        if current == 1:
            # 사람 입력(블로킹 되지만 내부에서 렌더 루프 돌며 이벤트 처리)
            move = pl.get_action(board)
        else:
            # AI 차례: 돌 계산 중에도 창 반응 유지
            # 소규모 딜레이로 UI가 '멈춘 것'처럼 보이지 않게 함
            # (mcts 내부에서 복사 상태로 탐색, 여유 루프)

            # 기존
            # acts, probs = pl.mcts.get_move_probs(board, temp=1e-3)
            # move = np.random.choice(acts, p=probs)
            # pl.mcts.update_with_move(-1)

            # 변경
            acts, probs = pl.mcts.get_move_probs(board, temp=1e-3)
            move = acts[int(np.argmax(probs))]          # 결정적으로 가장 강한 수 선택
            pl.mcts.update_with_move(-1)


        board.do_move(move)
        end, winner = board.game_end()
        if end:
            # 최종 한번 렌더 & 결과 표기
            who = None
            if winner == -1:
                who = "무승부"
            else:
                who = "플레이어" if players[winner] == human else "AI"

            msg = f"Game End! Winner: {who}" if winner != -1 else "Game End! Tie"
            ui.draw(board, status_text=msg)
            winner_announced = True

            # 결과를 잠깐 보여준 뒤 종료(또는 SPACE로 닫기)
            wait_ms = 1800
            s = pygame.time.get_ticks()
            while pygame.time.get_ticks() - s < wait_ms:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit(); sys.exit(0)
                    if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_SPACE, pygame.K_RETURN):
                        pygame.quit(); sys.exit(0)
                pygame.time.delay(30)
            running = False

    pygame.quit()
    if winner_announced:
        print("대국이 종료되었습니다.")


if __name__ == "__main__":
    run_pygame()
