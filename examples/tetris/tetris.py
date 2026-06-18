#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
控制台俄罗斯方块游戏
使用方向键控制：
    ← →   左右移动
    ↓     加速下落
    ↑     旋转
    空格   直接落底
    C     暂存/交换方块
    P     暂停
    Q     退出
"""

import random
import sys
import time

try:
    import curses
except ImportError:
    print("错误：需要 curses 模块。在 Windows 上请安装 'windows-curses'：")
    print("    pip install windows-curses")
    sys.exit(1)

# 游戏区域尺寸
BOARD_WIDTH = 10
BOARD_HEIGHT = 20

# 方块形状定义（每个方块包含4种旋转状态）
SHAPES = {
    "I": [
        [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
        [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
    ],
    "O": [
        [[1, 1], [1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1], [1, 1]],
    ],
    "T": [
        [[0, 1, 0], [1, 1, 1], [0, 0, 0]],
        [[0, 1, 0], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 1, 0], [0, 1, 0]],
    ],
    "S": [
        [[0, 1, 1], [1, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [0, 1, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 1], [1, 1, 0]],
        [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
    ],
    "Z": [
        [[1, 1, 0], [0, 1, 1], [0, 0, 0]],
        [[0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 0], [0, 1, 1]],
        [[0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ],
    "J": [
        [[1, 0, 0], [1, 1, 1], [0, 0, 0]],
        [[0, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 1], [0, 0, 1]],
        [[0, 1, 0], [0, 1, 0], [1, 1, 0]],
    ],
    "L": [
        [[0, 0, 1], [1, 1, 1], [0, 0, 0]],
        [[0, 1, 0], [0, 1, 0], [0, 1, 1]],
        [[0, 0, 0], [1, 1, 1], [1, 0, 0]],
        [[1, 1, 0], [0, 1, 0], [0, 1, 0]],
    ],
}

# 方块颜色对编号（curses 颜色对）
SHAPE_COLORS = {
    "I": 1,
    "O": 2,
    "T": 3,
    "S": 4,
    "Z": 5,
    "J": 6,
    "L": 7,
}


class Tetromino:
    """当前下落的方块"""

    def __init__(self, shape_name, x=None, y=None, rotation=0):
        self.shape_name = shape_name
        self.shape = SHAPES[shape_name]
        self.rotation = rotation
        self.x = x if x is not None else BOARD_WIDTH // 2 - len(self.shape[0]) // 2
        self.y = y if y is not None else 0

    def get_cells(self):
        """返回当前旋转状态下所有被占据的格子坐标 (相对坐标)"""
        cells = []
        pattern = self.shape[self.rotation % len(self.shape)]
        for r, row in enumerate(pattern):
            for c, val in enumerate(row):
                if val:
                    cells.append((r, c))
        return cells

    def get_absolute_positions(self):
        """返回方块在棋盘上的绝对坐标"""
        return [(self.y + r, self.x + c) for r, c in self.get_cells()]

    def rotate(self, clockwise=True):
        """旋转方块"""
        if clockwise:
            self.rotation = (self.rotation + 1) % len(self.shape)
        else:
            self.rotation = (self.rotation - 1) % len(self.shape)

    def clone(self):
        """创建副本，用于碰撞检测预演"""
        return Tetromino(self.shape_name, self.x, self.y, self.rotation)


class TetrisGame:
    """俄罗斯方块游戏主类"""

    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.board = [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        self.board_colors = [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        self.score = 0
        self.lines = 0
        self.level = 1
        self.game_over = False
        self.paused = False
        self.current = None
        self.next_piece_name = None
        self.hold_piece_name = None
        self.can_hold = True
        self.drop_timer = 0
        self.drop_interval = 1.0  # 初始下落间隔（秒）
        self.lock_delay = 0.5  # 触底锁定延迟
        self.lock_timer = None
        self.setup_colors()
        self.spawn_piece()

    def setup_colors(self):
        """初始化 curses 颜色"""
        curses.start_color()
        curses.use_default_colors()
        # 定义颜色对：方块颜色
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_CYAN)  # I
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_YELLOW)  # O
        curses.init_pair(3, curses.COLOR_MAGENTA, curses.COLOR_MAGENTA)  # T
        curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_GREEN)  # S
        curses.init_pair(5, curses.COLOR_RED, curses.COLOR_RED)  # Z
        curses.init_pair(6, curses.COLOR_BLUE, curses.COLOR_BLUE)  # J
        curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_WHITE)  # L
        # 边框和文字
        curses.init_pair(8, curses.COLOR_WHITE, -1)
        curses.init_pair(9, curses.COLOR_YELLOW, -1)
        curses.init_pair(10, curses.COLOR_RED, -1)

    def spawn_piece(self):
        """生成新方块"""
        if self.next_piece_name is None:
            self.next_piece_name = random.choice(list(SHAPES.keys()))
        shape_name = self.next_piece_name
        self.next_piece_name = random.choice(list(SHAPES.keys()))
        self.current = Tetromino(shape_name)
        self.lock_timer = None
        self.can_hold = True
        # 检查是否能放下，如果不能则游戏结束
        if self.check_collision(self.current):
            self.game_over = True

    def hold_piece(self):
        """暂存/交换当前方块，返回是否成功"""
        if not self.can_hold or self.game_over or self.paused:
            return False
        current_name = self.current.shape_name
        if self.hold_piece_name is None:
            self.hold_piece_name = current_name
            self.spawn_piece()
        else:
            self.hold_piece_name, current_name = current_name, self.hold_piece_name
            self.current = Tetromino(current_name)
            self.lock_timer = None
            if self.check_collision(self.current):
                self.game_over = True
        self.can_hold = False
        return True

    def check_collision(self, piece):
        """检查方块是否与边界或其他方块碰撞"""
        for r, c in piece.get_absolute_positions():
            if c < 0 or c >= BOARD_WIDTH or r >= BOARD_HEIGHT:
                return True
            if r >= 0 and self.board[r][c]:
                return True
        return False

    def lock_piece(self):
        """将当前方块固定到棋盘上"""
        color = SHAPE_COLORS[self.current.shape_name]
        for r, c in self.current.get_absolute_positions():
            if 0 <= r < BOARD_HEIGHT and 0 <= c < BOARD_WIDTH:
                self.board[r][c] = 1
                self.board_colors[r][c] = color
        self.clear_lines()
        self.spawn_piece()

    def clear_lines(self):
        """消除满行"""
        lines_cleared = 0
        new_board = []
        new_colors = []
        for r in range(BOARD_HEIGHT):
            if all(self.board[r]):
                lines_cleared += 1
            else:
                new_board.append(self.board[r][:])
                new_colors.append(self.board_colors[r][:])
        # 在顶部补空行
        while len(new_board) < BOARD_HEIGHT:
            new_board.insert(0, [0] * BOARD_WIDTH)
            new_colors.insert(0, [0] * BOARD_WIDTH)
        self.board = new_board
        self.board_colors = new_colors
        if lines_cleared > 0:
            self.lines += lines_cleared
            # 计分规则：1行100，2行300，3行600，4行1000
            points = {1: 100, 2: 300, 3: 600, 4: 1000}.get(
                lines_cleared, lines_cleared * 100
            )
            self.score += points * self.level
            self.level = self.lines // 10 + 1
            self.drop_interval = max(0.05, 1.0 - (self.level - 1) * 0.05)

    def move_piece(self, dx, dy):
        """移动方块，成功返回True"""
        test = self.current.clone()
        test.x += dx
        test.y += dy
        if not self.check_collision(test):
            self.current = test
            if dy > 0:
                self.lock_timer = None
            return True
        return False

    def rotate_piece(self, clockwise=True):
        """旋转方块，带踢墙（简单实现）"""
        test = self.current.clone()
        test.rotate(clockwise)
        if not self.check_collision(test):
            self.current = test
            self.lock_timer = None
            return True
        # 简单踢墙：尝试左右移动一格
        for kick in (-1, 1, -2, 2):
            test2 = test.clone()
            test2.x += kick
            if not self.check_collision(test2):
                self.current = test2
                self.lock_timer = None
                return True
        return False

    def hard_drop(self):
        """直接落底"""
        while self.move_piece(0, 1):
            pass
        self.lock_piece()

    def update(self, dt):
        """更新游戏状态，dt 为时间增量（秒）"""
        if self.game_over or self.paused:
            return
        self.drop_timer += dt
        if self.drop_timer >= self.drop_interval:
            self.drop_timer = 0
            if not self.move_piece(0, 1):
                # 无法下落，开始锁定计时
                if self.lock_timer is None:
                    self.lock_timer = 0
                else:
                    self.lock_timer += self.drop_interval
                    if self.lock_timer >= self.lock_delay:
                        self.lock_piece()

    def draw(self):
        """绘制游戏画面"""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()
        if h < 24 or w < 50:
            self.stdscr.addstr(0, 0, "终端太小，请调整窗口大小（至少 24x50）")
            self.stdscr.refresh()
            return

        # 绘制边框和棋盘
        board_x = 2
        board_y = 1
        # 上边框
        self.stdscr.addstr(
            board_y, board_x, "╔" + "══" * BOARD_WIDTH + "╗", curses.color_pair(8)
        )
        for r in range(BOARD_HEIGHT):
            self.stdscr.addstr(board_y + 1 + r, board_x, "║", curses.color_pair(8))
            for c in range(BOARD_WIDTH):
                self.stdscr.addstr(
                    board_y + 1 + r,
                    board_x + 1 + c * 2,
                    "  ",
                    (
                        curses.color_pair(self.board_colors[r][c])
                        if self.board[r][c]
                        else curses.color_pair(8)
                    ),
                )
            self.stdscr.addstr(
                board_y + 1 + r,
                board_x + 1 + BOARD_WIDTH * 2,
                "║",
                curses.color_pair(8),
            )
        # 下边框
        self.stdscr.addstr(
            board_y + 1 + BOARD_HEIGHT,
            board_x,
            "╚" + "══" * BOARD_WIDTH + "╝",
            curses.color_pair(8),
        )

        # 绘制当前方块
        if self.current and not self.game_over:
            color = SHAPE_COLORS[self.current.shape_name]
            for r, c in self.current.get_absolute_positions():
                if 0 <= r < BOARD_HEIGHT and 0 <= c < BOARD_WIDTH:
                    self.stdscr.addstr(
                        board_y + 1 + r,
                        board_x + 1 + c * 2,
                        "██",
                        curses.color_pair(color),
                    )

        # 绘制右侧信息面板
        info_x = board_x + 2 + BOARD_WIDTH * 2 + 3
        self.stdscr.addstr(
            board_y, info_x, "俄罗斯方块", curses.color_pair(9) | curses.A_BOLD
        )
        self.stdscr.addstr(
            board_y + 2, info_x, f"分数: {self.score}", curses.color_pair(8)
        )
        self.stdscr.addstr(
            board_y + 3, info_x, f"行数: {self.lines}", curses.color_pair(8)
        )
        self.stdscr.addstr(
            board_y + 4, info_x, f"等级: {self.level}", curses.color_pair(8)
        )

        # Hold 暂存方块显示
        self.stdscr.addstr(board_y + 6, info_x, "暂存(HOLD):", curses.color_pair(8))
        if self.hold_piece_name:
            hold_preview = SHAPES[self.hold_piece_name][0]
            hold_color = SHAPE_COLORS[self.hold_piece_name]
            for r, row in enumerate(hold_preview):
                line = ""
                for val in row:
                    line += "██" if val else "  "
                self.stdscr.addstr(
                    board_y + 7 + r,
                    info_x,
                    line,
                    curses.color_pair(hold_color) if any(row) else curses.color_pair(8),
                )

        # 下一个方块预览
        self.stdscr.addstr(board_y + 12, info_x, "下一个:", curses.color_pair(8))
        if self.next_piece_name:
            preview = SHAPES[self.next_piece_name][0]
            color = SHAPE_COLORS[self.next_piece_name]
            for r, row in enumerate(preview):
                line = ""
                for val in row:
                    line += "██" if val else "  "
                self.stdscr.addstr(
                    board_y + 13 + r,
                    info_x,
                    line,
                    curses.color_pair(color) if any(row) else curses.color_pair(8),
                )

        # 操作说明
        help_y = board_y + 19
        self.stdscr.addstr(help_y, info_x, "操作说明:", curses.color_pair(8))
        self.stdscr.addstr(help_y + 1, info_x, "← →  左右移动", curses.color_pair(8))
        self.stdscr.addstr(help_y + 2, info_x, "↓    加速下落", curses.color_pair(8))
        self.stdscr.addstr(help_y + 3, info_x, "↑    旋转", curses.color_pair(8))
        self.stdscr.addstr(help_y + 4, info_x, "空格  直接落底", curses.color_pair(8))
        self.stdscr.addstr(help_y + 5, info_x, "C    暂存/交换", curses.color_pair(8))
        self.stdscr.addstr(help_y + 6, info_x, "P    暂停", curses.color_pair(8))
        self.stdscr.addstr(help_y + 7, info_x, "Q    退出", curses.color_pair(8))

        # 游戏结束/暂停提示
        if self.game_over:
            msg = " 游戏结束！按 R 重新开始，Q 退出 "
            msg_y = board_y + BOARD_HEIGHT // 2
            msg_x = board_x + (BOARD_WIDTH * 2 - len(msg)) // 2
            self.stdscr.addstr(
                msg_y,
                msg_x,
                msg,
                curses.color_pair(10) | curses.A_BOLD | curses.A_REVERSE,
            )
        elif self.paused:
            msg = " 暂停中，按 P 继续 "
            msg_y = board_y + BOARD_HEIGHT // 2
            msg_x = board_x + (BOARD_WIDTH * 2 - len(msg)) // 2
            self.stdscr.addstr(
                msg_y,
                msg_x,
                msg,
                curses.color_pair(9) | curses.A_BOLD | curses.A_REVERSE,
            )

        self.stdscr.refresh()

    def reset(self):
        """重置游戏"""
        self.board = [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        self.board_colors = [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        self.score = 0
        self.lines = 0
        self.level = 1
        self.game_over = False
        self.paused = False
        self.next_piece_name = None
        self.hold_piece_name = None
        self.can_hold = True
        self.drop_interval = 1.0
        self.spawn_piece()

    def run(self):
        """主游戏循环"""
        self.stdscr.nodelay(True)
        self.stdscr.timeout(50)
        curses.curs_set(0)

        last_time = time.time()
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            key = self.stdscr.getch()

            if key == ord("q") or key == ord("Q"):
                break

            if self.game_over:
                if key == ord("r") or key == ord("R"):
                    self.reset()
                self.draw()
                time.sleep(0.05)
                continue

            if key == ord("p") or key == ord("P"):
                self.paused = not self.paused

            if not self.paused:
                if key == curses.KEY_LEFT:
                    self.move_piece(-1, 0)
                elif key == curses.KEY_RIGHT:
                    self.move_piece(1, 0)
                elif key == curses.KEY_DOWN:
                    if self.move_piece(0, 1):
                        self.score += 1
                elif key == curses.KEY_UP:
                    self.rotate_piece(clockwise=True)
                elif key == ord(" "):
                    self.hard_drop()
                elif key == ord("c") or key == ord("C"):
                    self.hold_piece()

                self.update(dt)

            self.draw()


def main(stdscr):
    game = TetrisGame(stdscr)
    game.run()


if __name__ == "__main__":
    curses.wrapper(main)
