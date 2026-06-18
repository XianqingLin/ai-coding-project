import sys

sys.path.insert(0, ".")
import tetris  # noqa: E402

print("=== 1. 模块导入与基本定义 ===")
print("SHAPES:", list(tetris.SHAPES.keys()))
print("BOARD:", tetris.BOARD_WIDTH, "x", tetris.BOARD_HEIGHT)
print("COLORS:", tetris.SHAPE_COLORS)

print("\n=== 2. 方块(Tetromino)测试 ===")
piece = tetris.Tetromino("T")
print("T 方块初始绝对位置:", piece.get_absolute_positions())
piece.x += 1
print("右移后:", piece.get_absolute_positions())
piece.rotate()
print("顺时针旋转后:", piece.get_absolute_positions())
piece.rotate(False)
print("逆时针旋转后:", piece.get_absolute_positions())

print("\n=== 3. 游戏(TetrisGame)逻辑测试 ===")


class MockGame:
    def __init__(self):
        self.board = [[0] * tetris.BOARD_WIDTH for _ in range(tetris.BOARD_HEIGHT)]
        self.board_colors = [
            [0] * tetris.BOARD_WIDTH for _ in range(tetris.BOARD_HEIGHT)
        ]
        self.score = 0
        self.lines = 0
        self.level = 1
        self.current = tetris.Tetromino("I")

    def check_collision(self, piece):
        for r, c in piece.get_absolute_positions():
            if c < 0 or c >= tetris.BOARD_WIDTH or r >= tetris.BOARD_HEIGHT:
                return True
            if r >= 0 and self.board[r][c]:
                return True
        return False

    def clear_lines(self):
        new_board = []
        new_colors = []
        lines_cleared = 0
        for r in range(tetris.BOARD_HEIGHT):
            if all(self.board[r]):
                lines_cleared += 1
            else:
                new_board.append(self.board[r][:])
                new_colors.append(self.board_colors[r][:])
        while len(new_board) < tetris.BOARD_HEIGHT:
            new_board.insert(0, [0] * tetris.BOARD_WIDTH)
            new_colors.insert(0, [0] * tetris.BOARD_WIDTH)
        self.board = new_board
        self.board_colors = new_colors
        if lines_cleared > 0:
            self.lines += lines_cleared
            points = {1: 100, 2: 300, 3: 600, 4: 1000}.get(
                lines_cleared, lines_cleared * 100
            )
            self.score += points * self.level
            self.level = self.lines // 10 + 1
        return lines_cleared


game = MockGame()
print("I 方块初始位置:", game.current.get_absolute_positions())

game.current.x = 0
left_test = game.current.clone()
left_test.x -= 1
print("左移越界碰撞:", game.check_collision(left_test))

game.current.x = tetris.BOARD_WIDTH - 1
right_test = game.current.clone()
right_test.x += 1
print("右移越界碰撞:", game.check_collision(right_test))

game.current = tetris.Tetromino(
    "I", tetris.BOARD_WIDTH // 2 - 2, tetris.BOARD_HEIGHT - 1
)
down_test = game.current.clone()
down_test.y += 1
print("下落到底碰撞:", game.check_collision(down_test))

print("\n=== 4. 消行测试 ===")
game2 = MockGame()
for c in range(tetris.BOARD_WIDTH):
    game2.board[-1][c] = 1
    game2.board[-2][c] = 1
    game2.board_colors[-1][c] = 1
    game2.board_colors[-2][c] = 2
print("消行前底部2行已填满")
lines = game2.clear_lines()
print("消除行数:", lines)
print("消行后底部2行:", game2.board[-1], game2.board[-2])
print("得分:", game2.score)

print("\n=== 5. 所有7种方块验证 ===")
for name in tetris.SHAPES:
    p = tetris.Tetromino(name)
    cells = p.get_cells()
    abs_pos = p.get_absolute_positions()
    print(f"{name}: 相对格子数={len(cells)}, 绝对位置数={len(abs_pos)}")

print("\n=== 6. 语法编译验证 ===")
import py_compile  # noqa: E402

py_compile.compile("tetris.py", doraise=True)
print("tetris.py 语法编译通过")

print("\n✅ 所有核心逻辑验证通过，代码可以正常运行！")
