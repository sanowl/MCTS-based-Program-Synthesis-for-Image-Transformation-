import random
import math
import copy
import time

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0
        self.action = action
        self.untried_actions = self.state.get_possible_actions()
    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.apply(action)
        child_node = Node(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node
    def best_child(self, c_param=1.0):
        weights = [
            (child.reward / child.visits) + c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[weights.index(max(weights))]
    def tree_policy(self):
        current_node = self
        while not current_node.state.is_terminal():
            if current_node.untried_actions:
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
    def backup(self, reward):
        current_node = self
        while current_node is not None:
            current_node.visits += 1
            current_node.reward += reward
            current_node = current_node.parent
class State:
    def __init__(self, program=None):
        self.program = program or []

    def apply(self, action):
        new_program = self.program + [action]
        return State(new_program)

    def is_terminal(self):
        return len(self.program) >= MAX_PROGRAM_LENGTH

    def get_possible_actions(self):
        return ACTIONS
    def evaluate(self, input_output_pairs):
        total_loss = 0
        for input_data, output_data in input_output_pairs:
            try:
                result = self.run_program(input_data)
                total_loss += compute_loss(result, output_data)
            except:
                total_loss += float('inf')
        return -total_loss
    def run_program(self, data):
        env = {'data': copy.deepcopy(data)}
        for action in self.program:
            action.execute(env)
        return env['data']
class Action:
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def execute(self, env):
        self.func(env)
def compute_loss(result, target):
    if len(result) != len(target) or any(len(r) != len(t) for r, t in zip(result, target)):
        return float('inf')
    loss = sum(r_cell != t_cell for r_row, t_row in zip(result, target) for r_cell, t_cell in zip(r_row, t_row))
    return loss
def mcts(root, input_output_pairs, time_limit):
    start_time = time.time()
    while time.time() - start_time < time_limit:
        leaf = root.tree_policy()
        reward = leaf.state.evaluate(input_output_pairs)
        leaf.backup(reward)
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.state.program
def op_rotate_90(env):
    env['data'] = [list(reversed(col)) for col in zip(*env['data'])]
def op_rotate_180(env):
    env['data'] = [row[::-1] for row in env['data'][::-1]]
def op_mirror_horizontal(env):
    env['data'] = [row[::-1] for row in env['data']]
def op_mirror_vertical(env):
    env['data'] = env['data'][::-1]
def op_translate(env, dx, dy):
    h, w = len(env['data']), len(env['data'][0])
    new_data = [[0]*w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                new_data[ny][nx] = env['data'][y][x]
    env['data'] = new_data

def op_change_color(env, from_color, to_color):
    env['data'] = [[to_color if cell == from_color else cell for cell in row] for row in env['data']]
def generate_actions():
    actions = [
        Action('rotate_90', op_rotate_90),
        Action('rotate_180', op_rotate_180),
        Action('mirror_horizontal', op_mirror_horizontal),
        Action('mirror_vertical', op_mirror_vertical),
    ]
    translations = [(-1,0), (1,0), (0,-1), (0,1)]
    for dx, dy in translations:
        actions.append(Action(f'translate_{dx}_{dy}', lambda env, dx=dx, dy=dy: op_translate(env, dx, dy)))
    colors = [1,2,3,4,5,6,7,8,9]
    for from_color in colors:
        for to_color in colors:
            if from_color != to_color:
                actions.append(Action(f'change_color_{from_color}_to_{to_color}', lambda env, f=from_color, t=to_color: op_change_color(env, f, t)))
    return actions

ACTIONS = generate_actions()
MAX_PROGRAM_LENGTH = 4

input_output_pairs = [
    (
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ],
        [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ]
    ),
    (
        [
            [0, 2, 0],
            [2, 0, 2],
            [0, 2, 0]
        ],
        [
            [0, 2, 0],
            [2, 0, 2],
            [0, 2, 0]
        ]
    )
]

initial_state = State()
root_node = Node(initial_state)
best_program = mcts(root_node, input_output_pairs, time_limit=10)
print("Best Program:")
for action in best_program:
    print(action.name)
print("Result:")
test_input = input_output_pairs[0][0]
test_state = State(best_program)
result = test_state.run_program(test_input)
for row in result:
    print(row)

print("Expected:")
for row in input_output_pairs[0][1]:
    print(row)
