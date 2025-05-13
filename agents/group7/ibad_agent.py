
from template import Agent
import random
from collections import defaultdict
import copy

BOARD = [['jk','2s','3s','4s','5s','6s','7s','8s','9s','jk'],
         ['6c','5c','4c','3c','2c','ah','kh','qh','th','ts'],
         ['7c','as','2d','3d','4d','5d','6d','7d','9h','qs'],
         ['8c','ks','6c','5c','4c','3c','2c','8d','8h','ks'],
         ['9c','qs','7c','6h','5h','4h','ah','9d','7h','as'],
         ['tc','ts','8c','7h','2h','3h','kh','td','6h','2d'],
         ['qc','9s','9c','8h','9h','th','qh','qd','5h','3d'],
         ['kc','8s','tc','qc','kc','ac','ad','kd','4h','4d'],
         ['ac','7s','6s','5s','4s','3s','2s','2h','3h','5d'],
         ['jk','ad','kd','qd','td','9d','8d','7d','6d','jk']]

COORDS = defaultdict(list)
for r in range(10):
    for c in range(10):
        COORDS[BOARD[r][c]].append((r, c))

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.random_moves = 0
        self.max_random_moves = random.randint(5, 10)

    def SelectAction(self, actions, game_state):
        agent = game_state.agents[self.id]
        chips = game_state.board.chips
        self.colour = agent.colour
        self.opp_colour = agent.opp_colour

        # Rule 1: Early Game Random Play
        if self.random_moves < self.max_random_moves and not self.has_opponent_sequence(game_state):
            self.random_moves += 1
            return random.choice([a for a in actions if a['type'] == 'place'])

        # Rule 2: Use One-Eyed Jack to Break Opponent 4-In-A-Row
        for a in actions:
            if a['type'] == 'remove' and self.breaks_four_in_a_row(chips, a['coords'], self.opp_colour):
                return a

        # Rule 3: Use Two-Eyed Jack to Complete or Extend Our 4-In-A-Row
        for a in actions:
            if a['type'] == 'place' and a['play_card'] in ['jd', 'jc'] and self.completes_our_four(chips, a['coords']):
                return a

        # Rule 4: Exploit opportunities to build on 2-3 chips
        promising_moves = self.find_promising_moves(actions, chips)
        if promising_moves:
            return random.choice(promising_moves)

        # Rule 5: Exploration fallback
        return random.choice(actions)

    def has_opponent_sequence(self, game_state):
        for agent in game_state.agents:
            if agent.opp_colour == self.colour and agent.completed_seqs > 0:
                return True
        return False

    def breaks_four_in_a_row(self, chips, coords, colour):
        r, c = coords
        return self.count_consecutive(chips, r, c, colour) >= 4

    def completes_our_four(self, chips, coords):
        r, c = coords
        return self.count_consecutive(chips, r, c, self.colour) >= 4

    def count_consecutive(self, chips, r, c, colour):
        count = 0
        for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
            temp = 1
            for i in range(1, 5):
                nr, nc = r + dr*i, c + dc*i
                if 0 <= nr < 10 and 0 <= nc < 10 and chips[nr][nc] == colour:
                    temp += 1
                else:
                    break
            count = max(count, temp)
        return count

    def find_promising_moves(self, actions, chips):
        promising = []
        for a in actions:
            if a['type'] == 'place':
                r, c = a['coords']
                count = self.count_consecutive(chips, r, c, self.colour)
                if count >= 2:
                    promising.append(a)
        return promising
