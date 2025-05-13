from template import Agent
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import copy
from collections import defaultdict, deque

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
for row in range(10):
    for col in range(10):
        COORDS[BOARD[row][col]].append((row,col))

class SequenceNet(nn.Module):
    def __init__(self, _id):
        super(SequenceNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256 * 10 * 10, 512)
        self.fc_policy = nn.Linear(512, 1000)  # Max possible actions (overestimate)
        self.fc_value = nn.Linear(512, 1)
        
        self.load(_id)
        
        self.criterion_policy = nn.CrossEntropyLoss()
        self.criterion_value = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 256 * 10 * 10)
        x = F.relu(self.fc1(x))
        
        policy = self.fc_policy(x)
        value = torch.tanh(self.fc_value(x))
        return policy, value
    
    def save(self, id):
        torch.save(self.state_dict(), f'agent_{id}_alphazero.dat')
    
    def load(self, id):
        try:
            self.load_state_dict(torch.load(f'agent_{id}_alphazero.dat'))
        except FileNotFoundError:
            pass
    
    def train_step(self, states, action_probs, values):
        self.optimizer.zero_grad()
        states = torch.tensor(states, dtype=torch.float32)
        action_probs = torch.tensor(action_probs, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        
        policy, value = self(states)
        policy_loss = self.criterion_policy(policy, action_probs)
        value_loss = self.criterion_value(value.squeeze(), values)
        loss = policy_loss + value_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()

class MCTSNode:
    def __init__(self, game_state, agent_id, parent=None, action=None, prior=0):
        self.game_state = copy.deepcopy(game_state)
        self.agent_id = agent_id
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.prior = prior
        self.unvisited_actions = self.get_legal_actions()
    
    def get_legal_actions(self):
        actions = []
        agent_state = self.game_state.agents[self.agent_id]
        chips = self.game_state.board.chips
        hand = agent_state.hand
        draft = self.game_state.board.draft
        
        if not agent_state.trade:
            for card in hand:
                if card[0] != 'j':
                    free_spaces = 0
                    for r, c in COORDS[card]:
                        if chips[r][c] == '_':
                            free_spaces += 1
                    if not free_spaces:
                        for draft_card in draft:
                            actions.append({'play_card': card, 'draft_card': draft_card, 'type': 'trade', 'coords': None})
            if actions:
                actions.append({'play_card': None, 'draft_card': None, 'type': 'trade', 'coords': None})
        
        for card in hand:
            if card in ['jd', 'jc']:
                for r in range(10):
                    for c in range(10):
                        if chips[r][c] == '_':
                            for draft_card in draft:
                                actions.append({'play_card': card, 'draft_card': draft_card, 'type': 'place', 'coords': (r, c)})
            elif card in ['jh', 'js']:
                for r in range(10):
                    for c in range(10):
                        if chips[r][c] == agent_state.opp_colour:
                            for draft_card in draft:
                                actions.append({'play_card': card, 'draft_card': draft_card, 'type': 'remove', 'coords': (r, c)})
            else:
                for r, c in COORDS[card]:
                    if chips[r][c] == '_':
                        for draft_card in draft:
                            actions.append({'play_card': card, 'draft_card': draft_card, 'type': 'place', 'coords': (r, c)})
        return actions
    
    def is_terminal(self):
        scores = {agent.colour: agent.completed_seqs for agent in self.game_state.agents}
        return scores[self.game_state.agents[self.agent_id].colour] >= 2 or scores[self.game_state.agents[self.agent_id].opp_colour] >= 2 or not self.game_state.board.draft
    
    def expand(self, action, prior):
        next_state = self.apply_action(action)
        child = MCTSNode(next_state, self.agent_id, self, action, prior)
        self.children.append(child)
        self.unvisited_actions.remove(action)
        return child
    
    def apply_action(self, action):
        state = copy.deepcopy(self.game_state)
        plr_state = state.agents[self.agent_id]
        card = action['play_card']
        draft = action['draft_card']
        
        if card:
            plr_state.hand.remove(card)
            state.deck.discards.append(card)
            state.board.draft.remove(draft)
            plr_state.hand.append(draft)
            state.board.draft.extend(state.deck.deal())
        
        if action['type'] == 'trade':
            plr_state.trade = True
        elif action['type'] == 'place':
            r, c = action['coords']
            state.board.chips[r][c] = plr_state.colour
        elif action['type'] == 'remove':
            r, c = action['coords']
            state.board.chips[r][c] = '_'
        
        plr_state.trade = False if action['type'] != 'trade' else plr_state.trade
        return state
    
    def backpropagate(self, value):
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.backpropagate(-value)

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.net = SequenceNet(_id)
        self.memory = deque(maxlen=10000)
        self.c_puct = 1.0
        self.num_simulations = 100
        self.temperature = 1.0
        self.batch_size = 32
    
    def state_to_tensor(self, game_state):
        chips = game_state.board.chips
        colour = game_state.agents[self.id].colour
        opp_colour = game_state.agents[self.id].opp_colour
        tensor = np.zeros((4, 10, 10), dtype=np.float32)
        
        for r in range(10):
            for c in range(10):
                if chips[r][c] == colour:
                    tensor[0, r, c] = 1
                elif chips[r][c] == opp_colour:
                    tensor[1, r, c] = 1
                elif chips[r][c] == '_':
                    tensor[2, r, c] = 1
                elif chips[r][c] == 'jk':
                    tensor[3, r, c] = 1
        return tensor
    
    def run_mcts(self, game_state):
        root = MCTSNode(game_state, self.id)
        state_tensor = torch.tensor([self.state_to_tensor(game_state)], dtype=torch.float32)
        policy, _ = self.net(state_tensor)
        policy = F.softmax(policy, dim=1).detach().numpy()[0]
        
        legal_actions = root.unvisited_actions
        action_indices = list(range(len(legal_actions)))
        priors = {action: policy[i] for i, action in enumerate(legal_actions)}
        
        for _ in range(self.num_simulations):
            node = root
            while node.children:
                node = self.select_child(node)
            
            if not node.is_terminal():
                state_tensor = torch.tensor([self.state_to_tensor(node.game_state)], dtype=torch.float32)
                policy, value = self.net(state_tensor)
                policy = F.softmax(policy, dim=1).detach().numpy()[0]
                value = value.item()
                
                for action in node.unvisited_actions:
                    node.expand(action, priors.get(action, 0.1))
            else:
                value = self.evaluate_terminal(node)
            
            node.backpropagate(value)
        
        visits = np.array([child.visits for child in root.children])
        pi = visits / visits.sum()
        if self.training:
            action_idx = np.random.choice(len(root.children), p=pi ** (1 / self.temperature))
        else:
            action_idx = np.argmax(visits)
        
        action = root.children[action_idx].action
        if self.training:
            self.memory.append((self.state_to_tensor(game_state), pi, None))
        return action
    
    def select_child(self, node):
        total_visits = sum(child.visits for child in node.children)
        best_score = -float('inf')
        best_child = None
        
        for child in node.children:
            q = child.value / (child.visits + 1e-8)
            u = self.c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        return best_child
    
    def evaluate_terminal(self, node):
        scores = {agent.colour: agent.completed_seqs for agent in node.game_state.agents}
        my_score = scores[node.game_state.agents[self.id].colour]
        opp_score = scores[node.game_state.agents[self.id].opp_colour]
        if my_score >= 2:
            return 1.0
        elif opp_score >= 2:
            return -1.0
        return 0.0
    
    def SelectAction(self, actions, game_state):
        self.training = False
        return self.run_mcts(game_state)
    
    def train(self, num_games=100):
        self.training = True
        for _ in range(num_games):
            game_state = copy.deepcopy(self.initial_game_state)  # Assume initial state is provided
            while not self.is_game_over(game_state):
                action = self.run_mcts(game_state)
                game_state = self.apply_action(game_state, action)
            
            final_value = self.evaluate_terminal(MCTSNode(game_state, self.id))
            for state, pi, _ in self.memory:
                self.memory.append((state, pi, final_value))
            
            if len(self.memory) >= self.batch_size:
                batch = random.sample(self.memory, self.batch_size)
                states, action_probs, values = zip(*batch)
                loss = self.net.train_step(states, action_probs, values)
    
    def save_net(self):
        self.net.save(self.id)