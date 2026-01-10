"""
CIFAR-100 PDDL Planner - SUBOPTIMAL BFS VERSION
==========================================================
Breadth-First Search planner for CIFAR-100 domain.
"""

import re
import tempfile
import heapq
from typing import Set, List, Dict, Tuple, Optional, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict, deque

# ==================== CONSTANTS ====================

CIFAR_100_CLASSES = {
    "apple", "aquarium-fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn-mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple", "motorcycle", "mountain", "mouse",
    "mushroom", "oak", "orange", "orchid", "otter", "palm", "pear",
    "pickup-truck", "pine", "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal",
    "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet-pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow", "wolf", "woman", "worm"
}

TOOLS = {'knife', 'dslr'} #NOTE did Iran forget sponge? or take it out
LOCATIONS = {'lab', 'outdoors'}
IGNORED_KEYWORDS = {'item', 'location', 'object', 'objects', '-', 'agent', 'define', 'problem', 'domain', 'init', 'goal'} # NOTE why and when are these ignored?

# ==================== DATA STRUCTURES ====================

@dataclass(frozen=True)
class Predicate:
    name: str
    args: Tuple[str, ...]
    
    def __str__(self) -> str:
        return f"({self.name} {' '.join(self.args)})" if self.args else f"({self.name})"
    
    @staticmethod
    def from_string(s: str) -> 'Predicate':
        parts = s.strip().strip('()').split()
        return Predicate(parts[0], tuple(parts[1:]) if len(parts) > 1 else ())


@dataclass(frozen=True)
class Action:
    name: str
    parameters: Tuple[str, ...]
    preconditions: FrozenSet[Predicate]
    add_effects: FrozenSet[Predicate]
    del_effects: FrozenSet[Predicate]

    def __str__(self) -> str:
        return f"({self.name} {' '.join(self.parameters)})"
    
    def instantiate(self, bindings: Dict[str, str]) -> 'Action':
        """
        Creates a concrete Action instance by replacing variables with objects.
        bindings example: {'?x': 'apple', '?loc': 'lab'}
        """
        # Helper lambda to substitute args in a single predicate
        
        # TODO: Return a new Action with substituted parameters, preconditions, and effects
        sub = lambda p: Predicate(p.name, tuple(bindings.get(arg, arg) for arg in p.args))
        return Action(
            self.name,
            tuple(bindings.get(p,p) for p in self.parameters),
            frozenset(sub(p) for p in self.preconditions),
            frozenset(sub(p) for p in self.add_effects),
            frozenset(sub(p) for p in self.del_effects),
        )

        


@dataclass(frozen=True)
class State:
    predicates: FrozenSet[Predicate]
    
    def apply_action(self, action: Action) -> 'State':
        """
        Returns a NEW State by applying the action's effects.
        Recall: Next State = (Current State - Del Effects) + Add Effects
        """
        # TODO: Implement this method
        return State((self.predicates - action.del_effects) | action.add_effects) #NOTE union? 

    def is_applicable(self, action: Action) -> bool:
        """
        Checks if an action can be applied to this state.
        Recall: All preconditions must exist in the current state.
        """
        # TODO: Implement this method
        return action.preconditions.issubset(self.predicates)
        
    def satisfies(self, goal: FrozenSet[Predicate]) -> bool:
        return goal.issubset(self.predicates)


@dataclass(order=True)
class SearchNode:
    f_score: int
    state: State = field(compare=False)
    action: Optional[Action] = field(compare=False)
    parent: Optional['SearchNode'] = field(compare=False)
    g_score: int = field(compare=False)
    
    def get_plan(self) -> List[Action]:
        plan, node = [], self
        while node.parent:
            plan.append(node.action)
            node = node.parent
        return list(reversed(plan))

# ==================== PDDL PARSER ====================

class PDDLParser:
    last_discarded = set()
    
    @staticmethod
    def _extract_predicates(block: str) -> Set[Predicate]:
        """Extract predicates from a block, handling 'and' wrappers and nested parens."""
        block = block.strip()
        if block.startswith('(and'):
            depth, i = 0, 4
            while i < len(block):
                if block[i] == '(':
                    depth += 1
                elif block[i] == ')':
                    depth -= 1
                    if depth == -1:
                        block = block[4:i].strip()
                        break
                i += 1
        
        preds, depth, curr = set(), 0, []
        for c in block:
            if c == '(':
                depth += 1
                curr.append(c)
            elif c == ')':
                depth -= 1
                curr.append(c)
                if depth == 0 and curr:
                    s = ''.join(curr).strip()
                    if s and not s.startswith('(not') and not s.startswith('(forall'):
                        try:
                            preds.add(Predicate.from_string(s))
                        except:
                            pass
                    curr = []
            elif depth > 0:
                curr.append(c)
        return preds
    
    @staticmethod
    def _parse_effects(body: str) -> Tuple[Set[Predicate], Set[Predicate]]:
        """Parse add and delete effects from action body."""
        m = re.search(r':effect\s+(\(.*)', body, re.DOTALL | re.IGNORECASE)
        if not m:
            return set(), set()
        
        effect_block = m.group(1).strip()
        depth, i = 0, 0
        while i < len(effect_block):
            if effect_block[i] == '(':
                depth += 1
            elif effect_block[i] == ')':
                depth -= 1
                if depth == 0:
                    effect_block = effect_block[:i+1]
                    break
            i += 1
        
        block = effect_block[4:].strip().rstrip(')') if effect_block.startswith('(and') else effect_block
        adds, dels, depth, curr = set(), set(), 0, []
        
        for c in block:
            if c == '(':
                depth += 1
                curr.append(c)
            elif c == ')':
                depth -= 1
                curr.append(c)
                if depth == 0 and curr:
                    s = ''.join(curr).strip()
                    if s.startswith('(not'):
                        inner = re.search(r'\(not\s+(\(.*?\))\s*\)', s, re.IGNORECASE)
                        if inner:
                            try:
                                dels.add(Predicate.from_string(inner.group(1)))
                            except:
                                pass
                    else:
                        try:
                            adds.add(Predicate.from_string(s))
                        except:
                            pass
                    curr = []
            elif depth > 0:
                curr.append(c)
        
        return adds, dels
    
    @staticmethod
    def parse_domain(path: str) -> Dict[str, Action]:
        """Parse domain file and return action schemas."""
        with open(path) as f:
            content = f.read()
        
        actions = {}
        for m in re.finditer(r'\(:action\s+(\S+)(.*?)(?=\(:action|\Z)', content, re.DOTALL | re.IGNORECASE):
            name, body = m.groups()
            
            params = []
            if params_match := re.search(r':parameters\s+\((.*?)\)', body, re.DOTALL | re.IGNORECASE):
                params = re.findall(r'\?[\w-]+', params_match.group(1))
            
            preconds = set()
            if precond_match := re.search(r':precondition\s+(\(.*?)(?=\s*:effect|\Z)', body, re.DOTALL | re.IGNORECASE):
                preconds = PDDLParser._extract_predicates(precond_match.group(1))
            
            adds, dels = PDDLParser._parse_effects(body)
            actions[name] = Action(name, tuple(params), frozenset(preconds), frozenset(adds), frozenset(dels))
        
        return actions
    
    @staticmethod
    def parse_problem(path: str) -> Tuple[Dict[str, Set[str]], State, FrozenSet[Predicate]]:
        """Parse problem file and return objects, initial state, and goal."""
        with open(path) as f:
            content = f.read()
        
        objs = defaultdict(set)
        PDDLParser.last_discarded = set()
        
        if om := re.search(r':objects(.*?)(?=\(:init)', content, re.DOTALL | re.IGNORECASE):
            ob = re.sub(r';.*$', '', om.group(1), flags=re.MULTILINE).strip()
            for token in re.findall(r'[^\s():;]+', ob):
                t_lower = token.lower()
                if t_lower in CIFAR_100_CLASSES:
                    objs['item'].add(t_lower)
                elif t_lower in TOOLS:
                    objs['item'].add(t_lower)
                    objs['tool'].add(t_lower)
                elif t_lower in LOCATIONS:
                    objs['location'].add(t_lower)
                elif t_lower not in IGNORED_KEYWORDS:
                    PDDLParser.last_discarded.add(t_lower)
        
        for t in TOOLS:
            objs['item'].add(t)
            objs['tool'].add(t)
        
        init = set()
        if im := re.search(r':init(.*?)(?=\(:goal|\Z)', content, re.DOTALL | re.IGNORECASE):
            init = PDDLParser._extract_predicates(im.group(1))
        
        goal = set()
        if gm := re.search(r':goal\s+(\(.*\))', content, re.DOTALL | re.IGNORECASE):
            goal = PDDLParser._extract_predicates(gm.group(1))
        
        return dict(objs), State(frozenset(init)), frozenset(goal)

# ==================== ACTION GROUNDING ====================

class ActionGrounder:
    def __init__(self, schemas: Dict[str, Action], objects: Dict[str, Set[str]]):
        self.schemas = schemas
        self.all_items = sorted(objects.get('item', set()))
        self.cifar_objects = sorted(set(self.all_items) - TOOLS)
        self.locations = sorted(LOCATIONS)
    
    def ground_all(self) -> List[Action]:
        """Ground all action schemas with concrete objects."""
        grounded = []
        
        for name, schema in self.schemas.items():
            if name == 'walk-between-rooms':
                grounded.extend(self._ground_walk(schema))
            elif name in ['pick-up', 'put-down']:
                grounded.extend(self._ground_item_location(schema))
            elif name in ['stack', 'unstack']:
                grounded.extend(self._ground_stack(schema))
            elif name in ['slice-object', 'clean-object', 'take-photo']:
                grounded.extend(self._ground_object_action(schema))
        
        return grounded
    
    def _ground_walk(self, schema: Action) -> List[Action]:
        return [schema.instantiate({schema.parameters[0]: l1, schema.parameters[1]: l2})
                for l1 in self.locations for l2 in self.locations if l1 != l2]
    
    def _ground_item_location(self, schema: Action) -> List[Action]:
        return [schema.instantiate({schema.parameters[0]: item, schema.parameters[1]: loc})
                for item in self.all_items for loc in self.locations]
    
    def _ground_stack(self, schema: Action) -> List[Action]:
        return [schema.instantiate({schema.parameters[0]: top, schema.parameters[1]: bottom, schema.parameters[2]: loc})
                for top in self.all_items for bottom in self.all_items 
                if top != bottom for loc in self.locations]
    
    def _ground_object_action(self, schema: Action) -> List[Action]:
        return [schema.instantiate({schema.parameters[0]: obj, schema.parameters[1]: loc})
                for obj in self.cifar_objects for loc in self.locations]

# ==================== BFS PLANNER ====================

def bfs_search(initial: State, goal: FrozenSet[Predicate], actions: List[Action], 
               max_iter: int = 50000, verbose: bool = True) -> Optional[List[Action]]:
    
    # 1. Check if we are already there
    if initial.satisfies(goal):
        if verbose:
            print("Goal already satisfied!")
        return []
    
    # 2. Setup Frontier and Visited set
    start_node = SearchNode(0, initial, None, None, 0)
    frontier = deque([start_node]) # FIFO Queue
    visited = {initial}

    iters = 0

    if verbose:
        print("\nBFS search starting")

    while frontier and iters < max_iter:
        iters += 1

        if verbose and iters % 1000 == 0:
            print(f"Iter {iters}, frontier: {len(frontier)}, visited: {len(visited)}")

        curr = frontier.popleft()

        if curr.state.satisfies(goal):
            if verbose:
                print(f"Found plan! Iterations: {iters}, Plan length (Depth): {curr.g_score}")
            return curr.get_plan()

        for action in actions:
            if curr.state.is_applicable(action):
                new_state = curr.state.apply_action(action)
                if new_state not in visited:
                    visited.add(new_state)
                    new_node = SearchNode(curr.g_score + 1, new_state, action, curr, curr.g_score + 1)
                    frontier.append(new_node)


# ==================== A* PLANNER ====================
# NOTE Algorithm by Claude

def heuristic(state: State, goal: FrozenSet[Predicate]) -> int:
    """
    Domain-aware admissible heuristic for CIFAR-100 planning.
    
    Estimates minimum actions needed by analyzing what's required:
    - For (at ?item ?loc) goals: considers agent location, holding status, item location
    - For (documented ?item) or (cut-into-pieces ?item): considers tool and position
    
    # NOTE Algorithm by Claude
    """
    if goal.issubset(state.predicates):
        return 0
    
    # Extract current state info
    agent_loc = None
    holding = None
    hand_empty = False
    item_locations = {}  # item -> location
    
    for pred in state.predicates:
        if pred.name == 'agent-at' and pred.args:
            agent_loc = pred.args[0]
        elif pred.name == 'holding' and pred.args:
            holding = pred.args[0]
        elif pred.name == 'hand-empty':
            hand_empty = True
        elif pred.name == 'at' and len(pred.args) == 2:
            item_locations[pred.args[0]] = pred.args[1]
    
    total_h = 0
    
    for goal_pred in goal:
        if goal_pred in state.predicates:
            continue  # Already satisfied
        
        if goal_pred.name == 'at' and len(goal_pred.args) == 2:
            # Goal: (at ?item ?goal_loc)
            item, goal_loc = goal_pred.args
            item_loc = item_locations.get(item)
            
            if holding == item:
                # Holding the item: need to go to goal_loc (maybe) + put-down
                if agent_loc == goal_loc:
                    total_h += 1  # Just put-down
                else:
                    total_h += 2  # Walk + put-down
            elif item_loc is not None:
                # Item is somewhere: need to go there, pick up, go to goal, put down
                steps = 0
                if agent_loc != item_loc:
                    steps += 1  # Walk to item
                if not hand_empty:
                    steps += 1  # Put down what we're holding first
                steps += 1  # Pick up item
                if item_loc != goal_loc:
                    steps += 1  # Walk to goal location
                steps += 1  # Put down item
                total_h += steps
            else:
                # Item location unknown, estimate conservatively
                total_h += 2
                
        elif goal_pred.name == 'documented' and goal_pred.args:
            # Goal: (documented ?item) - need dslr + be at item location
            item = goal_pred.args[0]
            item_loc = item_locations.get(item)
            
            steps = 0
            if holding != 'dslr':
                if not hand_empty:
                    steps += 1  # Put down current item
                steps += 1  # Pick up dslr (assume we need to get it)
            if item_loc and agent_loc != item_loc:
                steps += 1  # Walk to item
            steps += 1  # Take photo
            total_h += steps
            
        elif goal_pred.name == 'cut-into-pieces' and goal_pred.args:
            # Goal: (cut-into-pieces ?item) - need knife + be at item location  
            item = goal_pred.args[0]
            item_loc = item_locations.get(item)
            
            steps = 0
            if holding != 'knife':
                if not hand_empty:
                    steps += 1  # Put down current item
                steps += 1  # Pick up knife
            if item_loc and agent_loc != item_loc:
                steps += 1  # Walk to item
            steps += 1  # Slice
            total_h += steps
            
        else:
            # Unknown goal type, add 1 as baseline
            total_h += 1
    
    return total_h


def astar_search(initial: State, goal: FrozenSet[Predicate], actions: List[Action], 
                 max_iter: int = 100000, verbose: bool = True) -> Optional[List[Action]]:
    """
    A* search algorithm for PDDL planning.
    
    Uses a priority queue ordered by f_score = g_score + h_score where:
    - g_score: actual cost (number of actions taken so far)
    - h_score: heuristic estimate (unsatisfied goal predicates)
    
    This is much more efficient than BFS because it expands promising
    states first, guided by the heuristic.
    
    # NOTE Algorithm by Claude
    """
    
    # 1. Check if we are already at the goal
    if initial.satisfies(goal):
        if verbose:
            print("Goal already satisfied!")
        return []
    
    # 2. Setup priority queue (min-heap) and visited set
    h_initial = heuristic(initial, goal)
    start_node = SearchNode(h_initial, initial, None, None, 0)
    
    # Priority queue: list of (f_score, tie_breaker, node)
    # tie_breaker ensures FIFO ordering for nodes with equal f_score
    frontier = [(start_node.f_score, 0, start_node)]
    heapq.heapify(frontier)
    
    # Track best g_score for each state (allows reopening with better path)
    best_g = {initial: 0}
    
    iters = 0
    tie_breaker = 1  # Incrementing counter for stable heap ordering
    
    if verbose:
        print("\nA* search starting")
    
    while frontier and iters < max_iter:
        iters += 1
        
        if verbose and iters % 1000 == 0:
            print(f"Iter {iters}, frontier: {len(frontier)}, visited: {len(best_g)}")
        
        # Pop node with lowest f_score
        _, _, curr = heapq.heappop(frontier)
        
        # Skip if we've found a better path to this state
        if curr.g_score > best_g.get(curr.state, float('inf')):
            continue
        
        # Goal check
        if curr.state.satisfies(goal):
            if verbose:
                print(f"Found plan! Iterations: {iters}, Plan length: {curr.g_score}")
            return curr.get_plan()
        
        # Expand successors
        for action in actions:
            if curr.state.is_applicable(action):
                new_state = curr.state.apply_action(action)
                new_g = curr.g_score + 1
                
                # Only explore if this is a better path to new_state
                if new_g < best_g.get(new_state, float('inf')):
                    best_g[new_state] = new_g
                    h = heuristic(new_state, goal)
                    f = new_g + h
                    new_node = SearchNode(f, new_state, action, curr, new_g)
                    heapq.heappush(frontier, (f, tie_breaker, new_node))
                    tie_breaker += 1
    
    if verbose:
        print(f"✗ No solution found. Iterations: {iters}")
    return None


# ==================== PROBLEM GENERATION ====================

def create_custom_problem(base_path: str, init_overrides: Set[str], 
                         goals: Set[str], name: str = "custom") -> str:
    """Generate custom problem file with conflict resolution."""
    with open(base_path) as f:
        content = f.read()
    
    objs_match = (re.search(r'(\(:objects.*?\)(?=\s*\(:init))', content, re.DOTALL | re.IGNORECASE) or
                  re.search(r'(\(:objects.*?\))', content, re.DOTALL | re.IGNORECASE))
    objs_sec = objs_match.group(1) if objs_match else "(:objects)"
    
    base_init = set()
    if base_init_match := re.search(r':init(.*?)(?=\(:goal|\Z)', content, re.DOTALL | re.IGNORECASE):
        base_init = PDDLParser._extract_predicates(base_init_match.group(1))
    
    user_init = {Predicate.from_string(p) for p in init_overrides}
    user_goal = {Predicate.from_string(p) for p in goals}
    
    # Build conflict map
    user_defs = defaultdict(set)
    for p in user_init:
        if p.args:
            user_defs[p.args[0]].add(p.name)
        user_defs['GLOBAL'].add(p.name)
    
    # Resolve conflicts
    final_init = set()
    for bp in base_init:
        if bp in user_init:
            continue
        
        conflict = _has_conflict(bp, user_defs)
        if not conflict:
            final_init.add(bp)
    
    final_init.update(user_init)
    
    problem_str = f"""(define (problem {name}-problem)
  (:domain cifar100-process)
  {objs_sec}
  (:init {chr(10).join(f"    {p}" for p in sorted(map(str, final_init)))} )
  (:goal (and {chr(10).join(f"      {p}" for p in sorted(map(str, user_goal)))} ))
)"""
    
    tf = tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', delete=False)
    tf.write(problem_str)
    tf.close()
    return tf.name


def _has_conflict(pred: Predicate, user_defs: Dict[str, Set[str]]) -> bool:
    """Check if base predicate conflicts with user definitions."""
    global_defs = user_defs['GLOBAL']
    
    if pred.name == 'hand-empty' and 'holding' in global_defs:
        return True
    if pred.name == 'holding' and 'hand-empty' in global_defs:
        return True
    if pred.name == 'agent-at' and 'agent-at' in global_defs:
        return True
    
    if pred.args:
        obj_defs = user_defs[pred.args[0]]
        if pred.name in ['at', 'on-top'] and {'at', 'on-top'} & obj_defs:
            return True
        if pred.name == 'clear' and {'at', 'on-top', 'clear'} & obj_defs:
            return True
        if pred.name == 'whole' and 'cut-into-pieces' in obj_defs:
            return True
        if pred.name == 'wet' and 'clean' in obj_defs:
            return True
        if pred.name == 'clean' and 'wet' in obj_defs:
            return True
    
    return False

# ==================== VALIDATION ====================

def validate_user_conditions(conditions: Set[str]):
    """Validate user input for syntax and vocabulary."""
    valid_vocab = CIFAR_100_CLASSES | TOOLS | LOCATIONS
    
    for s in conditions:
        s_clean = s.strip()
        if not (s_clean.startswith('(') and s_clean.endswith(')')):
            raise ValueError(f"SYNTAX ERROR: '{s}' is missing parentheses.")
        
        pred = Predicate.from_string(s_clean)
        for arg in pred.args:
            if arg not in valid_vocab:
                raise ValueError(
                    f"VOCABULARY ERROR: Unknown object '{arg}' in predicate '{s}'.\n"
                    f"   (Check spelling. Valid args are 100 CIFAR items, 2 tools, or 'lab'/'outdoors')"
                )

# ==================== VISUALIZATION ====================

def print_plan_execution(plan: List[Dict], object_name: str, initial_conditions: Set[str]):
    """Print formatted execution trace."""
    print("\n" + "="*80)
    print(f"📋 EXECUTION TRACE: {object_name.upper()}")
    print("="*80)
    print(f"{'STEP':<6} | {'ACTION':<35} | {'STATE CHANGES / INITIAL STATE':<40}")
    print("-" * 90)
    
    init_str = ", ".join(sorted(initial_conditions))
    print(f"{'0':<6} | {'(INITIAL STATE)':<35} | {init_str}")
    
    if not plan:
        print("-" * 90)
        print("✓ GOAL ALREADY ACHIEVED (No actions needed)")
        print("="*80 + "\n")
        return
    
    for step in plan:
        changes = []
        if step['added']:
            changes.append(f"++ {', '.join(step['added'])}")
        if step['removed']:
            changes.append(f"-- {', '.join(step['removed'])}")
        changes_str = " | ".join(changes)
        
        print(f"{step['step']:<6} | {step['action']:<35} | {changes_str}")
    
    print("-" * 90)
    print("✓ GOAL ACHIEVED")
    print("="*80 + "\n")
