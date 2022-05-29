#!/usr/bin/env python
# coding: utf-8

# [<img src=https://img.youtube.com/vi/FnV0thLS1Fs/0.jpg width = 400>](https://www.youtube.com/watch?v=FnV0thLS1Fs)

# pixyz  
# last update 2022 05 03  
# ゆっくりしていってね！  

# **霊夢:今回はコンペ理解のために、主催者が書いてくれたコードを日本語訳してみたよ。**
# 
# **魔理沙:他に気になったことがあったら突っ込んでいくぞ。**
# 
# **Reimu: This time, I translated the code written by the organizer into Japanese to understand the competition.**
# 
# **Marisa: If you have any other concerns, I'll dig in.**

# ## Contents
# 
# * [**Official guide**](#Official_guide)
# 
# * [**addtest**](#add_test)
# 
# * [**Helperfunction**](#Helper_function)
# 
# * [**public agent**](#public_agent)

# **魔理沙:Helper functionについても書いたぜ**

# **霊夢:まずは、公式ガイドの日本語訳から始めるぜ**
# 
# https://www.kaggle.com/code/bovard/kore-intro-i-the-basics

# # Official_guide

# ## Welcome Commander!

# このノートブックでは、Kore艦隊のルールと環境を紹介します。
# 
# This notebook will give you an introduction to the Kore Fleet rules and environment

# In[ ]:


get_ipython().run_cell_magic('capture', '', '# install the latest version of kaggle_environments\n!pip install --upgrade kaggle_environments')


# In[ ]:


from kaggle_environments import make
env = make("kore_fleets", debug=True)
print(env.name, env.version)


# ## Let's start with the shipyard
# 
# 造船所は、アクションを割り当てる唯一のユニットです。 毎ターン、一つの造船所でどちらかをすることができます 
# 
# 1. 船を建造する。
# 2. 艦隊を進水させる。
# 
# では実際に、造船所がどのようになものかを見てみましょう
# 
# The shipyard it the only unit you assign actions to. You can either
# 
# 1. build more ships
# 2. launch a fleet of ships
# 
# but for now let's just see what a shipyard looks like

# In[ ]:


get_ipython().run_cell_magic('writefile', 'do_nothing.py', '# First we will make a do_nothing player to observe the game board\ndef do_nothing():\n    pass')


# In[ ]:


env.run(["/kaggle/working/do_nothing.py"])
env.render(mode="ipython", width=1000, height=800)


# # It did nothing!
# あなたの造船所（青色）は、0隻の船を持っています。（左上隅の0で示されています）、
# このままでは、何もしません。
# 
# 
# 1. KoreのあるマスのKore量が増えました（星が大きくなりました）
# 
# 2. 造船所の右下の数はゆっくりと10に増えました。この数は、その造船所でそのターンを建造できる船の最大数を表しています。 造船所が交代するたびに、この数字は0にリセットされます！
# 
# Your shipyard (in blue) had 0 ships the entire game (denoted by the 0 in the upper left hand corner) and did nothing.
# 
# Notice a few things did happen!
# 
# 1. the kore amount on the tiles with kore grew (the stars got bigger)
# 2. the number in the bottom righthand tile grew slowly to 10
# This number represents the number of ships that can be built a turn at that shipyard! This number is reset to 0 every time a shipyard changes hands!

# **魔理沙:　画面左側に表示されているKoreは造船所が持っている資源の数、Cargoは艦隊が持っている資源の数、Shipsは味方全体が持っている船のかずだぜ。**
# 
# **霊夢:　フィールドに散らばっている資源はターンが経過すると大きくなるんだね。どんな割合で大きくなっていくんだ？**
# 
# **魔理沙:　Koreはそのマスに艦隊がいなかった場合、毎ターン2％増加してくぜ。**
# 
# **Marisa: Kore displayed on the left side of the screen is the number of resources owned by the shipyard, Cargo is the number of resources owned by the fleet, and Ships is the number of ships owned by all allies.**
# 
# **Reimu: The resources scattered in the field grow larger as the turn passes. At what rate will it grow?**
# 
# **Marisa: Kore will increase by 2% each turn if there is no fleet in that square.**

# ## Building some ships
# 
# それでは、毎ターン船を建造しようとする簡単なエージェントを作りましょう！
# 
# 船の建造には10Koreの費用がかかり、Koreはゲーム開始時500あるので、50ターン船の建造が行われるでしょう。
# 
# Now let's make a simple player that tries to build a ship every turn!
# 
# Note that since ships cost 10 kore to build and we start with 500, we will expect to build ships for 50 turns!

# In[ ]:


get_ipython().run_cell_magic('writefile', 'builder.py', '# this one builds ships!\n\nfrom kaggle_environments.envs.kore_fleets.helpers import *\n\ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n    \n    # loop through all shipyards you control\n    for shipyard in me.shipyards:\n        # build a ship!\n        if kore_left >= spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n            shipyard.next_action = action\n\n    return me.next_actions')


# In[ ]:


env.run(["/kaggle/working/builder.py"])
env.render(mode="ipython", width=1000, height=800)


# **霊夢：一つの造船所で1ターンに船は何隻造船できるんだ？**
# 
# **魔理沙:造船所ができてからターンが経過するごとに作れる船の数が増えていって、最大で1ターンに10隻まで作ることができるぜ。具体的には、造船所が現在Y隻の船をスポーンできるようになったとすると、それからY^2 + 1ターン経過した後、Y+1隻の船を生産できるようになるぜ。次の表をみてくれだぜ。**
# 
# **Reimu: How many ships can be built in one turn at one shipyard?**
# 
# **Marisa: The number of ships that can be built increases with each turn since the shipyard was built, and you can build up to 10 ships per turn. Specifically, if the shipyard is now able to spawn Y ships, then Y ^ 2 + 1 turn later will be able to produce Y + 1 ships. See the table below.**
# 
# | Turns Controlled | Spawn Maximum | 
# | ---- | ---- |
# | 0 | 1 |
# | 2 | 2 |
# | 7 | 3 |
# | 17 | 4 |
# | 34 | 5 |
# | 60 | 6 |
# | 97 | 7 |
# | 147 | 8 |
# | 212 | 9 |
# | 294 | 10 |

# # We built 50 ships!
# 
# 造船所の左上にある数字が50に増え、Koreがなくなると止まります。 UIは、青いプレーヤーにも50隻の船を制御しないことを示していることに注意してください。
# 
# You saw the number in the upper left of the shipyard increment to 50 and stop when we ran out of kore! Note that the ui also shows the blue player also no contorls 50 ships.

# # To the stars!
# 
# 次に、艦隊を進水させてみましょう。 艦隊は一連の指示で発射されます（北南東西のいづれかに、毎ターン移動します）。 とりあえず北に艦隊を発射しましょう。
# 
# この艦隊が途中でKoreを拾い、より多くの船を建造できるようになるでしょう！
# 
# Next lets try launching a fleet. A fleet is launched with a series of instructions (go North, South, East, West and follows those, moving every turn). For now let's now worry about that and just launch a fleet to the north.
# 
# We expect this fleet to pick up some kore along the way, enabling us to build more ships!

# In[ ]:


get_ipython().run_cell_magic('writefile', 'launcher.py', '   \nfrom kaggle_environments.envs.kore_fleets.helpers import *\n\ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n\n    for shipyard in me.shipyards:\n        if kore_left >= spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n            shipyard.next_action = action\n        elif shipyard.ship_count > 0:\n            direction = Direction.NORTH\n            action = ShipyardAction.launch_fleet_with_flight_plan(2, direction.to_char())\n            shipyard.next_action = action\n\n    return me.next_actions')


# In[ ]:


env.run(["/kaggle/working/launcher.py"])
env.render(mode="ipython", width=1000, height=800)


# **魔理沙:艦隊は、造船所から発射されたターンにルートが決まっていて、後から変更することができないって事か？**
# 
# **Marisa: Does the fleet have a route on the turn it fires from the shipyard and can't be changed later?**

# # Flight Control
# 
# 今、あなたはこの戦略の弱点に気付いているかもしれません、これではKoreが効率的に回収できません！ 私たちは、これまで艦隊が行ったことのない場所に艦隊が行くことを可能にする飛行計画を立てなければなりません！
# 
# Now you might notice a certain weakness of our strategy, it doesn't mine out any of the board! Now we have to make a flight plan which allows our fleet go where no fleet has ever gone before!

# In[ ]:


from random import randint
randint(2, 9)


# In[ ]:


get_ipython().run_cell_magic('writefile', 'pilot.py', '   \nfrom kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\n# a flight plan\ndef build_flight_plan(dir_idx, size):\n    flight_plan = ""\n    for i in range(4):\n        flight_plan += Direction.from_index((dir_idx + i) % 4).to_char()\n        if not i == 3:\n            flight_plan += str(size)\n    return flight_plan\n\ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n\n    for shipyard in me.shipyards:\n        if shipyard.ship_count >= 50:\n            flight_plan = build_flight_plan(randint(0, 3), randint(2, 9))\n            action = ShipyardAction.launch_fleet_with_flight_plan(50, flight_plan)\n            shipyard.next_action = action\n        elif kore_left >= spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n            shipyard.next_action = action\n\n    return me.next_actions')


# In[ ]:


env.run(["/kaggle/working/pilot.py"])
env.render(mode="ipython", width=1000, height=800)


# 52ターン目にUIを一時停止すると、艦隊が進水するフライトプランを確認できます。
# 
# Notice that if you pause the UI on turn 50, you can see the flight plan your fleet will launch with.

# ## How flight plans work
# 
# 1. NESW方向の場合は、進水の方向を変更して一致させます
# 2. 数値の場合は、数値をデクリメントします
# 
# 1. if it's a direction NESW, change the fleet direction to match
# 2. if it's a number, decrement the number
# 
# examples:
# 
# #### N2S (go north, then continue for 2 squares, then go south)
# ```
# N2S
# 2S
# 1S
# S
# (fleet will continue south)
# ```
# 
# #### N10E (go north, then continue for 10 squares, then go east)
# ```
# N10E
# 10E
# 9E
# 8E
# 7E
# 6E
# 5E
# 4E
# 3E
# 2E
# 1E
# E
# (fleet will continue east)
# ```

# # Helper_function

# **霊夢:Helper functionの中身が知りたい**
# 
# **魔理沙:inspectを使えば良いんじゃね？**
# 
# **Reimu: I want to know the contents of the Helper function**
# 
# **Marisa: Should I use inspect?**

# In[ ]:


from kaggle_environments.envs.kore_fleets.helpers import *
import inspect

print(inspect.getsource(kaggle_environments.envs.kore_fleets.helpers))


# In[ ]:


get_ipython().run_cell_magic('writefile', 'kore_function.py', '\n# Copyright 2021 Kaggle Inc\n#\n# Licensed under the Apache License, Version 2.0 (the "License");\n# you may not use this file except in compliance with the License.\n# You may obtain a copy of the License at\n#\n#      http://www.apache.org/licenses/LICENSE-2.0\n#\n# Unless required by applicable law or agreed to in writing, software\n# distributed under the License is distributed on an "AS IS" BASIS,\n# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n# See the License for the specific language governing permissions and\n# limitations under the License.\n\nfrom copy import deepcopy\nfrom enum import Enum, auto\nfrom functools import wraps\nfrom kaggle_environments.helpers import Point, group_by, Direction\nfrom typing import *\nimport sys\nimport math\nimport random\nimport kaggle_environments.helpers\n\n\n# region Data Model Classes\nclass Observation(kaggle_environments.helpers.Observation):\n    """\n    Observation primarily used as a helper to construct the Board from the raw observation.\n    This provides bindings for the observation type described at https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kore/kore.json\n    """\n    @property\n    def kore(self) -> List[float]:\n        """Serialized list of available kore per cell on the board."""\n        return self["kore"]\n\n    @property\n    def players(self) -> List[List[int]]:\n        """List of players and their assets."""\n        return self["players"]\n\n    @property\n    def player(self) -> int:\n        """The current agent\'s player index."""\n        return self["player"]\n\n\nclass Configuration(kaggle_environments.helpers.Configuration):\n    """\n    Configuration provides access to tunable parameters in the environment.\n    This provides bindings for the configuration type described at https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kore/kore.json\n    """\n    @property\n    def agent_timeout(self) -> float:\n        """Maximum runtime (seconds) to initialize an agent."""\n        return self["agentTimeout"]\n\n    @property\n    def starting_kore(self) -> int:\n        """The starting amount of kore available on the board."""\n        return self["startingKore"]\n\n    @property\n    def size(self) -> int:\n        """The number of cells vertically and horizontally on the board."""\n        return self["size"]\n\n    @property\n    def spawn_cost(self) -> int:\n        """The amount of kore to spawn a new ship."""\n        return self["spawnCost"]\n\n    @property\n    def convert_cost(self) -> int:\n        """The amount of ships needed from a fleet to create a shipyard."""\n        return self["convertCost"]\n\n    @property\n    def regen_rate(self) -> float:\n        """The rate kore regenerates on the board."""\n        return self["regenRate"]\n\n    @property\n    def max_cell_kore(self) -> int:\n        """The maximum kore that can be in any cell."""\n        return self["maxRegenCellKore"]\n\n    @property\n    def random_seed(self) -> int:\n        """The seed to the random number generator (0 means no seed)."""\n        return self["randomSeed"]\n\n\nclass ShipyardActionType(Enum):\n    SPAWN = auto()\n    LAUNCH = auto()\n\n    def __str__(self) -> str:\n        return self.name\n\nclass ShipyardAction:\n\n    def __init__(self, type: ShipyardActionType, num_ships: Optional[int], flight_plan: Optional[str]) -> None:\n        self._type = type\n        assert num_ships >= 0, "must be a non-negative number"\n        assert num_ships == int(num_ships), "must be an integer"\n        self._num_ships = num_ships\n        self._flight_plan = flight_plan\n\n    def __str__(self) -> str:\n        if self._type == ShipyardActionType.SPAWN:\n            return f\'{self._type.name}_{self._num_ships}\'\n        if self._type == ShipyardActionType.LAUNCH:\n            return f\'{self._type.name}_{self._num_ships}_{self._flight_plan}\'\n    \n    @property\n    def name(self):\n        return str(self)\n\n    @staticmethod\n    def from_str(raw: str):\n        if not raw:\n            return None\n        if raw.startswith(ShipyardActionType.SPAWN.name):\n            return ShipyardAction.spawn_ships(int(raw.split("_")[1]))\n        if raw.startswith(ShipyardActionType.LAUNCH.name):\n            _, ship_str, plan_str = raw.split("_")\n            num_ships = int(ship_str)\n            return ShipyardAction.launch_fleet_with_flight_plan(num_ships, plan_str)\n\n    @staticmethod\n    def launch_fleet_in_direction(number_ships: int, direction: Direction):\n        flight_plan = None\n        if isinstance(direction, Direction):\n            flight_plan = direction.to_char()\n        else:\n            flight_plan = flight_plan.upper()\n        return ShipyardAction.launch_fleet_with_flight_plan(number_ships, flight_plan)\n        \n    @staticmethod\n    def launch_fleet_with_flight_plan(number_ships: int, flight_plan: str):\n        flight_plan = flight_plan.upper()\n        assert number_ships > 0, "must be a positive number_ships"\n        assert number_ships == int(number_ships), "must be an integer number_ships"\n        assert flight_plan is not None and len(flight_plan) > 0, "flight_plan must be a str of len > 0"\n        assert flight_plan[0].isalpha() and flight_plan[0] in "NESW", "flight_plan must start with a valid direciton NESW"\n        assert all([c in "NESWC0123456789" for c in flight_plan]), "flight_plan (" + flight_plan + ")can only contain NESWC0-9"\n        if len(flight_plan) > Fleet.max_flight_plan_len_for_ship_count(number_ships): \n            print("flight plan will be truncated: flight plan for " + str(number_ships) + " must be at most " + str(Fleet.max_flight_plan_len_for_ship_count(number_ships)))\n        return ShipyardAction(ShipyardActionType.LAUNCH, number_ships, flight_plan)\n\n    @staticmethod\n    def spawn_ships(number_ships: int):\n        assert number_ships == int(number_ships), "must be an integer number_ships"\n        return ShipyardAction(ShipyardActionType.SPAWN, number_ships, None)\n\n    @property\n    def action_type(self) -> ShipyardActionType:\n        return self._type\n    \n    @property\n    def num_ships(self) -> Optional[int]:\n        return self._num_ships\n\n    @property\n    def flight_plan(self) -> Optional[str]:\n        return self._flight_plan\n\n\nFleetId = NewType(\'FleetId\', str)\nShipyardId = NewType(\'ShipyardId\', str)\nPlayerId = NewType(\'PlayerId\', int)\n\n\nclass Cell:\n    def __init__(self, position: Point, kore: float, shipyard_id: Optional[ShipyardId], fleet_id: Optional[FleetId], board: \'Board\') -> None:\n        self._position = position\n        self._kore = kore\n        self._shipyard_id = shipyard_id\n        self._fleet_id = fleet_id\n        self._board = board\n\n    @property\n    def position(self) -> Point:\n        return self._position\n\n    @property\n    def kore(self) -> float:\n        return self._kore\n\n    @property\n    def shipyard_id(self) -> Optional[ShipyardId]:\n        return self._shipyard_id\n\n    @property\n    def fleet_id(self) -> Optional[FleetId]:\n        return self._fleet_id\n\n    @property\n    def fleet(self) -> Optional[\'Fleet\']:\n        """Returns the fleet on this cell if it exists and None otherwise."""\n        return self._board.fleets.get(self.fleet_id)\n\n    @property\n    def shipyard(self) -> Optional[\'Shipyard\']:\n        """Returns the shipyard on this cell if it exists and None otherwise."""\n        return self._board.shipyards.get(self.shipyard_id)\n\n    def neighbor(self, offset: Point) -> \'Cell\':\n        """Returns the cell at self.position + offset."""\n        (x, y) = self.position + offset\n        return self._board[x, y]\n\n    @property\n    def north(self) -> \'Cell\':\n        """Returns the cell north of this cell."""\n        return self.neighbor(Direction.NORTH.to_point())\n\n    @property\n    def south(self) -> \'Cell\':\n        """Returns the cell south of this cell."""\n        return self.neighbor(Direction.SOUTH.to_point())\n\n    @property\n    def east(self) -> \'Cell\':\n        """Returns the cell east of this cell."""\n        return self.neighbor(Direction.EAST.to_point())\n\n    @property\n    def west(self) -> \'Cell\':\n        """Returns the cell west of this cell."""\n        return self.neighbor(Direction.WEST.to_point())\n\n\nclass Fleet:\n    def __init__(self, fleet_id: FleetId, ship_count: int, direction: Direction, position: Point, kore: int, flight_plan: str, player_id: PlayerId, board: \'Board\') -> None:\n        self._id = fleet_id\n        self._ship_count = ship_count\n        self._direction = direction\n        self._position = position\n        self._flight_plan = flight_plan\n        self._kore = kore\n        self._player_id = player_id\n        self._board = board\n\n    @property\n    def id(self) -> FleetId:\n        return self._id\n\n    @property\n    def ship_count(self) -> int:\n        return self._ship_count\n\n    @property\n    def direction(self) -> Direction:\n        return self._direction\n\n    @property\n    def position(self) -> Point:\n        return self._position\n\n    @property\n    def kore(self) -> int:\n        return self._kore\n\n    @property\n    def player_id(self) -> PlayerId:\n        return self._player_id\n\n    @property\n    def cell(self) -> Cell:\n        """Returns the cell this fleet is on."""\n        return self._board[self.position]\n\n    @property\n    def player(self) -> \'Player\':\n        """Returns the player that owns this ship."""\n        return self._board.players[self.player_id]\n\n    @property\n    def flight_plan(self) -> str:\n        """Returns the current flight plan of the fleet"""\n        return self._flight_plan\n\n    @property\n    def collection_rate(self) -> float:\n        """ln(ship_count) / 10"""\n        return min(math.log(self.ship_count) / 20, .99)\n\n    @staticmethod\n    def max_flight_plan_len_for_ship_count(ship_count) -> int:\n        """Returns the length of the longest possible flight plan this fleet can be assigned"""\n        return math.floor(2 * math.log(ship_count)) + 1\n\n    @property\n    def _observation(self) -> List[int]:\n        """Converts a fleet back to the normalized observation subset that constructed it."""\n        return [self.position.to_index(self._board.configuration.size), self.kore, self.ship_count, self.direction.to_index(), self.flight_plan]\n\n    def less_than_other_allied_fleet(self, other):\n        if not self.ship_count == other.ship_count:\n            return self.ship_count < other.ship_count\n        if not self.kore == other.kore:\n            return self.kore < other.kore\n        return self.direction.to_index() > other.direction.to_index()\n\n\nupgrade_times = [pow(i,2) + 1 for i in range(1, 10)]\nSPAWN_VALUES = []\ncurrent = 0\nfor t in upgrade_times:\n    current += t\n    SPAWN_VALUES.append(current)\n\nclass Shipyard:\n    def __init__(self, shipyard_id: ShipyardId, ship_count: int, position: Point, player_id: PlayerId, turns_controlled: int, board: \'Board\', next_action: Optional[ShipyardAction] = None) -> None:\n        self._id = shipyard_id\n        self._ship_count = ship_count\n        self._position = position\n        self._player_id = player_id\n        self._turns_controlled = turns_controlled\n        self._board = board\n        self._next_action = next_action\n\n    @property\n    def id(self) -> ShipyardId:\n        return self._id\n\n    @property\n    def ship_count(self):\n        return self._ship_count\n\n    @property\n    def position(self) -> Point:\n        return self._position\n\n    @property\n    def player_id(self) -> PlayerId:\n        return self._player_id\n\n    @property\n    def max_spawn(self) -> int:\n        for idx, target in enumerate(SPAWN_VALUES):\n            if self._turns_controlled < target:\n                return idx + 1\n        return len(SPAWN_VALUES) + 1\n\n    @property\n    def cell(self) -> Cell:\n        """Returns the cell this shipyard is on."""\n        return self._board[self.position]\n\n    @property\n    def player(self) -> \'Player\':\n        return self._board.players[self.player_id]\n\n    @property\n    def next_action(self) -> ShipyardAction:\n        """Returns the action that will be executed by this shipyard when Board.next() is called (when the current turn ends)."""\n        return self._next_action\n\n    @next_action.setter\n    def next_action(self, value: Optional[ShipyardAction]) -> None:\n        """Sets the action that will be executed by this shipyard when Board.next() is called (when the current turn ends)."""\n        self._next_action = value\n\n    @property\n    def _observation(self) -> List[int]:\n        """Converts a shipyard back to the normalized observation subset that constructed it."""\n        return [self.position.to_index(self._board.configuration.size), self.ship_count, self._turns_controlled]\n\n\nclass Player:\n    def __init__(self, player_id: PlayerId, kore: int, shipyard_ids: List[ShipyardId], fleet_ids: List[FleetId], board: \'Board\') -> None:\n        self._id = player_id\n        self._kore = kore\n        self._shipyard_ids = shipyard_ids\n        self._fleet_ids = fleet_ids\n        self._board = board\n\n    @property\n    def id(self) -> PlayerId:\n        return self._id\n\n    @property\n    def kore(self) -> int:\n        return self._kore\n\n    @property\n    def shipyard_ids(self) -> List[ShipyardId]:\n        return self._shipyard_ids\n\n    @property\n    def fleet_ids(self) -> List[FleetId]:\n        return self._fleet_ids\n\n    @property\n    def shipyards(self) -> List[Shipyard]:\n        """Returns all shipyards owned by this player."""\n        return [\n            self._board.shipyards[shipyard_id]\n            for shipyard_id in self.shipyard_ids\n        ]\n\n    @property\n    def fleets(self) -> List[Fleet]:\n        """Returns all fleets owned by this player."""\n        return [\n            self._board.fleets[fleet_id]\n            for fleet_id in self.fleet_ids\n        ]\n\n    @property\n    def is_current_player(self) -> bool:\n        """Returns whether this player is the current player (generally if this returns True, this player is you)."""\n        return self.id == self._board.current_player_id\n\n    @property\n    def next_actions(self) -> Dict[str, str]:\n        """Returns all queued fleet and shipyard actions for this player formatted for the kore interpreter to receive as an agent response."""\n        shipyard_actions = {\n            shipyard.id: shipyard.next_action.name\n            for shipyard in self.shipyards\n            if shipyard.next_action is not None\n        }\n        return {**shipyard_actions}\n\n    @property\n    def _observation(self):\n        """Converts a player back to the normalized observation subset that constructed it."""\n        shipyards = {shipyard.id: shipyard._observation for shipyard in self.shipyards}\n        fleets = {fleet.id: fleet._observation for fleet in self.fleets}\n        return [self.kore, shipyards, fleets]\n# endregion\n\n\nclass Board:\n    def __init__(\n        self,\n        raw_observation: Dict[str, Any],\n        raw_configuration: Union[Configuration, Dict[str, Any]],\n        next_actions: Optional[List[Dict[str, str]]] = None\n    ) -> None:\n        """\n        Creates a board from the provided observation, configuration, and next_actions as specified by\n        https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kore/kore.json\n        Board tracks players (by id), fleets (by id), shipyards (by id), and cells (by position).\n        Each entity contains both key values (e.g. fleet.player_id) as well as entity references (e.g. fleet.player).\n        References are deep and chainable e.g.\n            [fleet.kore for player in board.players for fleet in player.fleets]\n            fleet.player.shipyards[0].cell.north.east.fleet\n        Consumers should not set or modify any attributes except and Shipyard.next_action\n        """\n        observation = Observation(raw_observation)\n        # next_actions is effectively a Dict[Union[[FleetId, FleetAction], [ShipyardId, ShipyardAction]]]\n        # but that type\'s not very expressible so we simplify it to Dict[str, str]\n        # Later we\'ll iterate through it once for each fleet and shipyard to pull all the actions out\n        next_actions = next_actions or ([{}] * len(observation.players))\n\n        self._step = observation.step\n        self._remaining_overage_time = observation.remaining_overage_time\n        self._configuration = Configuration(raw_configuration)\n        self._current_player_id = observation.player\n        self._players: Dict[PlayerId, Player] = {}\n        self._fleets: Dict[FleetId, Fleet] = {}\n        self._shipyards: Dict[ShipyardId, Shipyard] = {}\n        self._cells: Dict[Point, Cell] = {}\n\n        size = self.configuration.size\n        # Create a cell for every point in a size x size grid\n        for x in range(size):\n            for y in range(size):\n                position = Point(x, y)\n                kore = observation.kore[position.to_index(size)]\n                # We\'ll populate the cell\'s fleets and shipyards in _add_fleet and _add_shipyard\n                self.cells[position] = Cell(position, kore, None, None, self)\n\n        for (player_id, player_observation) in enumerate(observation.players):\n            # We know the len(player_observation) == 3 based on the schema -- this is a hack to have a tuple in json\n            [player_kore, player_shipyards, player_fleets] = player_observation\n            # We\'ll populate the player\'s fleets and shipyards in _add_fleet and _add_shipyard\n            self.players[player_id] = Player(player_id, player_kore, [], [], self)\n            player_actions = next_actions[player_id] or {}\n\n            for (fleet_id, [fleet_index, fleet_kore, ship_count, direction, flight_plan]) in player_fleets.items():\n                fleet_position = Point.from_index(fleet_index, size)\n                fleet_direction = Direction.from_index(direction)\n                self._add_fleet(Fleet(fleet_id, ship_count, fleet_direction, fleet_position, fleet_kore, flight_plan, player_id, self))\n\n            for (shipyard_id, [shipyard_index, ship_count, turns_controlled]) in player_shipyards.items():\n                shipyard_position = Point.from_index(shipyard_index, size)\n                raw_action = player_actions.get(shipyard_id)\n                action = ShipyardAction.from_str(raw_action)\n                self._add_shipyard(Shipyard(shipyard_id, ship_count, shipyard_position, player_id, turns_controlled, self, action))\n\n    @property\n    def configuration(self) -> Configuration:\n        return self._configuration\n\n    @property\n    def players(self) -> Dict[PlayerId, Player]:\n        return self._players\n\n    @property\n    def fleets(self) -> Dict[FleetId, Fleet]:\n        """Returns all fleets on the current board."""\n        return self._fleets\n\n    @property\n    def shipyards(self) -> Dict[ShipyardId, Shipyard]:\n        """Returns all shipyards on the current board."""\n        return self._shipyards\n\n    @property\n    def cells(self) -> Dict[Point, Cell]:\n        """Returns all cells on the current board."""\n        return self._cells\n\n    @property\n    def step(self) -> int:\n        return self._step\n\n    @property\n    def current_player_id(self) -> PlayerId:\n        return self._current_player_id\n\n    @property\n    def current_player(self) -> Player:\n        """Returns the current player (generally this is you)."""\n        return self._players[self.current_player_id]\n\n    @property\n    def opponents(self) -> List[Player]:\n        """\n        Returns all players that aren\'t the current player.\n        You can get all opponent fleets with [fleet for fleet in player.fleets for player in board.opponents]\n        """\n        return [player for player in self.players.values() if not player.is_current_player]\n\n    @property\n    def observation(self) -> Dict[str, Any]:\n        """Converts a Board back to the normalized observation that constructed it."""\n        size = self.configuration.size\n        kore = [self[Point.from_index(index, size)].kore for index in range(size * size)]\n        players = [player._observation for player in self.players.values()]\n\n        return {\n            "kore": kore,\n            "players": players,\n            "player": self.current_player_id,\n            "step": self.step,\n            "remainingOverageTime": self._remaining_overage_time,\n        }\n\n    def __deepcopy__(self, _) -> \'Board\':\n        actions = [player.next_actions for player in self.players.values()]\n        return Board(self.observation, self.configuration, actions)\n\n    def __getitem__(self, point: Union[Tuple[int, int], Point]) -> Cell:\n        """\n        This method will wrap the supplied position to fit within the board size and return the cell at that location.\n        e.g. on a 3x3 board, board[2, 1] is the same as board[5, 1]\n        """\n        if not isinstance(point, Point):\n            (x, y) = point\n            point = Point(x, y)\n        return self._cells[point % self.configuration.size]\n\n    def __str__(self) -> str:\n        """\n        The board is printed in a grid with the following rules:\n        Capital letters are shipyards\n        Lower case letters are fleets\n        Digits are cell kore and scale from 0-9 directly proportional to a value between 0 and self.configuration.max_cell_kore\n        Player 1 is letter a/A\n        Player 2 is letter b/B\n        etc.\n        """\n        size = self.configuration.size\n        result = \'\'\n        for y in range(size):\n            for x in range(size):\n                cell = self[(x, size - y - 1)]\n                result += \'|\'\n                result += (\n                    chr(ord(\'a\') + cell.ship.player_id)\n                    if cell.fleet is not None\n                    else \' \'\n                )\n                # This normalizes a value from 0 to max_cell kore to a value from 0 to 9\n                normalized_kore = int(9.0 * cell.kore / float(self.configuration.max_cell_kore))\n                result += str(normalized_kore)\n                result += (\n                    chr(ord(\'A\') + cell.shipyard.player_id)\n                    if cell.shipyard is not None\n                    else \' \'\n                )\n            result += \'|\\n\'\n        return result\n\n    def _add_fleet(self: \'Board\', fleet: Fleet) -> None:\n        fleet.player.fleet_ids.append(fleet.id)\n        fleet.cell._fleet_id = fleet.id\n        self._fleets[fleet.id] = fleet\n\n    def _add_shipyard(self: \'Board\', shipyard: Shipyard) -> None:\n        shipyard.player.shipyard_ids.append(shipyard.id)\n        shipyard.cell._shipyard_id = shipyard.id\n        shipyard.cell._kore = 0\n        self._shipyards[shipyard.id] = shipyard\n\n    def _delete_fleet(self: \'Board\', fleet: Fleet) -> None:\n        fleet.player.fleet_ids.remove(fleet.id)\n        if fleet.cell.fleet_id == fleet.id:\n            fleet.cell._fleet_id = None\n        del self._fleets[fleet.id]\n\n    def _delete_shipyard(self: \'Board\', shipyard: Shipyard) -> None:\n        shipyard.player.shipyard_ids.remove(shipyard.id)\n        if shipyard.cell.shipyard_id == shipyard.id:\n            shipyard.cell._shipyard_id = None\n        del self._shipyards[shipyard.id]\n\n    def get_fleet_at_point(self: \'Board\', position: Point) -> Optional[Fleet]:\n        matches = [fleet for fleet in self.fleets.values() if fleet.position == position]\n        if matches:\n            assert len(matches) == 1\n            return matches[0]\n        return None\n\n    def get_shipyard_at_point(self: \'Board\', position: Point) -> Optional[Shipyard]:\n        matches = [shipyard for shipyard in self.shipyards.values() if shipyard.position == position]\n        if matches:\n            assert len(matches) == 1\n            return matches[0]\n        return None\n\n    def get_cell_at_point(self: \'Board\', position: Point):\n        return self.cells.get(position)\n\n    def print(self: \'Board\') -> None:\n        size = self.configuration.size\n        player_chars = {\n            pid: alpha\n            for pid, alpha in  zip(self.players, "abcdef"[:len(self.players)])\n        }\n        print(self.configuration.size * "=")\n        for i in range(size):\n            row = ""\n            for j in range(size):\n                pos = Point(j, size - 1 - i)\n                curr_cell = self.cells[pos]\n                if curr_cell.shipyard is not None:\n                    row += player_chars[curr_cell.shipyard.player_id].upper()\n                elif curr_cell.fleet is not None:\n                    row += player_chars[curr_cell.fleet.player_id]\n                elif curr_cell.kore <= 50:\n                    row += " "\n                elif curr_cell.kore <= 250:\n                    row += "."\n                elif curr_cell.kore <= 400:\n                    row += "*"\n                elif curr_cell.kore > 400:\n                    row += "o"\n            print(row)\n        print(self.configuration.size * "=")\n\n    def print_kore(self: \'Board\') -> None:\n        size = self.configuration.size\n        print(self.configuration.size * "=")\n        for i in range(size):\n            row = ""\n            for j in range(size):\n                pos = Point(j, size - 1 - i)\n                curr_cell = self.cells[pos]\n                row += str(int(curr_cell.kore)) + ","\n            print(row)\n        print(self.configuration.size * "=")\n\n    def next(self) -> \'Board\':\n        """\n        Returns a new board with the current board\'s next actions applied.\n        The current board is unmodified.\n        This can form a kore interpreter, e.g.\n            next_observation = Board(current_observation, configuration, actions).next().observation\n        """\n        # Create a copy of the board to modify so we don\'t affect the current board\n        board = deepcopy(self)\n        configuration = board.configuration\n        convert_cost = configuration.convert_cost\n        spawn_cost = configuration.spawn_cost\n        uid_counter = 0\n\n        # This is a consistent way to generate unique strings to form fleet and shipyard ids\n        def create_uid():\n            nonlocal uid_counter\n            uid_counter += 1\n            return f"{self.step + 1}-{uid_counter}"\n\n        # this checks the validity of a flight plan\n        def is_valid_flight_plan(flight_plan):\n            return len([c for c in flight_plan if c not in "NESWC0123456789"]) == 0\n\n        # Process actions and store the results in the fleets and shipyards lists for collision checking\n        for player in board.players.values():\n            for shipyard in player.shipyards:\n                if shipyard.next_action == None:\n                    pass\n                elif shipyard.next_action.num_ships == 0:\n                    pass\n                elif (shipyard.next_action.action_type == ShipyardActionType.SPAWN \n                        and player.kore >= spawn_cost * shipyard.next_action.num_ships \n                        and shipyard.next_action.num_ships <= shipyard.max_spawn):\n                    # Handle SPAWN actions\n                    player._kore -= spawn_cost * shipyard.next_action.num_ships\n                    shipyard._ship_count += shipyard.next_action.num_ships\n                elif shipyard.next_action.action_type == ShipyardActionType.LAUNCH and shipyard.ship_count >= shipyard.next_action.num_ships:\n                    flight_plan = shipyard.next_action.flight_plan\n                    if not flight_plan or not is_valid_flight_plan(flight_plan):\n                        continue\n                    shipyard._ship_count -= shipyard.next_action.num_ships\n                    direction = Direction.from_char(flight_plan[0])\n                    max_flight_plan_len = Fleet.max_flight_plan_len_for_ship_count(shipyard.next_action.num_ships)\n                    if len(flight_plan) > max_flight_plan_len:\n                        flight_plan = flight_plan[:max_flight_plan_len]\n                    board._add_fleet(Fleet(FleetId(create_uid()), shipyard.next_action.num_ships, direction, shipyard.position, 0, flight_plan, player.id, board))\n                \n            # Clear the shipyard\'s action so it doesn\'t repeat the same action automatically\n            for shipyard in player.shipyards:\n                shipyard.next_action = None\n                shipyard._turns_controlled += 1\n\n            def find_first_non_digit(candidate_str):\n                for i in range(len(candidate_str)):\n                    if not candidate_str[i].isdigit():\n                        return i\n                else:\n                    return len(candidate_str) + 1\n                return 0\n\n            for fleet in player.fleets:\n                # remove any errant 0s\n                while fleet.flight_plan and fleet.flight_plan.startswith("0"):\n                    fleet._flight_plan = fleet.flight_plan[1:]\n                if fleet.flight_plan and fleet.flight_plan[0] == "C" and fleet.ship_count >= convert_cost and fleet.cell.shipyard_id is None:\n                    player._kore += fleet.kore\n                    fleet.cell._kore = 0\n                    board._add_shipyard(Shipyard(ShipyardId(create_uid()), fleet.ship_count - convert_cost, fleet.position, player.id, 0, board))\n                    board._delete_fleet(fleet)\n                    continue\n\n                while fleet.flight_plan and fleet.flight_plan[0] == "C":\n                    # couldn\'t build, remove the Convert and continue with flight plan\n                    fleet._flight_plan = fleet.flight_plan[1:]\n\n                if fleet.flight_plan and fleet.flight_plan[0].isalpha():\n                    fleet._direction = Direction.from_char(fleet.flight_plan[0])\n                    fleet._flight_plan = fleet.flight_plan[1:]\n                elif fleet.flight_plan:\n                    idx = find_first_non_digit(fleet.flight_plan)\n                    digits = int(fleet.flight_plan[:idx])\n                    rest = fleet.flight_plan[idx:]\n                    digits -= 1\n                    if digits > 0:\n                        fleet._flight_plan = str(digits) + rest\n                    else:\n                        fleet._flight_plan = rest\n\n                # continue moving in the fleet\'s direction\n                fleet.cell._fleet_id = None\n                fleet._position = fleet.position.translate(fleet.direction.to_point(), configuration.size)\n                # We don\'t set the new cell\'s fleet_id here as it would be overwritten by another fleet in the case of collision.\n\n            def combine_fleets(fid1: FleetId, fid2: FleetId) -> FleetId:\n                f1 = board.fleets[fid1]\n                f2 = board.fleets[fid2]\n                if f1.less_than_other_allied_fleet(f2):\n                    f1, f2 = f2, f1\n                    fid1, fid2 = fid2, fid1\n                f1._kore += f2.kore\n                f1._ship_count += f2._ship_count\n                board._delete_fleet(f2)\n                return fid1\n            \n            # resolve any allied fleets that ended up in the same square\n            fleets_by_loc = group_by(player.fleets, lambda fleet: fleet.position.to_index(configuration.size))\n            for value in fleets_by_loc.values():\n                value.sort(key=lambda fleet: (fleet.ship_count, fleet.kore, -fleet.direction.to_index()), reverse=True)\n                fid = value[0].id\n                for i in range (1, len(value)):\n                    fid = combine_fleets(fid, value[i].id)\n\n            # Lets just check and make sure.\n            assert player.kore >= 0\n\n        def resolve_collision(fleets: List[Fleet]) -> Tuple[Optional[Fleet], List[Fleet]]:\n            """\n            Accepts the list of fleets at a particular position (must not be empty).\n            Returns the fleet with the most ships or None in the case of a tie along with all other fleets.\n            """\n            if len(fleets) == 1:\n                return fleets[0], []\n            fleets_by_ships = group_by(fleets, lambda fleet: fleet.ship_count)\n            most_ships = max(fleets_by_ships.keys())\n            largest_fleets = fleets_by_ships[most_ships]\n            if len(largest_fleets) == 1:\n                # There was a winner, return it\n                winner = largest_fleets[0]\n                return winner, [fleet for fleet in fleets if fleet != winner]\n            # There was a tie for most ships, all are deleted\n            return None, fleets\n\n        # Check for fleet to fleet collisions\n        fleet_collision_groups = group_by(board.fleets.values(), lambda fleet: fleet.position)\n        for position, collided_fleets in fleet_collision_groups.items():\n            winner, deleted = resolve_collision(collided_fleets)\n            shipyard = group_by(board.shipyards.values(), lambda shipyard: shipyard.position).get(position)\n            if winner is not None:\n                winner.cell._fleet_id = winner.id\n                max_enemy_size = max([fleet.ship_count for fleet in deleted]) if deleted else 0\n                winner._ship_count -= max_enemy_size\n            for fleet in deleted:\n                board._delete_fleet(fleet)\n                if winner is not None:\n                    # Winner takes deleted fleets\' kore\n                    winner._kore += fleet.kore\n                elif winner is None and shipyard and shipyard[0].player:\n                    # Desposit the kore into the shipyard\n                    shipyard[0].player._kore += fleet.kore\n                elif winner is None:\n                    # Desposit the kore on the square\n                    board.cells[position]._kore += fleet.kore\n\n\n        # Check for fleet to shipyard collisions\n        for shipyard in list(board.shipyards.values()):\n            fleet = shipyard.cell.fleet\n            if fleet is not None and fleet.player_id != shipyard.player_id:\n                if fleet.ship_count > shipyard.ship_count:\n                    count = fleet.ship_count - shipyard.ship_count\n                    board._delete_shipyard(shipyard)\n                    board._add_shipyard(Shipyard(ShipyardId(create_uid()), count, shipyard.position, fleet.player.id, 1, board))\n                    fleet.player._kore += fleet.kore\n                    board._delete_fleet(fleet)\n                else:\n                    shipyard._ship_count -= fleet.ship_count\n                    shipyard.player._kore += fleet.kore\n                    board._delete_fleet(fleet)\n\n        # Deposit kore from fleets into shipyards\n        for shipyard in list(board.shipyards.values()):\n            fleet = shipyard.cell.fleet\n            if fleet is not None and fleet.player_id == shipyard.player_id:\n                shipyard.player._kore += fleet.kore\n                shipyard._ship_count += fleet.ship_count\n                board._delete_fleet(fleet)\n\n        # apply fleet to fleet damage on all orthagonally adjacent cells\n        incoming_fleet_dmg = DefaultDict(lambda: DefaultDict(int))\n        for fleet in board.fleets.values():\n            for direction in Direction.list_directions():\n                curr_pos = fleet.position.translate(direction.to_point(), board.configuration.size)\n                fleet_at_pos = board.get_fleet_at_point(curr_pos)\n                if fleet_at_pos and not fleet_at_pos.player_id == fleet.player_id:\n                    incoming_fleet_dmg[fleet_at_pos.id][fleet.id] = fleet.ship_count\n\n        # dump 1/2 kore to the cell of killed fleets\n        # mark the other 1/2 kore to go to surrounding fleets proportionally\n        to_distribute = DefaultDict(lambda: DefaultDict(int))\n        for fleet_id, fleet_dmg_dict in incoming_fleet_dmg.items():\n            fleet = board.fleets[fleet_id]\n            damage = sum(fleet_dmg_dict.values())\n            if damage >= fleet.ship_count:\n                fleet.cell._kore += fleet.kore / 2\n                to_split = fleet.kore / 2\n                for f_id, dmg in fleet_dmg_dict.items():\n                    to_distribute[f_id][fleet.position.to_index(board.configuration.size)] = to_split * dmg/damage\n                board._delete_fleet(fleet)\n            else:\n                fleet._ship_count -= damage\n\n        # give kore claimed above to surviving fleets, otherwise add it to the kore of the tile where the fleet died\n        for fleet_id, loc_kore_dict in to_distribute.items():\n            fleet = board.fleets.get(fleet_id)\n            if fleet:\n                fleet._kore += sum(loc_kore_dict.values())\n            else:\n                for loc_idx, kore in loc_kore_dict.items():\n                    board.cells.get(Point.from_index(loc_idx, board.configuration.size))._kore += kore\n\n        # Collect kore from cells into fleets\n        for fleet in board.fleets.values():\n            cell = fleet.cell\n            delta_kore = round(cell.kore * min(fleet.collection_rate, .99), 3)\n            if delta_kore > 0:\n                fleet._kore += delta_kore\n                cell._kore -= delta_kore\n\n        # Regenerate kore in cells\n        for cell in board.cells.values():\n            if cell.fleet_id is None and cell.shipyard_id is None:\n                if cell.kore < configuration.max_cell_kore:\n                    next_kore = round(cell.kore * (1 + configuration.regen_rate), 3)\n                    cell._kore = next_kore\n\n        board._step += 1\n\n        # self.print()\n\n        return board\n\n\ndef board_agent(agent: Callable[[Board], None]):\n    """\n    Decorator used to create an agent that modifies a board rather than an observation and a configuration\n    Automatically returns the modified board\'s next actions\n\n    @board_agent\n    def my_agent(board: Board) -> None:\n        ...\n    """\n    @wraps(agent)\n    def agent_wrapper(obs, config) -> Dict[str, str]:\n        board = Board(obs, config)\n        agent(board)\n        return board.current_player.next_actions\n\n    if agent.__module__ is not None and agent.__module__ in sys.modules:\n        setattr(sys.modules[agent.__module__], agent.__name__, agent_wrapper)\n    return agent_wrapper')


# **霊夢:Apache License, Version 2.0だったのでありがたく公開したよ。**
# 
# **魔理沙:中身を見ていこうか。**
# 
# Observation
# * kore (星の位置)
# * players (敵味方含めたプレイヤーの情報)
# * player (自分のプレイヤーID)
# 
# Configuration
# * agent_timeout (タイムアウトになる時間)  
# * starting_kore (開始時に所持しているKoreの量)  
# * size (フィールドのサイズ)  
# * spawn_cost (造船に必要なKoreの量)  
# * convert_cost (造船所を作る為に必要な船の数)  
# * regen_rate (艦隊が発掘できる資源の割合)  
# * max_cell_kore (星が持つ資源の最大量)  
# * random_seed (seed値)  
# 
# Board
# * configuration (ゲーム設定)
# * players (敵味方含めたプレイヤーの情報)
# * fleets (艦隊の情報)
# * shipyards (造船所の情報)
# * cells (それぞれのマスの情報)
# * step (現在のターン数）
# * current_player_id (自身のプレイヤーID)

# # Add test

# **Agentは毎ターン、自身が持つそれぞれの造船所が実行するアクションをリストとして出力しなければなりません。毎ターン、それぞれの造船所は以下のどちらかをすることができます。**
#   
#     1. 船を建造する。
#     2. 艦隊を進水させる。
#     
# **毎ターン、Agentはゲームのあらゆる側面に関する完全な情報、を読み込むことができます。具体的には以下のことを読み込むことができます。**    
# * 各プレイヤーのKoreの量
# * すべての艦隊の配置・船の数・Koreの運搬量
# * フライトプラン
# * フィールドにある資源の量
# 
# **Each turn, the Agent must output a list of actions to be performed by each of its shipyards. Each turn, each shipyard can either.**
#   
#     1. build a ship.
#     2. launch a fleet.
#     
# **Each turn, the Agent can read complete information about all aspects of the game. Specifically, it can read the following.**    
# * Amount of Kore for each player.
# * All fleet deployments, number of ships, and amount of Kore carried
# * Flight plans
# * Amount of resources on the field
# 

# **霊夢:フライトプランを指定する方法がなんだかピンとこないなあ**
# 
# **魔理沙:じゃあ実際にいくつかのルートを指定して動きをみてみよう**
# 
# **Reimu: Somehow it doesn't come out**
# 
# **Marisa: Let's actually specify some routes and see how they move**

# In[ ]:


get_ipython().run_cell_magic('writefile', 'pilot2.py', '   \nfrom kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n    for shipyard in me.shipyards:\n        if shipyard.ship_count >= 50:\n            flight_plan = "N2S"\n            action = ShipyardAction.launch_fleet_with_flight_plan(50, flight_plan)\n            shipyard.next_action = action\n        elif kore_left >= spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n            shipyard.next_action = action\n\n    return me.next_actions')


# In[ ]:


env.run(["/kaggle/working/pilot2.py"])
env.render(mode="ipython", width=1000, height=800)


# **魔理沙:フライトプランをN2Sにして動かしてみたぜ。**
# 
# **霊夢:北に3マス動いてから、南に移動しているね。**
# 
# **魔理沙:もうひとつ試してみよう。**
# 
# **Marisa: I changed the flight plan to N2S and tried it.**
# 
# **Reimu: You've moved 3 squares north and then south.**
# 
# **Marisa: Let's try another one.**

# In[ ]:


get_ipython().run_cell_magic('writefile', 'pilot3.py', '   \nfrom kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n    for shipyard in me.shipyards:\n        if shipyard.ship_count >= 50:\n            flight_plan = "N10E10S"\n            action = ShipyardAction.launch_fleet_with_flight_plan(50, flight_plan)\n            shipyard.next_action = action\n        elif kore_left >= spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n            shipyard.next_action = action\n\n    return me.next_actions')


# In[ ]:


env.run(["/kaggle/working/pilot3.py"])
env.render(mode="ipython", width=1000, height=800)


# **魔理沙:今度はフライトプランをN10E10Sにして実行してみたぜ。**
# 
# **霊夢:なるほど、今度は北に11マス、東に11マス、そして南に移動し続けたな。つまり、数字は前の文字が表す方角にその数だけ進むことを表しているってことか！**
# 
# **Marisa: This time I set the flight plan to N10E10S and executed it.**
# 
# **Reimu: I see, this time I kept moving 11 squares to the north, 11 squares to the east, and south. In other words, the number means that the number goes in the direction indicated by the previous letter!**

# # public agent

# **霊夢:なんとなくやる事はわかってきたけど、まだ全体像がつかめてないな。**
# 
# **魔理沙:実際にBovard氏が書いてくれたいくつかのAgentを読み込んで戦わせてみるぜ**
# 
# **Reimu: I'm kind of getting an idea of what we're going to do, but I don't have the whole picture yet.**
# 
# **Marisa:I'm actually going to load some Agents that Bovard wrote and let them fight**

# In[ ]:


random_agent = "../input/kore-python-random-agent/agent.py"
miner_agent = "../input/kore-miner-agent/miner.py"
attacker_agent = "../input/kore-attacker-agent/attacker.py"
balanced_agent = "../input/kore-balanced-agent/balanced.py"


# In[ ]:


get_ipython().system('cp ../input/kore-python-random-agent/agent.py ./')
get_ipython().system('cp ../input/kore-miner-agent/miner.py ./')
get_ipython().system('cp ../input/kore-attacker-agent/attacker.py ./')
get_ipython().system('cp ../input/kore-balanced-agent/balanced.py ./')


# ## battle1 random_agent vs random_agent

# In[ ]:


env.run([random_agent,random_agent])
env.render(mode="ipython", width=1000, height=800)


# ## battle2 miner_agent vs miner_agent

# In[ ]:


env.run([miner_agent,miner_agent])
env.render(mode="ipython", width=1000, height=800)


# ## battle3 attacker_agent vs attacker_agent

# In[ ]:


env.run([attacker_agent,attacker_agent])
env.render(mode="ipython", width=1000, height=800)


# ## battle4 balanced_agent vs balanced_agent

# In[ ]:


env.run([balanced_agent,balanced_agent])
env.render(mode="ipython", width=1000, height=800)


# ## battle5 4 agents

# In[ ]:


import math


# In[ ]:


env.run([random_agent,miner_agent,attacker_agent,balanced_agent])
env.render(mode="ipython", width=1000, height=800)


# **霊夢:Nameerrorが起きたけど原因はわからなかったよ**
# 
# **Reimu: Nameerror happened but I couldn't figure out why**

# **霊夢:なんとなくルールは分かってきたよ！**
# 
# **魔理沙:次回はIntro2を日本語訳していくぞ！**
# 
# **Reimu: Somehow I understand the rules!**
# 
# **Marisa: Next time, I'll translate Intro2 into Japanese!**