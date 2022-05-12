#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook contains
# - Plotting the match statistics
# - Visualize gameplay (see the last cell)
# 
# Refer to my [discussion post](https://www.kaggle.com/competitions/kore-2022/discussion/320987) for plans and suggestions.

# In[ ]:


get_ipython().run_cell_magic('capture', '', "\n%reset -sf\n!pip install --user kaggle-environments > /dev/null\n!rm *.py *.pickle\n\nfrom IPython.core.magic import register_cell_magic\n\n@register_cell_magic\ndef writefile_and_run(line, cell):\n    argz = line.split()\n    file = argz[-1]\n    mode = 'w'\n    if len(argz) == 2 and argz[0] == '-a':\n        mode = 'a'\n    with open(file, mode) as f:\n        f.write(cell)\n    get_ipython().run_cell(cell)")


# In[ ]:


get_ipython().run_cell_magic('writefile_and_run', 'kore_analysis.py', '\nimport os, re, json, enum, glob, shutil, collections, requests, pickle\n\nimport numpy as np\nimport matplotlib\nimport matplotlib.animation\nimport matplotlib.patheffects\nimport matplotlib.pyplot as plt\nimport IPython.display\n\nimport kaggle_environments\n\n\nplt.rcParams["interactive"] = False\nplt.rcParams["animation.html"] = "jshtml"\nplt.rcParams["animation.embed_limit"] = 70.0   # default 20.0 stopped at step 200\nplt.rcParams["figure.figsize"] = [8,8]\nplt.rcParams["figure.dpi"] = 100\nplt.rcParams["savefig.facecolor"] = "white"\n\n\ndef load_from_simulated_game(home_agent_path, away_agent_path):\n    env = kaggle_environments.make("kore_fleets", debug=True)\n    env.run([home_agent_path, away_agent_path])\n    return env\n\ndef load_from_replay_json(json_path):\n    with open(path_to_json, \'r\') as f:\n        match = json.load(f)\n    env = kaggle_environments.make("kore_fleets", steps=match[\'steps\'],\n                                   configuration=match[\'configuration\'])\n    home_agent = "home"\n    away_agent = "away"\n    return env\n\ndef load_from_episode_id(episode_id):\n    # kaggle.com/code/robga/kore-episode-scraper-match-downloader/\n    base_url = "https://www.kaggle.com/api/i/competitions.EpisodeService/"\n    get_url = base_url + "GetEpisodeReplay"\n    req = requests.post(get_url, json = {"episodeId": int(episode_id)}).json()\n    match = json.loads(req["replay"])\n    env = kaggle_environments.make("kore_fleets", steps=match[\'steps\'],\n                                   configuration=match[\'configuration\'])\n    return env')


# # Run match or load

# In[ ]:


# various ways to load your agent

# simulate a match between two agents
home_agent_path = "../input/kore-beta-1st-place-solution/main.py"
away_agent_path = "../input/kore-starter-6th-in-beta-rule-based-agent/main.py"
# env = load_from_simulated_game(home_agent_path, away_agent_path)

# from match replay file saved
file_pattern = "../input/kore-episode-scraper-match-downloader/*.json"
path_to_json = sorted(fn for fn in glob.glob(file_pattern) if "_" not in fn)[-2]
env = load_from_replay_json(path_to_json)

# from episode_id
episode_id = 36615058
env = load_from_episode_id(episode_id)


# In[ ]:


# save env object for animation later
with open('env.pickle', 'wb') as handle:
    pickle.dump(env, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


# object extracted
type(env.steps), len(env.steps)


# In[ ]:


env.render(mode="ipython", width=720, height=680)


# # Information Extraction
# 
# Here we compile the information of the action, shipyards, fleets, and kore storage fo each player.

# In[ ]:


get_ipython().run_cell_magic('writefile_and_run', '-a kore_analysis.py', '\nclass FlightPlanClass(enum.IntEnum):\n    invalid = 0\n    unknown = 1\n    acyclic = 2\n    boomerang = 3\n    rectangle = 4\n\ndef kore_mining_rate(kore_amount, fleetsize):\n    kore_amount_before_regeneration = kore_amount / 1.02\n    if kore_amount_before_regeneration < 500:\n        kore_amount = kore_amount_before_regeneration\n    precentage_mining_rate = np.log(max(1,fleetsize)) / 20\n    kore_amount = kore_amount / (1-precentage_mining_rate)\n    return kore_amount * precentage_mining_rate\n\ndef calculate_mining_rates(kore_amount_matrices, agent_fleetsize_matrices):\n    return [sum(kore_mining_rate(kore_amount, fleetsize) \n                for kore_amount, fleetsize in zip(kore_amounts, fleetsizes))\n            for kore_amounts, fleetsizes in zip(kore_amount_matrices, agent_fleetsize_matrices)]\n\ndef split_into_number_and_char(srr):\n    # https://stackoverflow.com/q/430079/5894029\n    arr = []\n    for word in re.split(\'(\\d+)\', srr):\n        try:\n            num = int(word)\n            arr.append(num)\n        except ValueError:\n            for c in word:\n                arr.append(c)\n    return arr\n\ndef extract_flight_plan(x,y,dir_idx,plan,endpoints=set()):\n    dir_to_dxdy = [(0,1), (1,0), (0,-1), (-1,0)]  # NESW\n    dcode_to_dxdy = {"N":(0,1), "E":(1,0), "S":(0,-1), "W":(-1,0)}\n    dx,dy = dir_to_dxdy[dir_idx]\n    \n    plan = collections.deque(split_into_number_and_char(plan))\n    \n    cx,cy = x, y\n    path = [(cx,cy)]\n    construct = []\n    first_move_complete = False\n    \n    while plan:\n        if first_move_complete and (cx, cy) in endpoints:\n            return path, construct, (cx, cy) == (x,y)\n        first_move_complete = True\n        word = plan.popleft()\n        if type(word) == int:\n            cx += dx\n            cy += dy\n            path.append((cx,cy))\n            word -= 1\n            if word > 0:\n                plan.appendleft(word)\n            continue\n        if word == "C":\n            construct.append((cx,cy))\n            continue\n        dx,dy = dcode_to_dxdy[word]\n        cx += dx\n        cy += dy\n        path.append((cx,cy))\n        \n    is_cyclic = False\n    for _ in range(30):\n        if cx == x and cy == y:\n            is_cyclic = True\n        cx += dx\n        cy += dy\n\n    return path, construct, is_cyclic\n\ndef fleetplan_matching(flight_plan):\n    # plan class, target_x, target_y, polarity, construct\n    \n    # plan class - boomerang, rectangle, acyclic\n    # polarity - whether first move vertical or horizontal\n    # whether construct is genuine will not be analyzed here\n    \n    if not re.match("^[NSEW][NSEWC0-9]*$", flight_plan):\n        return (FlightPlanClass.invalid, 0, 0, False, [])\n\n    polarity = (flight_plan[0] == "N") or (flight_plan[0] == "S")\n\n    path, construct, is_cyclic = extract_flight_plan(0,0,0,flight_plan)\n    \n    x_max_extent = 0\n    y_max_extent = 0\n    target_x, target_y = 0, 0\n    for x,y in path:\n        if abs(x) > x_max_extent or abs(y) > y_max_extent:\n            x_max_extent = abs(x)\n            y_max_extent = abs(y)\n            target_x, target_y = x, y\n            \n    # orbit\n    if re.match("^[NSEW]$", flight_plan):\n        return (FlightPlanClass.acyclic, target_x, target_y, polarity, construct)\n\n    # sneek peek, yo-yo\n    for d1,d2 in zip("NSEW", "SNWE"):\n        if re.match(f"^[{d1}][0-9]*[{d2}][0-9]*$", flight_plan):\n            return (FlightPlanClass.boomerang, target_x, target_y, polarity, construct)\n    \n    # travelling\n    for d1,d2 in zip("NSEW", "SNWE"):\n        if re.match(f"[NSEW][0-9]*[NSEW][0-9]*$", flight_plan):\n            return (FlightPlanClass.acyclic, target_x, target_y, polarity, construct)\n\n    # flat rectangle, rectangle\n    if is_cyclic:\n        for d1,d2 in zip("NSEW", "SNWE"):\n            if re.match(f"^[{d1}][0-9]*[NSEW][0-9]*[{d2}][0-9]*[NSEW][0-9]*$", flight_plan):  \n                return (FlightPlanClass.rectangle, target_x, target_y, polarity, construct)\n\n    # crowbar, boomerang\n    if is_cyclic:\n        for d1,d2 in zip("NSEW", "SNWE"):\n            if re.match(f"^[NSEW][0-9]*[{d1}][0-9]*[{d2}][0-9]*[NSEW][0-9]*$", flight_plan):  \n                return (FlightPlanClass.boomerang, target_x, target_y, polarity, construct)\n    \n    return (FlightPlanClass.unknown, target_x, target_y, polarity, construct)\n\nclass KoreMatch():\n    def __init__(self, match_info, home_agent="home", away_agent="away", save_animation=False):\n        self.match_info = match_info\n        self.home_agent = home_agent\n        self.away_agent = away_agent\n        self.save_animation = save_animation\n\n        res = match_info\n        self.home_actions = [home_info["action"] for home_info,_ in res]\n        self.away_actions = [away_info["action"] for _,away_info in res]\n        \n        self.home_kore_stored = [info[0]["observation"]["players"][0][0] for info in res]\n        self.away_kore_stored = [info[0]["observation"]["players"][1][0] for info in res]\n\n        self.home_shipyards = [info[0]["observation"]["players"][0][1] for info in res]\n        self.away_shipyards = [info[0]["observation"]["players"][1][1] for info in res]        \n        self.all_shipyards_locations = [\n            set(kaggle_environments.helpers.Point.from_index(int(loc_idx), 21) for loc_idx, _, _ in home_shipyards.values()) |\n            set(kaggle_environments.helpers.Point.from_index(int(loc_idx), 21) for loc_idx, _, _ in away_shipyards.values())\n            for home_shipyards, away_shipyards in zip(self.home_shipyards, self.away_shipyards)\n         ]\n        self.home_fleets = [info[0]["observation"]["players"][0][2] for info in res]\n        self.away_fleets = [info[0]["observation"]["players"][1][2] for info in res]\n        \n        self.home_kore_carried = [sum(x[1] for x in fleet_info.values()) for fleet_info in self.home_fleets]\n        self.away_kore_carried = [sum(x[1] for x in fleet_info.values()) for fleet_info in self.away_fleets]\n        \n        self.home_ship_standby = [sum(shipyard[1] for shipyard in shipyards.values()) for shipyards in self.home_shipyards]\n        self.away_ship_standby = [sum(shipyard[1] for shipyard in shipyards.values()) for shipyards in self.away_shipyards]\n        self.home_ship_launched = [sum(fleet[2] for fleet in fleets.values()) for fleets in self.home_fleets]\n        self.away_ship_launched = [sum(fleet[2] for fleet in fleets.values()) for fleets in self.away_fleets]\n        \n        self.home_fleetsize_matrices = [[0 for _ in info[0]["observation"]["kore"]] for info in res]\n        self.away_fleetsize_matrices = [[0 for _ in info[0]["observation"]["kore"]] for info in res]\n\n        for turn, (home_fleets_info, away_fleets_info) in enumerate(zip(self.home_fleets, self.away_fleets)):\n            for home_fleet_info in home_fleets_info.values():\n                location, fleetsize = home_fleet_info[0], home_fleet_info[2]\n                self.home_fleetsize_matrices[turn][location] += fleetsize\n            for away_fleet_info in away_fleets_info.values():\n                location, fleetsize = away_fleet_info[0], away_fleet_info[2]\n                self.away_fleetsize_matrices[turn][location] += fleetsize\n                \n        self.kore_amount_matrices = [info[0]["observation"]["kore"] for info in res]\n\n        self.home_mining_rates = calculate_mining_rates(self.kore_amount_matrices, self.home_fleetsize_matrices)\n        self.away_mining_rates = calculate_mining_rates(self.kore_amount_matrices, self.away_fleetsize_matrices)\n\n        self.home_spawing_costs = [-10 * sum(int(action.split("_")[1]) for action in actions.values() \n                                        if action.startswith("SPAWN")) for actions in self.home_actions]\n        self.away_spawing_costs = [-10 * sum(int(action.split("_")[1]) for action in actions.values() \n                                        if action.startswith("SPAWN")) for actions in self.away_actions]\n\n        self.home_launch_counts = [[int(action.split("_")[1]) for action in actions.values() if action.startswith("LAUNCH")]\n                                   for actions in self.home_actions]\n        self.away_launch_counts = [[int(action.split("_")[1]) for action in actions.values() if action.startswith("LAUNCH")] \n                                   for actions in self.away_actions]\n        self.home_launch_plans = [[(action.split("_")[2]) for action in actions.values() if action.startswith("LAUNCH")]\n                                  for actions in self.home_actions]\n        self.away_launch_plans = [[(action.split("_")[2]) for action in actions.values() if action.startswith("LAUNCH")] \n                                  for actions in self.away_actions]\n        \n        self.home_combat_diffs = [(a2+b2-a1-b1)-x-y for x,y,a1,b1,a2,b2 in \n                             zip(self.home_mining_rates[1:], self.home_spawing_costs[1:], self.home_kore_carried, self.home_kore_stored, \n                                 self.home_kore_carried[1:], self.home_kore_stored[1:])]\n        self.away_combat_diffs = [(a2+b2-a1-b1)-x-y for x,y,a1,b1,a2,b2 in \n                             zip(self.away_mining_rates[1:], self.away_spawing_costs[1:], self.away_kore_carried, self.away_kore_stored, \n                                 self.away_kore_carried[1:], self.away_kore_stored[1:])]\n        \n        self.home_kore_asset_sums = 500*np.array(list(map(len,self.home_shipyards))) \\\n                                   + 10*np.array(self.home_ship_standby) + 10*np.array(self.home_ship_launched) \\\n                                      + np.array(self.home_kore_stored) + np.array(self.home_kore_carried)\n        self.away_kore_asset_sums = 500*np.array(list(map(len,self.away_shipyards))) \\\n                                   + 10*np.array(self.away_ship_standby) + 10*np.array(self.away_ship_launched) \\\n                                      + np.array(self.away_kore_stored) + np.array(self.away_kore_carried)\n        \n    def plot_statistics_kore(self):\n        plt.figure(figsize=(15,5))\n        plt.plot(self.home_kore_stored, label=self.home_agent + " (stored)", color="blue", linestyle="dotted")\n        plt.plot(self.away_kore_stored, label=self.away_agent + " (stored)", color="red", linestyle="dotted")\n        plt.plot(self.home_kore_carried, label=self.home_agent + " (carried)", color="blue")\n        plt.plot(self.away_kore_carried, label=self.away_agent + " (carried)", color="red")\n        plt.title("Kore carried and stored over time")\n        plt.xlim(-20,400+20)\n        plt.legend()\n        plt.show()\n        \n    def plot_statistics_shipyards(self):\n        plt.figure(figsize=(15,4))\n        plt.stairs(list(map(len,self.home_shipyards)), label=self.home_agent, lw=1.5, baseline=None, color="blue")\n        plt.stairs(list(map(len,self.away_shipyards)), label=self.away_agent, lw=1.5, baseline=None, color="red")\n        plt.title("Number of shipyards over time")\n        plt.xlim(-20,400+20)\n        plt.legend()\n        plt.show()\n        \n    def plot_statistics_ships(self):        \n        plt.figure(figsize=(15,5))\n        plt.stairs(self.home_ship_standby, label=self.home_agent + " (standby)", baseline=None, color="blue")\n        plt.stairs(self.away_ship_standby, label=self.away_agent + " (standby)", baseline=None, color="red")\n        plt.stairs(self.home_ship_launched, label=self.home_agent + " (launched)", baseline=None, color="blue", linestyle="dotted")\n        plt.stairs(self.away_ship_launched, label=self.away_agent + " (launched)", baseline=None, color="red", linestyle="dotted")\n        plt.title("Ships standby and launched over time")\n        plt.xlim(-20,400+20)\n        plt.legend()\n        plt.show()\n \n    def plot_statistics_kore_rates(self):        \n        plt.figure(figsize=(15,5))\n        plt.plot(self.home_mining_rates, label=self.home_agent + " (mining)", color="blue")\n        plt.plot(self.away_mining_rates, label=self.away_agent + " (mining)", color="red")\n        plt.stairs(self.home_spawing_costs, label=self.home_agent + " (spawning)", baseline=None, color="blue")\n        plt.stairs(self.away_spawing_costs, label=self.away_agent + " (spawning)", baseline=None, color="red")\n        plt.title("Kore change rates over time")\n        plt.xlim(-20,400+20)\n        plt.legend()\n        plt.show()\n\n    def plot_statistics_combat_diffs(self):\n        plt.figure(figsize=(15,5))\n        plt.stairs(self.home_combat_diffs, label=self.home_agent + "(combat)", baseline=None, color="blue")\n        plt.stairs(self.away_combat_diffs, label=self.away_agent + "(combat)", baseline=None, color="red")\n        plt.title("Kore combat diffs over time")\n        plt.xlim(-20,400+20)\n        plt.legend()\n        plt.show()\n        \n    def plot_statistics_asset_sums(self):        \n        plt.figure(figsize=(15,3))\n        plt.stairs(self.home_kore_asset_sums, label=self.home_agent, baseline=None, color="blue")\n        plt.stairs(self.away_kore_asset_sums, label=self.away_agent, baseline=None, color="red")\n        plt.title("Value of assets in terms of Kore over time")\n        plt.xlim(-20,400+20)\n        plt.legend()\n        plt.show()\n        \n    def plot_statistics_launch_sizes(self):\n        display_limit = 100\n        limits = [1,2,3,5,8,13,21,34,55]\n        \n        plt.figure(figsize=(15,6))\n        for limit in limits:\n            plt.axhline(limit, color="gainsboro", zorder=0)\n            plt.axhline(-limit, color="gainsboro", zorder=0)\n        \n        home_xpts, home_ypts = [], []\n        home_xpts_extra, home_ypts_extra = [], []\n        for turn_idx, (launch_counts, launch_plans) in enumerate(zip(self.home_launch_counts, self.home_launch_plans)):\n            for launch_count, launch_plan in zip(launch_counts, launch_plans):\n                _, _, is_cyclic = extract_flight_plan(0,0,0,launch_plan)\n                if "C" in launch_plan:\n                    home_xpts_extra.append(turn_idx)\n                    home_ypts_extra.append(launch_count)\n                    continue\n                home_xpts.append(turn_idx)\n                home_ypts.append(launch_count)\n        plt.scatter(home_xpts, home_ypts, color="blue", s=4, label=self.home_agent)\n        plt.scatter(home_xpts_extra, home_ypts_extra, color="red", s=7)\n\n        away_xpts, away_ypts = [], []\n        away_xpts_extra, away_ypts_extra = [], []\n        home_xpts_build, home_ypts_build = [], []\n        for turn_idx, (launch_counts, launch_plans) in enumerate(zip(self.home_launch_counts, self.home_launch_plans)):\n            for launch_count, launch_plan in zip(launch_counts, launch_plans):\n                _, _, is_cyclic = extract_flight_plan(0,0,0,launch_plan)\n                if "C" in launch_plan:\n                    away_xpts_extra.append(turn_idx)\n                    away_ypts_extra.append(-launch_count)\n                    continue\n                away_xpts.append(turn_idx)\n                away_ypts.append(-launch_count)\n        plt.scatter(away_xpts, away_ypts, color="red", s=4, label=self.away_agent)        \n        plt.scatter(away_xpts_extra, away_ypts_extra, color="blue", s=7)\n\n        plt.title("Launch sizes over time")\n        plt.xlim(-20,400+20)\n        plt.yscale(\'symlog\', linthresh=9)\n        plt.yticks([-x for x in limits[2:]] + limits[2:], [-x for x in limits[2:]] + limits[2:])\n        plt.legend()\n        plt.show()\n\n    def plot_statistics_launch_plan_shapes(self):\n        plt.figure(figsize=(15,3))\n        xpts = []\n        ypts = []\n        \n        for turn_idx, launch_plans in enumerate(kore_match.home_launch_plans):\n            for launch_plan in launch_plans:\n                plan_class, target_x, target_y, polarity, construct = fleetplan_matching(launch_plan)\n                xpts.append(turn_idx)\n                ypts.append(int(plan_class) + np.random.randn()/10)\n\n        plt.scatter(xpts, ypts, color="blue", s=4, label=self.home_agent)        \n\n        xpts = []\n        ypts = []\n        \n        for turn_idx, launch_plans in enumerate(kore_match.away_launch_plans):\n            for launch_plan in launch_plans:\n                plan_class, target_x, target_y, polarity, construct = fleetplan_matching(launch_plan)\n                xpts.append(turn_idx)\n                ypts.append(int(plan_class) + np.random.randn()/10)\n\n        plt.scatter(xpts, ypts, color="red", s=4, label=self.away_agent)        \n\n        plt.title("Distribution of flight plans classes over time")\n        plt.xlim(-20,400+20)\n        plt.yticks([e.value for e in FlightPlanClass], [e.name for e in FlightPlanClass])        \n        plt.legend()\n        plt.show()')


# In[ ]:


kore_match = KoreMatch(env.steps)


# In[ ]:


kore_match.home_actions[178]  # name-player: instruction_shipcount(_flightplan)


# In[ ]:


kore_match.home_kore_stored[178]


# In[ ]:


kore_match.away_shipyards[178]  # name-player: location, ship count, turn existence


# In[ ]:


kore_match.home_fleets[178]  # name-player: location, kore carried, ship count, direction?, remaining flight plan


# # Statisical Visualizations

# In[ ]:


kore_match.plot_statistics_kore()


# In[ ]:


kore_match.plot_statistics_shipyards()


# In[ ]:


kore_match.plot_statistics_ships()


# In[ ]:


kore_match.plot_statistics_kore_rates()


# In[ ]:


kore_match.plot_statistics_combat_diffs()


# In[ ]:


kore_match.plot_statistics_asset_sums()


# In[ ]:


kore_match.plot_statistics_launch_sizes()


# In[ ]:


kore_match.plot_statistics_launch_plan_shapes()


# # Animation Generation

# In[ ]:


get_ipython().run_cell_magic('writefile_and_run', '-a kore_analysis.py', '\ndef draw_fleet(x,y,dir_idx,ships_size,kore_amount,color):\n    mx,my = x+0.5, y+0.5\n    \n    icon_size = 0.4\n    tip = (0, icon_size)\n    left_wing = (icon_size/1.5, -icon_size)\n    right_wing = (-icon_size/1.5, -icon_size)\n    \n    polygon = plt.Polygon([tip, left_wing, right_wing], color=color, alpha=0.3)\n    transform = matplotlib.transforms.Affine2D().rotate_deg(270*dir_idx).translate(mx,my)\n    polygon.set_transform(transform + plt.gca().transData)\n    plt.gca().add_patch(\n        polygon\n    )\n    \n    text = plt.text(x+0.1, y+0.75, ships_size, color="purple",\n                    horizontalalignment=\'left\', verticalalignment=\'center\')\n    text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=2, foreground=\'w\', alpha=0.8)])\n    \n    kore_amount = int(kore_amount)\n    if kore_amount > 0:\n        text = plt.text(x+0.1, y+0.25, kore_amount, color="grey",\n                        horizontalalignment=\'left\', verticalalignment=\'center\')\n        text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=3, foreground=\'w\', alpha=0.8)])\n\ndef draw_flight_plan(x,y,dir_idx,plan,fleetsize,color):\n\n    path, construct, is_cyclic = extract_flight_plan(x,y,dir_idx,plan)\n\n    px = np.array([x for x,y in path]) + 0.5\n    py = np.array([y for x,y in path]) + 0.5\n    for ox in [-21,0,21]:\n        for oy in [-21,0,21]:\n            plt.plot(px+ox, py+oy, color=color, lw=np.log(fleetsize)**2/1.5, alpha=0.3, solid_capstyle=\'round\')\n    for x,y in construct:\n        plt.scatter(x+0.5, y+0.5, s=100, marker="x", color=color)\n\ndef existence_to_production_capacity(existence):\n    if existence >= 294: return 10\n    if existence >= 212: return 9\n    if existence >= 147: return 8\n    if existence >= 97: return 7\n    if existence >= 60: return 6\n    if existence >= 34: return 5\n    if existence >= 17: return 4\n    if existence >= 7: return 3\n    if existence >= 2: return 2\n    return 1\n            \ndef draw_shipyard(x,y,ships_size,existence,color):\n    plt.text(x+0.5,y+0.5,"⊕", fontsize=23, color=color,\n             horizontalalignment=\'center\', verticalalignment=\'center\', alpha=0.5)\n    if ships_size > 0:\n        text = plt.text(x+0.1, y+0.75, ships_size, color="purple",\n                        horizontalalignment=\'left\', verticalalignment=\'center\')\n        text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=2, foreground=\'w\', alpha=0.8)])\n    text = plt.text(x+0.9, y+0.25, existence_to_production_capacity(existence), color="black",\n                    horizontalalignment=\'right\', verticalalignment=\'center\')\n    text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=2, foreground=\'w\', alpha=0.8)])\n\ndef draw_kore_amounts(kore_amounts, excluded_xys={}):\n    for loc_idx,kore_amount in enumerate(kore_amounts):\n        x,y = kaggle_environments.helpers.Point.from_index(loc_idx, 21)\n        color = "gainsboro"\n        if kore_amount >= 20: color = "silver"\n        if kore_amount >= 100: color = "gray"\n        if kore_amount >= 500: color = "black"\n        if (x,y) not in excluded_xys and kore_amount > 0:\n            text = plt.text(x, y, int(kore_amount), color=color, fontsize=7,\n                            horizontalalignment=\'center\', verticalalignment=\'center\')\n            text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=3, foreground=\'w\', alpha=0.8)])\n\ndef draw_statistics(turn_num, home_stored_kore, away_stored_kore):\n    plt.text(0-0.5, 21-0.5, f"Kore: {home_stored_kore:.0f}" , color="blue",\n             horizontalalignment=\'left\', verticalalignment=\'bottom\')\n    plt.text(21-0.5, 21-0.5, f"Kore: {away_stored_kore:.0f}" , color="red",\n             horizontalalignment=\'right\', verticalalignment=\'bottom\')\n    plt.text(21/2-0.5, 21-0.5, f"Turn: {turn_num:.0f}" , color="black",\n             horizontalalignment=\'center\', verticalalignment=\'bottom\')    ')


# In[ ]:


get_ipython().run_cell_magic('writefile_and_run', '-a kore_analysis.py', '\ndef render_turn(self, turn):\n    res = self.match_info\n    # -0.5 for all x,y for all functions called here as a stopgap to shift axes\n    turn_info = res[turn][0]["observation"]\n    \n    plt.gca().cla()\n    plt.gcf().clf()\n    plt.gcf().subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=None, hspace=None)\n    excluded_xys = set()\n\n    color="blue"\n    player_idx=0\n\n    for shipyard_info in turn_info["players"][player_idx][1].values():\n        loc_idx, ships_count, existence = shipyard_info\n        x,y = kaggle_environments.helpers.Point.from_index(loc_idx, 21)\n        draw_shipyard(x-0.5,y-0.5, ships_count, existence, color)\n        excluded_xys.add((x,y))\n\n    for fleet_info in turn_info["players"][player_idx][2].values():\n        loc_idx, kore_amount, ships_size, dir_idx, flight_plan = fleet_info\n        x,y = kaggle_environments.helpers.Point.from_index(loc_idx, 21)\n        draw_fleet(x-0.5,y-0.5,dir_idx,ships_size,kore_amount, color)\n        draw_flight_plan(x-0.5,y-0.5,dir_idx,flight_plan,ships_size, color)\n        excluded_xys.add((x,y))\n\n    color="red"\n    player_idx=1\n\n    for shipyard_info in turn_info["players"][player_idx][1].values():\n        loc_idx, ships_count, existence = shipyard_info\n        x,y = kaggle_environments.helpers.Point.from_index(loc_idx, 21)\n        draw_shipyard(x-0.5,y-0.5, ships_count, existence, color)\n        excluded_xys.add((x,y))\n\n    for fleet_info in turn_info["players"][player_idx][2].values():\n        loc_idx, kore_amount, ships_size, dir_idx, flight_plan = fleet_info\n        x,y = kaggle_environments.helpers.Point.from_index(loc_idx, 21)\n        draw_fleet(x-0.5,y-0.5,dir_idx,ships_size,kore_amount, color)\n        draw_flight_plan(x-0.5,y-0.5,dir_idx,flight_plan,ships_size, color)\n        excluded_xys.add((x,y))\n\n    draw_kore_amounts(turn_info["kore"], excluded_xys=excluded_xys)\n    draw_statistics(turn, self.home_kore_stored[turn], self.away_kore_stored[turn])\n\n    plt.gca().set_xlim(0-0.5,21-0.5)\n    plt.gca().set_ylim(0-0.5,21-0.5)\n    plt.gca().set_aspect(\'equal\')\n    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))\n    plt.gca().yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))\n    \n    if self.save_animation:\n        plt.savefig(f"frames/{turn:03}.png")\n\n# https://stackoverflow.com/a/24865663/5894029\nKoreMatch.render_turn = lambda self, turn: render_turn(self, turn)')


# In[ ]:


# kore_match.render_turn(200)


# In[ ]:


get_ipython().run_cell_magic('writefile_and_run', '-a kore_analysis.py', '\ndef animate(self):\n    if self.save_animation:\n        os.system("mkdir -p frames")\n\n    self.anim = matplotlib.animation.FuncAnimation(plt.gcf(), self.render_turn, frames=len(self.match_info))\n    self.html_animation = IPython.display.HTML(self.anim.to_jshtml())\n    plt.close()\n    \n    if self.save_animation:\n        os.system("convert -resize 75% -loop 0 frames/*.png gameplay.gif")\n        os.system("rm -rf frames/*.png")\n\nKoreMatch.animate = lambda self: animate(self)')


# # Function Export

# In[ ]:


get_ipython().run_line_magic('reset', '-sf')

# you can generate the animation anywhere given the env object
import pickle
with open('env.pickle', 'rb') as handle:
    env = pickle.load(handle)


# In[ ]:


# just import the the KoreMatch class
# kore_analysis might be a different directory, recommend to copy to local directory
from kore_analysis import KoreMatch

kore_match = KoreMatch(env.steps, save_animation=True)
kore_match.animate()


# # Strategy Visualizations

# In[ ]:


kore_match.html_animation

