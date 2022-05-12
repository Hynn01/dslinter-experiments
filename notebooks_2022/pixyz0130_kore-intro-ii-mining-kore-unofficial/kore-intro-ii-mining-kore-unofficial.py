#!/usr/bin/env python
# coding: utf-8

# [![](https://img.youtube.com/vi/FnV0thLS1Fs/0.jpg)](https://www.youtube.com/watch?v=FnV0thLS1Fs)

# pixyz  
# last update 2022 05 03  
# ゆっくりしていってね！  

# version8  #Minning root - box_miner

# こっちも観てね！
# 
# Kore Intro I: The Basics 日本語訳 ゆっくり実況  
# https://www.kaggle.com/code/pixyz0130/kore-intro-i-the-basics

# # Contents
# 
# * [**Official guide**](#Official_guide)
# 
# * [**add test**](#add_test)
# 
# * [**Kore map**](#Kore_map)

# **霊夢:今回はコンペ理解のために、主催者のBovard氏が書いてくれたコードを日本語訳してみたよ。今回はIntro Ⅱをやっていくよ。**
# 
# **魔理沙:他に気になったことがあったら突っ込んでいくぞ。**
# 
# **Reimu: This time, I translated the code written by the organizer into Japanese to understand the competition.**
# 
# **Marisa: If you have any other concerns, I'll dig in.**

# # Official_guide

# **霊夢:まずは公式ガイドの日本語訳からやっていくよ**
# 
# **Reimu: Let's start with the Japanese translation of the official guide**
# 
# https://www.kaggle.com/code/bovard/kore-intro-ii-mining-kore/notebook

# ## Welcome back Commander!
# 
# In part 2, we'll look at how to mine Kore, and learn more about flight paths.
# 
# パート2では、Koreを採掘する方法を見て、飛行経路について詳しく学びます。

# In[ ]:


get_ipython().run_cell_magic('capture', '', '# install the latest version of kaggle_environments\n!pip install --upgrade kaggle_environments')


# In[ ]:


from kaggle_environments import make
env = make("kore_fleets", debug=True)
print(env.name, env.version)


# ## Mining
# 
# In Kore Fleets, a fleet picks up a certain % of the kore on the square it occupies at the end of its turn. This relationship is logrithmic, so many smaller fleets will pick up more kore than one large one. See the table below:
# 
# コレ艦隊では、艦隊はそのターンの終わりに居るマスで特定の割合のKoreを拾います。 この割合は対数的であるため、1つの大きな艦隊よりも複数の小さな艦隊の方が多くのKoreを獲得できます。 以下の表を参照してください。
# 
# | Number Ships | % mined  |
# | --- | --- | 
# | 1 | 0% |
# | 2 | 3% |
# | 3 | 5% |
# | 5 | 8% |
# | 8 | 10% |
# | 13 | 13% |
# | 21 | 15% |
# | 34 | 18% |
# | 55 | 20% |
# | 91 | 23% |
# | 149 | 25% |
# | 245 | 28% |
# | 404 | 30% |
# 
# The exact formula to get the % mined is `ln(num_ships_in_fleet) / 20`.
# 
# For example, `ln(55) / 20 = .20037`
# 
# Let's look at an example, 4 smaller fleets with 8 ships each vs one flet with 32 ships. For simplicites sake, let's ignore the 2% regrowth a turn.
# 
# この%を求める式は、 `ln（艦隊が持つ船の数）/20`です。
# 
# たとえば、 `ln（55）/ 20 = .20037`　となります。
# 
# 例を見てみましょう。それぞれ8隻の船を持つ艦隊が4隊であるパターンと、32隻の艦隊が1隊であるパターンです。 簡単にするために、1ターンの2％の再成長を無視しましょう。
# 
# ```
# Kore Mined, 4 fleets of 8 (10% mining rate)
# initial_kore = 100
# after first 8-ship fleet => 90
# after second 8-ship fleet => 81
# after third 8-ship fleet => 72.9
# after fourth 8-ship fleet => 65.6
# 
# Total Kore mined 100 - 65.6 = 34.4
# 
# Kore Mined, 1 fleet of 32 (17% mining rate)
# initial_kore = 100
# after first 32-ship fleet => 83
# 
# Total Kore mined 100 - 83 = 17
# ```
# 
# ```
# 8隻の船をもつ艦隊が4隊の場合 (10%)
# 
# （最初のKoreの数） = 100
# （1つ目の艦隊が発掘したあと） => 90
# （2つ目の艦隊が発掘したあと） => 81
# （3つ目の艦隊が発掘したあと） => 72.9
# （4つ目の艦隊が発掘したあと） => 65.6
# 
# トータルの発掘量 100 - 65.6 = 34.4
# 
# 32隻の船を持つ艦隊が1隊の場合 (17%)
# （最初のKoreの数） = 100
# （1つ目の艦隊が発掘したあと） => 83
# 
# トータルの発掘量 100 - 83 = 17
# ```
# 
# 
# So the smaller fleets mined 2x more!
# 
# つまり、より小さな艦隊は2倍以上採掘しました！

# ## The Downside of smaller fleets
# 
# The downside of smaller fleets is that they can have shorter flight plan instructions! The `length` of a flight plan is equal to the length of the string representing it.
# 
# For example, the flight plan "go north, continue 8 spaces, then to south", represented as `"N8S"` has a length of 3. A more complicated flight plan that goes in a loop, "N8E8S8W" (note you don't need a trailing 8), has length 7.
# 
# The below table shows the largest flight instructions that can be given to a fleet.
# 
# 小規模な艦隊の欠点は、飛行計画の指示が短くなることです。 飛行計画の「長さ」は、飛行計画を表す文字列の長さと同じです。
# 
# たとえば、「N8S」として表される「北に行き、8スペース進み、次に南に行く」飛行計画の長さは3です。ループするより複雑な飛行計画「N8E8S8W」（末尾に8は必要ありません）の場合、長さは7です。
# 
# 以下の表は、艦隊に与えることができる最大の飛行指示を示しています。
# 
# | Number Ships | Max Flight Plan Length  |
# | --- | --- | 
# | 1 | 1 |
# | 2 | 2 |
# | 3 | 3 |
# | 5 | 4 |
# | 8 | 5 |
# | 13 | 6 |
# | 21 | 7 |
# | 34 | 8 |
# | 55 | 9 |
# | 91 | 10 |
# | 149 | 11 |
# | 245 | 12 |
# | 404 | 13 |
# 
# Observant readers will notice these are an approximation of the [Fibbonaci numbers](https://en.wikipedia.org/wiki/Fibonacci_number), and and this is given by the formula `floor(2 * ln(num_ships)) + 1`[](http://)
# 
# 注意深い読者は、これらが[フィボナッチ数](https://en.wikipedia.org/wiki/Fibonacci_number)の近似値であり、これは式 `floor（2 * ln（num_ships））+1`で与えられることに気付くでしょう。

# **霊夢:船の数が少ない艦隊だと行動が制限されちゃうんだね。**
# 
# **魔理沙:ループするためには最低でも21隻必要なんだぜ。**
# 
# **Reimu: If you have a fleet with a small number of ships, your actions will be restricted.**
# 
# **Marisa: You need at least 21 ships to loop.**

# ## Putting them together
# 
# Large fleets are more manuverable, but don't mine quickly, while smaller fleets mine quickly, but are more limited. To take advantage of both, use the *fleet coalescence* mechanic!
# 
# 大きな艦隊はより機動性がありますが、効率的に採掘できません。一方、小さな艦隊は素早く採掘しますが、機動性は制限されます。 両方を利用するには、*艦隊合体*メカニックを使用してください！ｊ
# 
# > Any allied fleets that currently occupy the same space are added to the largest allied fleet in that location. Ship size, current kore, and finally direction (NESW) are used to determine the largest fleet.
# 
# > 現在同じマスにいるすべての同チームの艦隊は、その場所で最大の艦隊に変化します。 最大の艦隊を決定するために、船のサイズ、現在のコレ、そして最後に方向（NESW）が使用されます。
# 
# This means when two or more allied fleets end up occupying the same square, they will join the larger one (in terms of ships)! So your bigger fleets can "pick up" your smaller fleets after they are done mining.
# 
# これは、2つ以上の同盟艦隊が同じマスを占領することになった場合、それらはより大きな艦隊に加わることを意味します！ したがって、より大きな艦隊は、採掘が完了した後、より小さな艦隊を「拾う」ことができます。
# 
# Let's see if we can code a simple example of this!
# 
# この簡単な例をコーディングできるかどうか見てみましょう！

# 艦隊1は2隻で、1マス東に進んで、その後南に直進し続けます。  
# 艦隊2は3隻で、3マス東に進んで、その後南に直進し続けます。  
# 艦隊3は3隻で、艦隊1と2が戻ってくるのを見計らって、4マス東に進んで、その後西に直進して造船所に戻ります。

# In[ ]:


get_ipython().run_cell_magic('writefile', 'miner.py', '   \nfrom kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n\n    period = 4 + config.size + 1\n    \n    for shipyard in me.shipyards:\n        action = None\n        if turn % period == 4:\n            action = ShipyardAction.launch_fleet_with_flight_plan(2, "ES")\n        elif turn % period == 6: \n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E2S")\n        elif turn % period == 4 + config.size:\n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E3W")\n            shipyard.next_action = action\n        elif kore_left >= spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n        shipyard.next_action = action\n\n    return me.next_actions')


# In[ ]:


env.run(["/kaggle/working/miner.py"])
env.render(mode="ipython", width=1000, height=800)


# That works! But it's very slow. A better version might be a box miner, let's give that a shot!
# 
# それはうまくいきます！ しかし、それは非常に遅いです。 より良いバージョンはボックスマイナーかもしれません、それを試してみましょう！ｋ

# 艦隊1は21隻で、10マス東に進み、10マス北に進み、10マス西に進み、そして南に直進し造船所に戻ります。  
# 艦隊2から9は3隻で、2ターンおきに出発し、9~2マス東に進み、その後北に直進し続けます。  
# 艦隊10は2隻で、東に1マス進み、その後北に直進し続けます。  
# 艦隊11は2隻で、北に直進し続けます。

# In[ ]:


get_ipython().run_cell_magic('writefile', 'box_miner.py', '   \nfrom kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n\n    period = 40\n    \n    for shipyard in me.shipyards:\n        action = None\n        if turn < 40:\n            action = ShipyardAction.spawn_ships(1)\n        elif turn % period == 1:\n            action = ShipyardAction.launch_fleet_with_flight_plan(21, "E9N9W9S")\n        elif turn % period == 3: \n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E8N")\n        elif turn % period == 5: \n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E7N")\n        elif turn % period == 7: \n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E6N")\n        elif turn % period == 9: \n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E5N")\n        elif turn % period == 11: \n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E4N")\n        elif turn % period == 13: \n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E3N")\n        elif turn % period == 15: \n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E2N")\n        elif turn % period == 17: \n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E1N")\n        elif turn % period == 19: \n            action = ShipyardAction.launch_fleet_with_flight_plan(2, "EN")\n        elif turn % period == 21: \n            action = ShipyardAction.launch_fleet_with_flight_plan(2, "N")\n        elif kore_left >= spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n            \n        \n        shipyard.next_action = action\n\n    return me.next_actions')


# In[ ]:


env.run(["/kaggle/working/box_miner.py"])
env.render(mode="ipython", width=1000, height=800)


# **魔理沙:これを使うとすごく効率的に資源が回収できそうだな。**
# 
# **霊夢:次回はIntro Ⅲを訳していこう！**
# 
# **Marisa: It seems that resources can be recovered very efficiently using this.**
# 
# **Reimu: Let's translate Intro III next time!**

# # add test

# 霊夢:githubにkaggle_environmentsがあるから、その中の/envs/kore_fleetsを見よう
# 
# https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/kore_fleets

# In[ ]:


env = make("kore_fleets",configuration={"randomSeed":42}, debug=True)
env.run(["/kaggle/working/box_miner.py"])
env.render(mode="ipython", width=1000, height=800)


# **霊夢:makeの時にrandomSeedを指定してあげると、毎回同じマップを表示することができるね。**
# 
# **魔理沙:こういったことも、githubを読むとよくわかるから、大切だぞ**
# 
# **Reimu: If you specify randomSeed at the time of make, you can display the same map every time.**
# 
# **Marisa: It's important because you can understand these things by reading github**

# **霊夢:船の数が多い艦隊を進水させるよりも、船の数が少ない艦隊をたくさん進水させた方が、効率的に資源を回収できるんだね。**
# 
# **魔理沙:理論上はそうだけど実際はどうなんだろうか、実際に試してみよう！**
# 
# **Reimu: It's more efficient to launch a fleet with a small number of ships than to launch a fleet with a large number of ships.**
# 
# **Marisa: That's true in theory, but let's try it out!**

# **32隻の艦隊を、東西南北にランダムに直進させます。**
# 
# **Randomly move a fleet of 32 ships straight north, south, east and west.**

# In[ ]:


get_ipython().run_cell_magic('writefile', 'pilot1.py', '   \nfrom kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\n# a flight plan\ndef build_flight_plan(dir_idx):\n    flight_plan = Direction.from_index(dir_idx).to_char()\n    return flight_plan\n\ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n    period = config.size\n    flight_size = 32\n    \n    for shipyard in me.shipyards:\n        if shipyard.ship_count >= 32:\n            if turn % period == 0:\n                flight_plan = build_flight_plan(randint(0, 3))\n                action = ShipyardAction.launch_fleet_with_flight_plan(32, flight_plan)\n                shipyard.next_action = action\n            elif kore_left >= spawn_cost:\n                action = ShipyardAction.spawn_ships(1)\n                shipyard.next_action = action\n        elif kore_left >= spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n            shipyard.next_action = action\n\n    return me.next_actions')


# In[ ]:


env.run(["/kaggle/working/pilot1.py"])
env.render(mode="ipython", width=1000, height=800)


# **8隻の艦隊を、さっきの4倍の周期で、東西南北にランダムに直進させます。**
# 
# **Randomly move a fleet of 8 ships straight from north, south, east, and west in a cycle four times as long as before.**

# In[ ]:


get_ipython().run_cell_magic('writefile', 'pilot2.py', '   \nfrom kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\n# a flight plan\ndef build_flight_plan(dir_idx):\n    flight_plan = Direction.from_index(dir_idx).to_char()\n    return flight_plan\n\ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n    period = config.size\n    flight_size = 8\n    per = config.size//4\n    for shipyard in me.shipyards:\n        if shipyard.ship_count >= 8:\n            if turn % period == per or turn % period == per*2 or turn % period == per*3 or turn % period == 0:                \n                flight_plan = build_flight_plan(randint(0, 3))\n                action = ShipyardAction.launch_fleet_with_flight_plan(8, flight_plan)\n                shipyard.next_action = action\n            elif kore_left >= spawn_cost:\n                action = ShipyardAction.spawn_ships(1)\n                shipyard.next_action = action                \n        elif kore_left >= spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n            shipyard.next_action = action\n\n    return me.next_actions')


# In[ ]:


env.run(["/kaggle/working/pilot2.py"])
env.render(mode="ipython", width=1000, height=800)


# **魔理沙:十字にランダムに艦隊を進水するコードを動かしてみたぜ。**
# 
# **霊夢:ちゃんと8隻の艦隊の方が32隻の艦隊より多く資源を集められているぞ。**
# 
# 
# **Marisa: I tried moving the code to launch the fleet randomly in a cross.**
# 
# **Reimu: Well, 8 fleets are collecting more resources than 32 fleets.**
# 

# **霊夢:observationをつかうと、Koreの情報を取得できるみたいだね。**
# 
# **魔理沙:そうだね、それを使って、box_miner.pyを改造してみよう**
# 
# **Reimu: It seems that you can get information about Kore by using observation.**
# 
# **Marisa: Yeah, let's use it to modify box_miner.py**

# # Kore map

# **get_koreでkoreの情報を取得します。**
# 
# **count_koreでkoreの数が多いルートを3本厳選します。**
# 
# **Get kore information with get_kore.**
# 
# **Carefully select 3 routes with a large number of kore in count_kore.**

# In[ ]:


get_ipython().run_cell_magic('writefile', 'box_miner2.py', '   \nfrom kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\nimport numpy as np\nfrom scipy.stats import rankdata\n\ndef get_kore(obs, config):\n    GRID_SIZE = config.size\n    kore_grid_ = np.flip(np.array(obs["kore"], dtype=np.float32).reshape(GRID_SIZE, GRID_SIZE), axis=0)\n    return kore_grid_\n\ndef count_kore(kore_grid,config):\n    SIZE = config.size\n    count = np.zeros(SIZE//2)\n    for i in range(SIZE//2):\n        for j in range(SIZE//2):\n            count[i] += kore_grid[j+11][i+10]\n    count = np.argsort(count)[::-1]\n    res = np.zeros(SIZE//2)\n    for i in range(3):\n        res[count[i]] = 1\n    return res\n    \ndef agent(obs, config):\n    board = Board(obs, config)\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n    kore_grid = get_kore(obs, config)\n    period = 40\n    order = count_kore(kore_grid,config)\n    \n    for shipyard in me.shipyards:\n        max_spawn = shipyard.max_spawn\n        action = None\n        if turn < 40:\n            action = ShipyardAction.spawn_ships(min([int(kore_left//spawn_cost),max_spawn]))\n        elif turn % period == 1:\n            action = ShipyardAction.launch_fleet_with_flight_plan(21, "E9N9W9S")\n        elif turn % period == 3 and order[9]: \n            action = ShipyardAction.launch_fleet_with_flight_plan(9, "E8N")\n        elif turn % period == 5 and order[8]: \n            action = ShipyardAction.launch_fleet_with_flight_plan(9, "E7N")\n        elif turn % period == 7 and order[7]: \n            action = ShipyardAction.launch_fleet_with_flight_plan(9, "E6N")\n        elif turn % period == 9 and order[6]: \n            action = ShipyardAction.launch_fleet_with_flight_plan(9, "E5N")\n        elif turn % period == 11 and order[5]: \n            action = ShipyardAction.launch_fleet_with_flight_plan(9, "E4N")\n        elif turn % period == 13 and order[4]: \n            action = ShipyardAction.launch_fleet_with_flight_plan(9, "E3N")\n        elif turn % period == 15 and order[3]: \n            action = ShipyardAction.launch_fleet_with_flight_plan(9, "E2N")\n        elif turn % period == 17 and order[2]: \n            action = ShipyardAction.launch_fleet_with_flight_plan(9, "E1N")\n        elif turn % period == 19 and order[1]: \n            action = ShipyardAction.launch_fleet_with_flight_plan(9, "EN")\n        elif turn % period == 21 and order[0]: \n            action = ShipyardAction.launch_fleet_with_flight_plan(9, "N")\n        elif kore_left >= spawn_cost:\n            action = ShipyardAction.spawn_ships(min([int(kore_left//spawn_cost),max_spawn]))\n#             action = ShipyardAction.spawn_ships(1)            \n        \n        shipyard.next_action = action\n\n    return me.next_actions')


# In[ ]:


env.run(["/kaggle/working/box_miner2.py"])
env.render(mode="ipython", width=1000, height=800)


# **霊夢:ちゃんと多いルートを選んで通ってるように見えるね。**
# 
# **魔理沙:次回はintroⅢの日本語訳をやっていくぞ**
# 
# **Reimu: It looks like you've chosen a lot of routes.**
# 
# **Marisa: Next time, I'll do a Japanese translation of intro III**

# # Minning root - box_miner

# **霊夢:序盤にどれだけ資源を集められるかが重要そうだよね。**
# 
# **魔理沙:そうだね。序盤は資源が少ないから、その中でどれだけ効率よく資源を回収できるかが肝になってきそうだね。**
# 
# **霊夢:色々な戦略を比較して、どのやり方が序盤に資源を集めやすいかを調べてみよう**

# **霊夢: まずは、box_minnerを改良して、逆方向に出撃できるようにしてみよう。**

# In[ ]:


get_ipython().run_cell_magic('writefile', 'minning.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef box_minning(agent,turn,box_size):\n\n    period = 40\n    for shipyard in agent.shipyards:\n        if shipyard.ship_count >= 40 and turn % period == 1:\n            box_size = 1\n        if shipyard.ship_count >= 89 and turn % period == 1:\n            box_size = 2\n        action = None\n        \n        \n        if turn > 40:\n            if turn % period == 1:\n                action = ShipyardAction.launch_fleet_with_flight_plan(21, "E9N9W9S")\n            elif turn % period == 3: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E8N")\n            elif turn % period == 5: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E7N")\n            elif turn % period == 7: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E6N")\n            elif turn % period == 9: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E5N")\n            elif turn % period == 11: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E4N")\n            elif turn % period == 13: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E3N")\n            elif turn % period == 15: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E2N")\n            elif turn % period == 17: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E1N")\n            elif turn % period == 19: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2, "EN")\n            elif turn % period == 21: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2, "N")\n        if box_size > 1:\n            if turn % period == 2:\n                action = ShipyardAction.launch_fleet_with_flight_plan(21, "W9S9E9N")\n            elif turn % period == 4: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W8S")\n            elif turn % period == 6: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W7S")\n            elif turn % period == 8: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W6S")\n            elif turn % period == 10: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W5S")\n            elif turn % period == 12: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W4S")\n            elif turn % period == 14: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W3S")\n            elif turn % period == 16: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W2S")\n            elif turn % period == 18: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W1S")\n            elif turn % period == 20: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2, "WS")\n            elif turn % period == 22: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2, "S")\n            \n        shipyard.next_action = action\n        \n    return int(box_size)')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'spawn.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef spawn(agent,spawn_cost,kore_left):\n    for shipyard in agent.shipyards:\n        if shipyard.next_action:\n            continue\n        if kore_left >= spawn_cost * shipyard.max_spawn:\n            action = ShipyardAction.spawn_ships(shipyard.max_spawn)\n            shipyard.next_action = action\n            kore_left -= spawn_cost * shipyard.max_spawn\n        elif kore_left >= spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n            shipyard.next_action = action\n            kore_left -= spawn_cost\n\n    return')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'main.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\nfrom minning import box_minning\nfrom spawn import spawn\n\nbox_num = 0\n\ndef agent(obs, config):\n    global box_num\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n    \n    box_num = box_minning(me,turn,box_num)\n    spawn(me,spawn_cost,kore_left)\n    return me.next_actions')


# In[ ]:


env.run(["/kaggle/working/main.py"])
env.render(mode="ipython", width=1000, height=800)


# **霊夢:50隻を作るには最速で20ターンかかるね**
# 
# **霊夢:box_minerは一周で40ターン使うから、100ターン目までには1周しかできないね。**
# 
# **魔理沙:最速で21隻作るのは11ターンで済むから、12ターン目からは出撃できるんじゃないか？**
# 
# **霊夢:そうかもしれない、やってみよう!**

# In[ ]:


get_ipython().run_cell_magic('writefile', 'minning2.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef box_minning(agent,turn,box_size):\n\n    period = 40\n    for shipyard in agent.shipyards:\n        if shipyard.ship_count >= 40 and turn % period == 10:\n            box_size = 1\n        if shipyard.ship_count >= 89 and turn % period == 10:\n            box_size = 2\n        action = None\n        \n        \n        if turn > 9:\n            if turn % period == 10:\n                action = ShipyardAction.launch_fleet_with_flight_plan(21, "E9N9W9S")\n            elif turn % period == 12: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E8N")\n            elif turn % period == 14: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E7N")\n            elif turn % period == 16: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E6N")\n            elif turn % period == 18: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E5N")\n            elif turn % period == 20: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E4N")\n            elif turn % period == 22: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E3N")\n            elif turn % period == 24: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E2N")\n            elif turn % period == 26: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E1N")\n            elif turn % period == 28: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2, "EN")\n            elif turn % period == 30: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2, "N")\n        if box_size > 1:\n            if turn % period == 11:\n                action = ShipyardAction.launch_fleet_with_flight_plan(21, "W9S9E9N")\n            elif turn % period == 13: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W8S")\n            elif turn % period == 15: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W7S")\n            elif turn % period == 17: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W6S")\n            elif turn % period == 19: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W5S")\n            elif turn % period == 21: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W4S")\n            elif turn % period == 23: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W3S")\n            elif turn % period == 25: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W2S")\n            elif turn % period == 27: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W1S")\n            elif turn % period == 29: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2, "WS")\n            elif turn % period == 31: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2, "S")\n            \n        shipyard.next_action = action\n        \n    return int(box_size)')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'main2.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\nfrom minning2 import box_minning\nfrom spawn import spawn\n\nbox_num = 0\n\ndef agent(obs, config):\n    global box_num\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n    \n    box_num = box_minning(me,turn,box_num)\n    spawn(me,spawn_cost,kore_left)\n    return me.next_actions')


# In[ ]:


env.run(["/kaggle/working/main2.py"])
env.render(mode="ipython", width=1000, height=800)

