#!/usr/bin/env python
# coding: utf-8

# [![](https://img.youtube.com/vi/FnV0thLS1Fs/0.jpg)](https://www.youtube.com/watch?v=FnV0thLS1Fs)

# pixyz  
# last update 2022 04 23  
# ゆっくりしていってね！  

# # Overview

# **霊夢：今日はゲームAIを作成するコンペだね。**
# 
# **魔理沙：まずは概要を読んでみよう。**

# In this turn-based simulation game you control a small armada of spaceships. As you mine the rare mineral “kore” from the depths of space, you teleport it back to your homeworld. But it turns out you aren’t the only civilization with this goal. In each game two players will 
# compete to collect the most kore from the board. Whoever has the largest kore cache by the end of 400 turns—or eliminates all of their opponents from the board before that—will be the winner!
# 
# Your algorithms determine the movements of your fleets to collect kore, but it's up to you to figure out how to make effective and efficient moves. You control your ships, build new ships, create shipyards, eliminate opponents, and mine the kore on the game board.
# 
# May your fleet live long and prosper!
# 
# このターンベースのシミュレーションゲームでは、宇宙船の小さな艦隊を操作します。宇宙の奥深くから希少な鉱物「コレ」を採掘するとき、それを故郷にテレポートします。しかし、この目標を掲げている文明はあなただけではないことがわかりました。各ゲームでは、2人のプレーヤーがボードから最も多くのコレーを集めるために競います。400ターンの終わりまでに最大のコレキャッシュを持っている人、またはそれ以前にボードからすべての対戦相手を排除した人が勝者になります！
# 
# あなたのアルゴリズムはコレを集めるためにあなたの艦隊の動きを決定します、しかし効果的で効率的な動きをする方法を理解するのはあなた次第です。あなたは自分の船を制御し、新しい船を建造し、造船所を作り、敵を排除し、ゲームボードでコレを採掘します。
# 
# あなたの艦隊が長生きして繁栄しますように！
# 
# ![](https://i.imgur.com/BtSsuHD.gif)

# # Rule

# **霊夢：自分は、船を操作する、船を造船する、造船所を作ることができるんだね。**
# 
# **魔理沙:勝利条件は、相手より資源を集める、もしくは、敵をすべて排除する、ことだね**
# 
# **霊夢:でもこれだけじゃ全体像がわからないなあ。**
# 
# **魔理沙:そうだね、じゃあ最初はゲーム画面を表示して実際の対戦を見てみよう！**
# 
# **Reimu: You can operate a ship, build a ship, build a shipyard.**
# 
# **Marisa: The victory condition is to collect resources from the opponent or eliminate all enemies**
# 
# **Reimu: But I don't know the whole picture with this alone.**
# 
# **Marisa: Well, let's first display the game screen and see the actual match!**

# **魔理沙:主催者のBovard氏がnotebookを出してくれていたので、参考にしたぜ。**
# 
# **Marisa: The organizer, Mr. Bovard, put out a notebook, so I used it as a reference.**
# 
# https://www.kaggle.com/code/bovard/kore-intro-i-the-basics

# **kaggle_enviromentsを使って、環境を作成します。**
# 
# **Use kaggle_enviroments to create your environment.**

# # Init

# In[ ]:


get_ipython().run_cell_magic('capture', '', '# install the latest version of kaggle_environments\n!pip install --upgrade kaggle_environments')


# In[ ]:


from kaggle_environments import make
env = make("kore_fleets", debug=True)
print(env.name, env.version)


# **ゲームを実行させるにはAIとなるpythonコードが必要です。**
# 
# **You need python code that will be AI to run the game.**

# In[ ]:


get_ipython().run_cell_magic('writefile', 'do_nothing.py', '# First we will make a do_nothing player to observe the game board\ndef do_nothing():\n    pass')


# **pythonコードdo_nothing.pyを実行させてゲーム画面を表示させます。**
# 
# **Run the python code do_nothing.py to display the game screen.**

# # Game screen

# In[ ]:


env.run(["/kaggle/working/do_nothing.py"])
env.render(mode="ipython", width=1000, height=800)


# **ゲーム画面は、21×21のフィールドがあり、右には資源量や艦数、造船所の数などの情報が書かれています。**
# 
# **ゲーム開始と同時にプレイヤーには500コアが支給されます。コアは、造船するときなどに使います。**
# 
# **The game screen has a 21x21 field, and information such as the amount of resources, the number of ships, and the number of shipyards are written on the right.**
# 
# **Players will be provided with 500 cores as soon as the game starts. The core is used when building a ship.**

# **魔理沙:AIはpythonファイルで動かしてるけど、他の拡張子のファイルでは動かないのかな？**
# 
# **霊夢:両プレイヤーが全滅しなかった場合は、Koreが多い方が勝ちになるのかな？**
# 
# **魔理沙:資源が色々なところに散らばっているね、資源の位置情報は、開始と同時に取得できるのかな？**
# 
# **霊夢:ターンが経過すると、星が大きくなって、得られるKoreが多くなるみたいだね。小さい資源を何回も取りに行くか、大きくなるまで待って取るか、どっちが良いんだろう？**
# 
# **Marisa: AI works with python files, but doesn't it work with files with other extensions?**
# 
# **Reimu: If both players aren't wiped out, will the one with more Kore win?**
# 
# **Marisa: Resources are scattered all over the place, can you get the location information of resources at the same time as the start?**
# 
# **Reimu: As the turn goes by, the stars get bigger and you get more Kore. Which is better, go to get small resources many times or wait until they grow up?**

# <img src="http://3.bp.blogspot.com/-KmQQLtEkmLw/U1T3r7D0NdI/AAAAAAAAfVI/c2d4n2kG00U/s400/figure_question.png" width="100">

# **魔理沙:次に、造船をするコードを動かしてみよう！**
# 
# **Marisa: Next, let's move the shipbuilding code!**

# # Make ship

# In[ ]:


get_ipython().run_cell_magic('writefile', 'builder.py', '# this one builds ships!\n\nfrom kaggle_environments.envs.kore_fleets.helpers import *\n\ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n    \n    # loop through all shipyards you control\n    for shipyard in me.shipyards:\n        # build a ship!\n        if kore_left >= spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n            shipyard.next_action = action\n\n    return me.next_actions')


# **ShipyardAction.spawn_shipsで造船をすることができます。造船するにはKoreが10個消費します。**
# 
# **You can build a ship at ShipyardAction.spawn_ships. Kore consumes 10 to build a ship.**

# In[ ]:


env.run(["/kaggle/working/builder.py"])
env.render(mode="ipython", width=1000, height=800)


# **造船するとShipsが増え、造船所の右上の数字が増えます。**
# 
# **Shipbuilding will increase Ships and the number in the upper right corner of the shipyard.**

# **魔理沙:造船所が持てる船の数には限界があるのかな？**
# 
# **霊夢:Koreが足りない状態で船を作ろうとしたらどうなるのかな？**
# 
# **Marisa: Is there a limit to the number of ships a shipyard can have?**
# 
# **Reimu: What if you try to build a ship with a shortage of Kore?**

# <img src="http://3.bp.blogspot.com/-KmQQLtEkmLw/U1T3r7D0NdI/AAAAAAAAfVI/c2d4n2kG00U/s400/figure_question.png" width="100">

# # Sortie

# In[ ]:


get_ipython().run_cell_magic('writefile', 'launcher.py', '   \nfrom kaggle_environments.envs.kore_fleets.helpers import *\n\ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n    for shipyard in me.shipyards:\n        if kore_left >= spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n            shipyard.next_action = action\n        elif shipyard.ship_count > 0:\n            direction = Direction.NORTH\n            action = ShipyardAction.launch_fleet_with_flight_plan(2, direction.to_char())\n            shipyard.next_action = action\n\n    return me.next_actions')


# **ShipyardAction.launch_fleet_with_flight_planで艦数とルートを決めるとその方向に指定した数の船が出撃します**
# 
# **If you decide the number of ships and the route in ShipyardAction.launch_fleet_with_flight_plan, the specified number of ships will sortie in that direction**

# In[ ]:


env.run(["/kaggle/working/launcher.py"])
env.render(mode="ipython", width=1000, height=800)


# **艦隊が生成されると、左上の艦隊のマークの数字がカウントされます。**
# 
# **フィールドは北と南、東と西、でつながっており、一番上にいる艦隊が上方向に進むと、同じ列の一番下にワープします。左右でも同様です。**
# 
# **When a fleet is generated, the number of the fleet mark on the upper left is counted.**
# 
# **The fields are connected north and south, east and west, and as the top fleet moves upwards, it warps to the bottom of the same row. The same is true for left and right.**

# **魔理沙:一つ艦隊の船の数は最大で何隻までだろう？**
# 
# **霊夢:造船所に戻ってきた船はどうなるのかな？**
# 
# **Marisa: What is the maximum number of ships in one fleet?**
# 
# **Reimu: What happens to the ship returning to the shipyard?**

# <img src="http://3.bp.blogspot.com/-KmQQLtEkmLw/U1T3r7D0NdI/AAAAAAAAfVI/c2d4n2kG00U/s400/figure_question.png" width="100">

# **Marisa; Let's control the ship next!**
# 
# **魔理沙；次は船を制御してみよう！**

# # Build flight plan

# In[ ]:


from random import randint
randint(2, 9)


# In[ ]:


get_ipython().run_cell_magic('writefile', 'pilot.py', '   \nfrom kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\n# a flight plan\ndef build_flight_plan(dir_idx, size):\n    flight_plan = ""\n    for i in range(4):\n        flight_plan += Direction.from_index((dir_idx + i) % 4).to_char()\n        if not i == 3:\n            flight_plan += str(size)\n    return flight_plan\n\ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n\n    for shipyard in me.shipyards:\n        if shipyard.ship_count >= 50:\n            flight_plan = build_flight_plan(randint(0, 3), randint(2, 9))\n            action = ShipyardAction.launch_fleet_with_flight_plan(50, flight_plan)\n            shipyard.next_action = action\n        elif kore_left >= spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n            shipyard.next_action = action\n\n    return me.next_actions')


# **flight_planで艦隊が進むルートを文字列で指定しています。**
# 
# **flight_plan specifies the route the fleet will take as a string.**

# In[ ]:


env.run(["/kaggle/working/pilot.py"])
env.render(mode="ipython", width=1000, height=800)


# **艦隊はそのターンの終わりに居たマスでKoreの特定の％を拾います。 この関係は対数的であるため、多くの小さな艦隊は1つの大きな艦隊よりも多くのコレを獲得します。 以下の表を参照してください。**
# 
# **The fleet picks up a certain percentage of Kore in the squares that were at the end of the turn. Because this relationship is logarithmic, many small fleets get more collection than one large fleet. See the table below.**
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

# **数式として表すと、`ln(num_ships_in_fleet) / 20`になります。**
# 
# **Expressed as a formula, it is `ln (num_ships_in_fleet) / 20`.**

# **ターンの終わりに味方の2つの艦隊が同じマスにいる場合、その場所で合体し、船の数、Koreの数が合計された艦隊に変化します。合体した艦隊は、合体する前の艦隊の中で最も船の数が多い艦隊のルートに従い行動します。実際に試してみましょう。**
# 
# **If two friendly fleets are in the same square at the end of the turn, they will merge at that location and change to a fleet with the total number of ships and Kore. The combined fleet will follow the route of the fleet with the largest number of ships in the fleet before it was combined. Let's try it out.**

# # Pick up a ship

# In[ ]:


get_ipython().run_cell_magic('writefile', 'miner.py', '   \nfrom kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n\n    period = 4 + config.size + 1\n    \n    for shipyard in me.shipyards:\n        action = None\n        if turn % period == 4:\n            action = ShipyardAction.launch_fleet_with_flight_plan(2, "ES")\n        elif turn % period == 6: \n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E2S")\n        elif turn % period == 4 + config.size:\n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E3W")\n            shipyard.next_action = action\n        elif kore_left >= spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n        shipyard.next_action = action\n\n    return me.next_actions')


# In[ ]:


env.run(["/kaggle/working/miner.py"])
env.render(mode="ipython", width=1000, height=800)


# **魔理沙:船の数が同じ艦隊が合体したらどっちのルートに従うのかな？**
# 
# **Marisa: Which route would you follow if a fleet with the same number of ships merged?**

# <img src="http://3.bp.blogspot.com/-KmQQLtEkmLw/U1T3r7D0NdI/AAAAAAAAfVI/c2d4n2kG00U/s400/figure_question.png" width="100">

# **魔理沙:次は造船所を作ってみよう！**
# 
# **Marisa: Next, let's build a shipyard!**

# # Make Shipyard

# In[ ]:


get_ipython().run_cell_magic('writefile', 'expander.py', '   \nfrom kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\n# a flight plan\ndef build_flight_plan(dir_idx, size):\n    flight_plan = ""\n    for i in range(4):\n        flight_plan += Direction.from_index((dir_idx + i) % 4).to_char()\n        if not i == 3:\n            flight_plan += str(size)\n    return flight_plan\n\ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    convert_cost = board.configuration.convert_cost\n    kore_left = me.kore\n\n    for shipyard in me.shipyards:\n        action = None\n        if kore_left >= 500 and shipyard.ship_count >= convert_cost:\n            flight_plan = build_flight_plan(randint(0, 3), randint(10, 15))\n            flight_plan = flight_plan[:6] + "C"\n            action = ShipyardAction.launch_fleet_with_flight_plan(convert_cost, flight_plan)\n        elif shipyard.ship_count >= convert_cost:\n            flight_plan = build_flight_plan(randint(0, 3), randint(2, 9))\n            action = ShipyardAction.launch_fleet_with_flight_plan(convert_cost, flight_plan)\n        elif kore_left >= spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n        shipyard.next_action = action\n\n    return me.next_actions')


# **造船所を作るには、文字列の中に”C”が入ったルートを、50隻以上の艦隊に指定する必要があります。造船所を作るには50隻の船を消費します。**
# 
# **To create a shipyard, you need to specify a route with a "C" in the string for a fleet of 50 or more ships. It consumes 50 ships to build a shipyard.**

# In[ ]:


env.run(["/kaggle/working/expander.py"])
env.render(mode="ipython", width=1000, height=800)


# **霊夢:Koreマスの上で造船所を建てることは可能なのかな？可能ならそのマスのKoreはどうなるのかな？**
# 
# **魔理沙:造船所を作る前に艦隊が敵に攻撃されて、船の数が50未満になったらどうなるのかな？**
# 
# **Reimu: Is it possible to build a shipyard on Koremas? If possible, what will happen to Kore in that square?**
# 
# **Marisa: What if the fleet is attacked by an enemy before the shipyard is built and the number of ships is less than 50?**

# <img src="http://3.bp.blogspot.com/-KmQQLtEkmLw/U1T3r7D0NdI/AAAAAAAAfVI/c2d4n2kG00U/s400/figure_question.png" width="100">

# **魔理沙:そしたら最後に、対戦させてみよう！**
# 
# **Marisa: And finally, let's play!**

# # Battle

# In[ ]:


get_ipython().run_cell_magic('writefile', 'starter.py', '# from https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kore_fleets/starter_bots/python/main.py\n   \nfrom kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n\n    for shipyard in me.shipyards:\n        if shipyard.ship_count > 10:\n            direction = Direction.from_index(turn % 4)\n            action = ShipyardAction.launch_fleet_with_flight_plan(randint(2,3), direction.to_char())\n            shipyard.next_action = action\n        elif kore_left > spawn_cost * shipyard.max_spawn:\n            action = ShipyardAction.spawn_ships(shipyard.max_spawn)\n            shipyard.next_action = action\n            kore_left -= spawn_cost * shipyard.max_spawn\n        elif kore_left > spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n            shipyard.next_action = action\n            kore_left -= spawn_cost\n\n    return me.next_actions')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'attacker.py', '# from https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kore_fleets/starter_bots/python/main.py\n   \nfrom kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef get_closest_enemy_shipyard(board, position, me):\n    min_dist = 1000000\n    enemy_shipyard = None\n    for shipyard in board.shipyards.values():\n        if shipyard.player_id == me.id:\n            continue\n        dist = position.distance_to(shipyard.position, board.configuration.size)\n        if dist < min_dist:\n            min_dist = dist\n            enemy_shipyard = shipyard\n    return enemy_shipyard\n\n\ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n\n    for shipyard in me.shipyards:\n        action = None\n        if turn % 100 < 20 and shipyard.ship_count >= 50:\n            closest_enemy_shipyard = get_closest_enemy_shipyard(board, shipyard.position, me)\n            if not closest_enemy_shipyard:\n                continue\n            enemy_pos = closest_enemy_shipyard.position\n            my_pos = shipyard.position\n            flight_plan = "N" if enemy_pos.y > my_pos.y else "S"\n            flight_plan += str(abs(enemy_pos.y - my_pos.y) - 1)\n            flight_plan += "W" if enemy_pos.x < my_pos.x else "E"\n            action = ShipyardAction.launch_fleet_with_flight_plan(shipyard.ship_count, flight_plan)            \n        elif shipyard.ship_count >= 10 and turn % 7 == 0 and turn % 100 > 20 and turn % 100 < 90:\n            direction = Direction.from_index(turn % 4)\n            opposite = direction.opposite()\n            flight_plan = direction.to_char() + "9" + opposite.to_char()\n            action = ShipyardAction.launch_fleet_with_flight_plan(10, flight_plan)\n        elif kore_left > spawn_cost * shipyard.max_spawn:\n            action = ShipyardAction.spawn_ships(shipyard.max_spawn)\n            kore_left -= spawn_cost * shipyard.max_spawn\n        elif kore_left > spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n            kore_left -= spawn_cost\n        shipyard.next_action = action\n\n    return me.next_actions')


# **env.runで複数のパスを指定すると対戦ができます。**
# 
# **You can play against each other by specifying multiple paths in env.run.**

# In[ ]:


starter_agent_path = "/kaggle/working/starter.py"
attacker_agent_path = "/kaggle/working/attacker.py"
env.run([attacker_agent_path, starter_agent_path, starter_agent_path, starter_agent_path])
env.render(mode="ipython", width=1000, height=800)


# **敵同士の艦隊が同じマスに居るとき、衝突が起きます。衝突が起きると衝突した艦隊の中で船の数が一番が多い艦隊が生き残り、船の数は2番目に多い艦隊との差になります。**
# 
# **艦隊は敵の造船所を攻撃することも可能です。敵の造船所と味方の艦隊が同じマスに居たとき、艦隊の船の数の分だけ造船所の所有する船の数が減少します。所有する船の数が0を下回ると、攻撃した側の造船所に変化します。**
# 
# **Collisions occur when fleets of enemies are in the same square. When a collision occurs, the fleet with the largest number of ships survives, and the number of ships is different from the fleet with the second largest number.**
# 
# **The fleet can also attack enemy shipyards. When an enemy shipyard and a friendly fleet are in the same square, the number of ships owned by the shipyard is reduced by the number of ships in the fleet. If the number of ships you own falls below 0, it will change to the attacking shipyard.**

# **魔理沙:勝つために相手を狙って攻撃することは、戦略として有効なのかな？**
# 
# **霊夢:いろいろなチームとの対戦リプレイがリーダーボードで閲覧できるから、参考にできるね。**
# 
# **Marisa: Is it effective as a strategy to attack the opponent in order to win?**
# 
# **Reimu: You can see the replays against various teams on the leaderboard, so you can refer to it.**

# ![](https://1.bp.blogspot.com/-PNcKwFw1PpM/U1T3oDIr9CI/AAAAAAAAfT4/gEn86X8Ppx0/s400/figure_goodjob.png)
