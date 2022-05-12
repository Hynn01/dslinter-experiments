#!/usr/bin/env python
# coding: utf-8

# [![](https://img.youtube.com/vi/FnV0thLS1Fs/0.jpg)](https://www.youtube.com/watch?v=FnV0thLS1Fs)

# pixyz  
# last update 2022 05 08  
# ゆっくりしていってね！  

# **霊夢:今回はコンペ理解のために、Miningについて検証していくぞ。**
# 
# **魔理沙:他に気になったことがあったら突っ込んでいくぞ。**
# 
# **Reimu: This time, I translated the code written by the organizer into Japanese to understand the competition.**
# 
# **Marisa: If you have any other concerns, I'll dig in.**

# # Method

# **霊夢:色々なagentを、特定のマップで動かしてみて、以下の4つの変動をグラフにして比較してみるよ。**
# 
# * Kore:Shipyardsが持っている資源の数
# * Cargo:艦隊が持っている資源の数
# * Ships:全体が持っている船の数
# * All:これまでに獲得した資源の数 (Kore + Cargo + Ships×10 - 500)
# 
# **10,500 の値はそれぞれ、造船のコスト、ゲーム開始時に所持している資源の数の事です。**
# 
# **魔理沙:randomSeed = 42に設定して、Koreの配置は全検証において同じになるようにしたぜ**
# 
# **Reimu: Try running different agents on a specific map and compare the following four fluctuations in a graph.**
# 
# * Kore: Number of resources that Shipyards have
# * Cargo: The number of resources the fleet has
# * Ships: Number of ships the player has
# * All: Number of resources acquired so far (Kore + Cargo + Ships × 10 -500)
# 
# **Values of 10,500 are the cost of shipbuilding and the number of resources you have at the start of the game, respectively.**
# 
# **Marisa: I set randomSeed = 42 so that Kore's placement is the same for all validations**

# In[ ]:


get_ipython().run_cell_magic('capture', '', '# install the latest version of kaggle_environments\n!pip install --upgrade kaggle_environments')


# In[ ]:


from kaggle_environments import make
import matplotlib.pylab as plt
env = make("kore_fleets",configuration={"randomSeed":42}, debug=True)
print(env.name, env.version)


# In[ ]:


def make_graph(steps):
    Kore = [steps[i][0]["reward"] for i in range(400)]
    Cargo = [0]*400
    Ships = [0]*400
    All = [0]*400
    for i in range(400):
        for x in steps[i][0]['observation']["players"][0][2].values():
            Cargo[i] += x[1]
    for i in range(400):
        for x in steps[i][0]['observation']["players"][0][1].values():
            Ships[i] += x[1]
        for x in steps[i][0]['observation']["players"][0][2].values():
            Ships[i] += x[2]
    for i in range(400):
        All[i] = Kore[i]+Cargo[i]+(Ships[i]-50)*10 
            
    fig, ax = plt.subplots(figsize = (20, 10))
    plt.rcParams["font.size"] = 12
    plt.subplot(2,2,1)
    plt.plot(Kore)
    plt.title("Kore")
    plt.grid()
    plt.subplot(2,2,2)    
    plt.plot(Cargo)
    plt.title("Cargo")
    plt.grid()
    plt.subplot(2,2,3)
    plt.plot(Ships)
    plt.title('Ships')
    plt.grid()
    plt.subplot(2,2,4)
    plt.plot(All)
    plt.title('All')
    plt.grid()
    # plt.title(df_main.loc[0,columns[0]])


# In[ ]:


def vs_make_graph(step_list,agent_name):
    Kore = [] 
    Cargo = []
    Ships = []
    All = []
    for j,steps in enumerate(step_list):
        Kore.append([steps[i][0]["reward"] for i in range(400)])
        Cargo.append([0]*400) 
        Ships.append([0]*400)
        All.append([0]*400)
        for i in range(400):
            for x in steps[i][0]['observation']["players"][0][2].values():
                Cargo[j][i] += x[1]
        for i in range(400):
            for x in steps[i][0]['observation']["players"][0][1].values():
                Ships[j][i] += x[1]
            for x in steps[i][0]['observation']["players"][0][2].values():
                Ships[j][i] += x[2]
        for i in range(400):
            All[j][i] = Kore[j][i]+Cargo[j][i]+(Ships[j][i]-50)*10 

    fig, ax = plt.subplots(figsize = (20, 30))
    plt.rcParams["font.size"] = 20
    plt.subplot(4,1,1)
    for i in range(len(step_list)):    
        plt.plot(Kore[i],label=agent_name[i])
    plt.title("Kore")
    plt.legend()
    plt.grid()
    plt.subplot(4,1,2)    
    for i in range(len(step_list)):    
        plt.plot(Cargo[i],label=agent_name[i])
    plt.title("Cargo")
    plt.legend()
    plt.grid()
    plt.subplot(4,1,3)
    for i in range(len(step_list)): 
        plt.plot(Ships[i],label=agent_name[i])
    plt.title('Ships')
    plt.legend()
    plt.grid()
    plt.subplot(4,1,4)
    for i in range(len(step_list)):    
        plt.plot(All[i],label=agent_name[i])
    plt.title('All')
    plt.legend()
    plt.grid()


# # No.1 one_box_mining

# In[ ]:


get_ipython().run_cell_magic('writefile', 'box_miner.py', '   \nfrom kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef agent(obs, config):\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n\n    period = 40\n    \n    for shipyard in me.shipyards:\n        action = None\n        if turn < 40:\n            action = ShipyardAction.spawn_ships(1)\n        elif turn % period == 1:\n            action = ShipyardAction.launch_fleet_with_flight_plan(21, "E9N9W9S")\n        elif turn % period == 3: \n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E8N")\n        elif turn % period == 5: \n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E7N")\n        elif turn % period == 7: \n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E6N")\n        elif turn % period == 9: \n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E5N")\n        elif turn % period == 11: \n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E4N")\n        elif turn % period == 13: \n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E3N")\n        elif turn % period == 15: \n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E2N")\n        elif turn % period == 17: \n            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E1N")\n        elif turn % period == 19: \n            action = ShipyardAction.launch_fleet_with_flight_plan(2, "EN")\n        elif turn % period == 21: \n            action = ShipyardAction.launch_fleet_with_flight_plan(2, "N")\n        elif kore_left >= spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n            \n        \n        shipyard.next_action = action\n\n    return me.next_actions')


# **魔理沙:Kore Intro IIに出てきたbox_minerだぜ。**
# 
# **Marisa: It's the box_miner from Kore Intro II.**

# In[ ]:


step1 = env.run(["/kaggle/working/box_miner.py"])
env.render(mode="ipython", width=1000, height=800)


# ## Results

# In[ ]:


make_graph(step1)


# # No.2 two_box_mining

# In[ ]:


get_ipython().run_cell_magic('writefile', 'two_minning.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef box_minning(agent,turn,box_size):\n\n    period = 40\n    for shipyard in agent.shipyards:\n        if shipyard.ship_count >= 40 and turn % period == 1:\n            box_size = 1\n        if shipyard.ship_count >= 89 and turn % period == 1:\n            box_size = 2\n        action = None\n        \n        \n        if turn > 40:\n            if turn % period == 1:\n                action = ShipyardAction.launch_fleet_with_flight_plan(21, "E9N9W9S")\n            elif turn % period == 3: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E8N")\n            elif turn % period == 5: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E7N")\n            elif turn % period == 7: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E6N")\n            elif turn % period == 9: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E5N")\n            elif turn % period == 11: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E4N")\n            elif turn % period == 13: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E3N")\n            elif turn % period == 15: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E2N")\n            elif turn % period == 17: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E1N")\n            elif turn % period == 19: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2, "EN")\n            elif turn % period == 21: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2, "N")\n        if box_size > 1:\n            if turn % period == 2:\n                action = ShipyardAction.launch_fleet_with_flight_plan(21, "W9S9E9N")\n            elif turn % period == 4: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W8S")\n            elif turn % period == 6: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W7S")\n            elif turn % period == 8: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W6S")\n            elif turn % period == 10: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W5S")\n            elif turn % period == 12: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W4S")\n            elif turn % period == 14: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W3S")\n            elif turn % period == 16: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W2S")\n            elif turn % period == 18: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W1S")\n            elif turn % period == 20: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2, "WS")\n            elif turn % period == 22: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2, "S")\n            \n        shipyard.next_action = action\n        \n    return int(box_size)')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'spawn.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef spawn(agent,spawn_cost,kore_left):\n    for shipyard in agent.shipyards:\n        if shipyard.next_action:\n            continue\n        if kore_left >= spawn_cost * shipyard.max_spawn:\n            action = ShipyardAction.spawn_ships(shipyard.max_spawn)\n            shipyard.next_action = action\n            kore_left -= spawn_cost * shipyard.max_spawn\n        elif kore_left >= spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n            shipyard.next_action = action\n            kore_left -= spawn_cost\n\n    return')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'main.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\nfrom two_minning import box_minning\nfrom spawn import spawn\n\nbox_num = 0\n\ndef agent(obs, config):\n    global box_num\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n    \n    box_num = box_minning(me,turn,box_num)\n    spawn(me,spawn_cost,kore_left)\n    return me.next_actions')


# In[ ]:


step2 = env.run(["/kaggle/working/main.py"])
env.render(mode="ipython", width=1000, height=800)


# ## Results

# In[ ]:


make_graph(step2)


# # No.3 four_box_miner

# In[ ]:


get_ipython().run_cell_magic('writefile', 'four_minning.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef box_minning(agent,turn,box_size):\n\n    period = 40\n    for shipyard in agent.shipyards:\n        if shipyard.ship_count >= 40 and turn % period == 10 and box_size < 1:\n            box_size = 1\n        if shipyard.ship_count >= 89 and turn % period == 10 and box_size < 2:\n            box_size = 2\n        if shipyard.ship_count >= 140 and turn % period == 10 and box_size < 3:\n            box_size = 3\n        if shipyard.ship_count >= 190 and turn % period == 10 and box_size < 4:\n            box_size = 4\n        action = None\n        \n        \n        if turn > 9:\n            if turn % period == 10:\n                action = ShipyardAction.launch_fleet_with_flight_plan(21, "E9N9W9S")\n            elif turn % period == 12: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E8N")\n            elif turn % period == 14: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E7N")\n            elif turn % period == 16: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E6N")\n            elif turn % period == 18: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E5N")\n            elif turn % period == 20: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E4N")\n            elif turn % period == 22: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E3N")\n            elif turn % period == 24: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E2N")\n            elif turn % period == 26: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E1N")\n            elif turn % period == 28: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2, "EN")\n        if box_size > 1:\n            if turn % period == 11:\n                action = ShipyardAction.launch_fleet_with_flight_plan(21, "W9S9E9N")\n            elif turn % period == 13: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W8S")\n            elif turn % period == 15: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W7S")\n            elif turn % period == 17: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W6S")\n            elif turn % period == 19: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W5S")\n            elif turn % period == 21: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W4S")\n            elif turn % period == 23: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W3S")\n            elif turn % period == 25: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W2S")\n            elif turn % period == 27: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W1S")\n            elif turn % period == 29: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2, "WS")\n        if box_size > 2:\n            if turn % period == 30:\n                action = ShipyardAction.launch_fleet_with_flight_plan(21, "E9S9W9N")\n            elif turn % period == 32: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E8S")\n            elif turn % period == 34: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E7S")\n            elif turn % period == 36: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E6S")\n            elif turn % period == 38: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E5S")\n            elif turn % period == 0: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E4S")\n            elif turn % period == 2: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E3S")\n            elif turn % period == 4: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E2S")\n            elif turn % period == 6: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "E1S")\n            elif turn % period == 8: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2, "ES")\n        if box_size > 3:\n            if turn % period == 31:\n                action = ShipyardAction.launch_fleet_with_flight_plan(21, "W9N9E9S")\n            elif turn % period == 33: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W8N")\n            elif turn % period == 35: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W7N")\n            elif turn % period == 37: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W6N")\n            elif turn % period == 39: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W5N")\n            elif turn % period == 1: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W4N")\n            elif turn % period == 3: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W3N")\n            elif turn % period == 5: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W2N")\n            elif turn % period == 7: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3, "W1N")\n            elif turn % period == 9: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2, "WN")    \n        shipyard.next_action = action\n        \n    return int(box_size)')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'main2.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\nfrom four_minning import box_minning\nfrom spawn import spawn\n\nbox_num = 0\n\ndef agent(obs, config):\n    global box_num\n    board = Board(obs, config)\n    me=board.current_player\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n    \n    box_num = box_minning(me,turn,box_num)\n    spawn(me,spawn_cost,kore_left)\n    return me.next_actions')


# In[ ]:


step3 = env.run(["/kaggle/working/main2.py"])
env.render(mode="ipython", width=1000, height=800)


# ## Results

# In[ ]:


make_graph(step3)


# # one_box_mining vs two_box_mining vs four_box_mining

# In[ ]:


step_list = [step1,step2,step3]
agent_name = ["one_box_mining","two_box_mining","four_box_mining"]
vs_make_graph(step_list,agent_name)


# **魔理沙:Koreは、200ターン目までどのエージェントも同じくらいだけど、最終的には、four_box_miningが一番多くなってるぜ。**
# 
# **霊夢:Cargoは150ターン目から増加してきて、300ターン目あたりからは振幅がほぼ一定になるね。**
# 
# **魔理沙:Shipsは200ターン目あたりまではどのエージェントも同じくらいだけど、そこからone_box_miningは少しずつ増加していって、two_box_miningは大きく増加していって、four_box_miningは完全に増加しなくなるね。four_box_miningが増加しなくなるのは、艦隊を出し続けているせいで、造船する暇がないからだろうな**
# 
# **霊夢:All (取得した資源の総数：Kore + Cargo + Ships*10 -500) は、150ターン目あたりまではどのエージェントも大差ないけど、そこから差が開いて行って、最終的にはfour_box_miningが最も多くなったよ。**
# 
# **魔理沙:two_box_miningはone_box_miningの約2倍になってるけど、for_box_miningはtwo_box_miningの2倍やone_box_miningの4倍より少なそうだね。**
# 
# **Marisa: Kore is about the same for all agents until turn 200, but in the end, four_box_mining is the most.**
# 
# **Reimu: Cargo increases from turn 150, and the amplitude becomes almost constant from around turn 300.**
# 
# **Marisa: Ships is about the same for all agents until around turn 200, but from there one_box_mining is increasing little by little, two_box_mining is increasing significantly, and four_box_mining isn't increasing altogether. The reason why four_box_mining doesn't increase is probably because we don't have time to build a ship because we keep out the fleet**
# 
# **Reimu: All (total number of resources acquired: Kore + Cargo + Ships * 10 -500) is not much different for any agent until around the 150th turn, but the difference comes out from there, and finally four_box_mining is the most.**
# 
# **Marisa: Two_box_mining is about twice as much as one_box_mining, but for_box_mining is less than twice as much as two_box_mining and four times as much as one_box_mining.**

# # No.4 4box&1shipyard

# In[ ]:


get_ipython().run_cell_magic('writefile', 'four_minning2.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef box_minning(agent,turn,box_size,ships_num):\n\n    period = 40\n    for shipyard in agent.shipyards:\n        if not shipyard.id == \'0-1\':\n            continue\n        if not shipyard.next_action == None:\n            continue\n        \n        if shipyard.ship_count >= 40 and turn % period == 10 and box_size < 1:\n            box_size = 1\n        if shipyard.ship_count >= 89 and turn % period == 10 and box_size < 2:\n            box_size = 2\n        if shipyard.ship_count >= 140 and turn % period == 10 and box_size < 3:\n            box_size = 3\n        if shipyard.ship_count >= 190 and turn % period == 10 and box_size < 4:\n            box_size = 4\n        action = None\n        \n        k =0 if box_size == 0 else (ships_num - 47*box_size) // (10 * box_size)\n        \n        if turn > 9:\n            if turn % period == 10:\n                action = ShipyardAction.launch_fleet_with_flight_plan(21 + k, "E9N9W9S")\n            elif turn % period == 12: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "E8N")\n            elif turn % period == 14: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "E7N")\n            elif turn % period == 16: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "E6N")\n            elif turn % period == 18: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "E5N")\n            elif turn % period == 20: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "E4N")\n            elif turn % period == 22: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "E3N")\n            elif turn % period == 24: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "E2N")\n            elif turn % period == 26: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "E1N")\n            elif turn % period == 28: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2 + k, "EN")\n        if box_size > 1:\n            if turn % period == 11:\n                action = ShipyardAction.launch_fleet_with_flight_plan(21 + k, "W9S9E9N")\n            elif turn % period == 13: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "W8S")\n            elif turn % period == 15: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "W7S")\n            elif turn % period == 17: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "W6S")\n            elif turn % period == 19: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "W5S")\n            elif turn % period == 21: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "W4S")\n            elif turn % period == 23: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "W3S")\n            elif turn % period == 25: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "W2S")\n            elif turn % period == 27: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "W1S")\n            elif turn % period == 29: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2 + k, "WS")\n        if box_size > 2:\n            if turn % period == 30:\n                action = ShipyardAction.launch_fleet_with_flight_plan(21 + k, "E9S9W9N")\n            elif turn % period == 32: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "E8S")\n            elif turn % period == 34: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "E7S")\n            elif turn % period == 36: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "E6S")\n            elif turn % period == 38: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "E5S")\n            elif turn % period == 0: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "E4S")\n            elif turn % period == 2: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "E3S")\n            elif turn % period == 4: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "E2S")\n            elif turn % period == 6: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "E1S")\n            elif turn % period == 8: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2 + k, "ES")\n        if box_size > 3:\n            if turn % period == 31:\n                action = ShipyardAction.launch_fleet_with_flight_plan(21 + k, "W9N9E9S")\n            elif turn % period == 33: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "W8N")\n            elif turn % period == 35: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "W7N")\n            elif turn % period == 37: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "W6N")\n            elif turn % period == 39: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "W5N")\n            elif turn % period == 1: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "W4N")\n            elif turn % period == 3: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "W3N")\n            elif turn % period == 5: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "W2N")\n            elif turn % period == 7: \n                action = ShipyardAction.launch_fleet_with_flight_plan(3 + k, "W1N")\n            elif turn % period == 9: \n                action = ShipyardAction.launch_fleet_with_flight_plan(2 + k, "WN")    \n        shipyard.next_action = action\n        \n    return int(box_size)')


# %%writefile make_shipyard.py

# In[ ]:


get_ipython().run_cell_magic('writefile', 'sub_function.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\n\ndef make_shipyard(agent,turn,box_num):\n    period = 40\n    action = None\n    for shipyard in agent.shipyards:\n        if not shipyard.id == \'0-1\':\n            continue\n        if box_num == 2 and (turn % period > 29 or turn % period < 10) and shipyard.ship_count >= 50:\n            action = ShipyardAction.launch_fleet_with_flight_plan(50, "NC")\n            shipyard.next_action = action\n            return True\n        \n    return False\n\ndef deport(agent,turn):\n    period = 40\n    for shipyard in agent.shipyards:\n        action = None\n        if shipyard.id == \'0-1\':\n            continue\n        if shipyard.ship_count >= 47:\n            action = ShipyardAction.launch_fleet_with_flight_plan(shipyard.ship_count, "S")\n        shipyard.next_action = action\n    \n    return\n            \ndef all_ship_count(agent):\n    res = 0\n    for shipyard in agent.shipyards:\n        res += shipyard.ship_count\n    \n    for fleet in agent.fleets:\n        res += fleet.ship_count\n    return res')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'main3.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\nfrom four_minning2 import box_minning\nfrom sub_function import make_shipyard,deport,all_ship_count\nfrom spawn import spawn\n\nbox_num = 0\nmake = False\ndef agent(obs, config):\n    global box_num\n    global make\n    board = Board(obs, config)\n    me=board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n    ships = all_ship_count(me)\n    \n    if not make: make = make_shipyard(me,turn,box_num)    \n    box_num = box_minning(me,turn,box_num,ships)\n    deport(me,turn)\n    spawn(me,spawn_cost,kore_left)\n    return me.next_actions')


# In[ ]:


step4 = env.run(["/kaggle/working/main3.py"])
env.render(mode="ipython", width=1000, height=800)


# In[ ]:


make_graph(step4)


# # four_box_minnig vs 4box&1shipyard

# In[ ]:


step_list = [step3,step4]
agent_name = ["four_box_mining","4box&1shipyard"]
vs_make_graph(step_list,agent_name)


# In[ ]:


get_ipython().system('mkdir ./submission ')
get_ipython().system('cp ./four_minning2.py ./submission/four_minning2.py')
get_ipython().system('cp ./sub_function.py ./submission/sub_function.py')
get_ipython().system('cp ./spawn.py ./submission/spawn.py')
get_ipython().system('cp ./main3.py ./submission/main.py')
get_ipython().system('zip -r ./submission.tar.gz ./submission ')

