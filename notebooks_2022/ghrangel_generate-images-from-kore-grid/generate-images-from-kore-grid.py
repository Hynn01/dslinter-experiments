#!/usr/bin/env python
# coding: utf-8

# <h1 id="dataset" style="color:#FFFFFF; background:#205375; border:1.5px dotted;"> 
#     <center>kaggle_environments
#         <a class="anchor-link" href="#dataset" target="_self"></a>
#     </center>
# </h1>

# In[ ]:


get_ipython().run_cell_magic('capture', '', '# install the latest version of kaggle_environments\n!pip install --upgrade kaggle_environments')


# In[ ]:


from kaggle_environments import make
env = make("kore_fleets", debug=True)
print(env.name, env.version)


# <h1 id="dataset" style="color:#FFFFFF; background:#A0BCC2; border:1.5px dotted;"> 
#     <center>miner.py
#         <a class="anchor-link" href="#dataset" target="_self"></a>
#     </center>
# </h1>

# In[ ]:


get_ipython().run_cell_magic('writefile', 'miner.py', '\nimport pandas as pd\nfrom random import random, sample, randint\nfrom kaggle_environments import utils\nfrom kaggle_environments.helpers import Point, Direction\nfrom kaggle_environments.envs.kore_fleets.helpers import Board, ShipyardAction\nfrom PIL import Image as im\nimport numpy as np\nimport matplotlib.pyplot as plt \n\ndef get_kore(obs, config):\n    GRID_SIZE = config.size\n    kore_grid_ = np.flip(np.array(obs["kore"], dtype=np.float32).reshape(GRID_SIZE, GRID_SIZE), axis=0)\n    return kore_grid_\n\ndef save_image(array):    \n    #array = np.where(array >10, 50, array)\n    \n    # in the final turns the kore increases but I normalize the data\n    array = array / np.sqrt(np.sum(array**2))\n    array=array*255\n    array=np.uint16(array)\n    \n    #put data 5,5 or data 15,15 in 255\n    array[5,5]=255\n    #array[15,15]=255\n    array = (array * 255).astype(np.uint16)\n   \n    data = im.fromarray(array)\n    data.save(\'separated100.png\')    \n    \ndef generate_image(array):\n    # in the final turns the kore increases but I normalize the data\n    array = array / np.sqrt(np.sum(array**2))\n    array=array*255\n    array=np.uint8(array)  \n    \n    #array = np.where(array >30, 200, array)    \n    array[5,5]=255\n    #array[15,15]=255\n    print(array)  \n    \n    plt.imshow(array)\n    plt.show() \n    \ndef miner_agent(obs, config):\n    board = Board(obs, config)\n    me = board.current_player\n    kore_left = me.kore\n    shipyards = me.shipyards\n    \n    spawn_cost = board.configuration.spawn_cost\n    # randomize shipyard order\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_grid = get_kore(obs, config)\n    #*************************************************************\n    #*********************IMPORTANT*******************************\n    #*************************************************************\n    # GENERATE 1,50,100 AND 200 TURNS\n    #you can change the turns\n    # in the final turns the kore increases but I normalize the data\n    \n    if turn==1:\n        generate_image(kore_grid)\n        save_image(kore_grid)\n        \n    if turn==50:\n        generate_image(kore_grid)\n        \n    if turn==100:\n        generate_image(kore_grid)\n        \n    if turn==200:\n        generate_image(kore_grid)       \n        \n        \n    shipyards = sample(shipyards, len(shipyards))\n    period =7\n    \n    for shipyard in shipyards:\n        action = None\n        #I only put a flight plan, to test\n        if turn % period == 2:\n            if(shipyard.ship_count >= 22):\n                action = ShipyardAction.launch_fleet_with_flight_plan(22, "S3E8N3W")\n                shipyard.next_action = action\n           \n            elif kore_left > board.configuration.spawn_cost * shipyard.max_spawn:\n                kore_left -= board.configuration.spawn_cost\n                if kore_left >= spawn_cost:\n                    shipyard.next_action = ShipyardAction.spawn_ships(min(shipyard.max_spawn, int(kore_left/spawn_cost))) \n    return me.next_actions')


# <h1 id="dataset" style="color:#FFFFFF; background:#A0BCC2; border:1.5px dotted;"> 
#     <center>GENERATE IMAGES IN  1,50,100 AND 200 TURNS
#         <a class="anchor-link" href="#dataset" target="_self"></a>
#     </center>
#     <center style="color:#000000;" >IF I HELP YOU, GIVE ME A VOTE
#         <a class="anchor-link" href="#dataset" target="_self"></a>
#     </center>
# </h1>

# In[ ]:


env.run(["/kaggle/working/miner.py", "do_nothing"])
env.render(mode="ipython", width=1000, height=800)

