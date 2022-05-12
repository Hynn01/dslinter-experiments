#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import json
from pathlib import Path
import os
import random
from tqdm.notebook import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import matplotlib.pyplot as plt


# In[ ]:


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed = 42
seed_everything(seed)


# # Preprocessing

# In[ ]:


def balance_samples(smps, n, maxn):
    out = []
    labels = [sample[-1] for sample in samples]
    types = len(Counter(labels).keys())
    groups = [[] for i in range(types + 1)]
    for smp in smps:
        if(len(groups[smp[-1]]) < maxn):
            groups[smp[-1]].append(smp)
            out.append(smp)
    for i in range(types):
        while(len(groups[i]) < n):
            groups[i].append(groups[i][random.randint(0, len(groups[i]) - 1)])
            out.append(groups[i][random.randint(0, len(groups[i]) - 1)])
    return out
    
def decode_pos(p):
    return (p - (p - 1) % 21 - 1) // 21 - 1, (p - 1) % 21 
def get_shipyard_info(lst, sy_id):
    p, sp, shp = lst
    x, y = decode_pos(p)
    return [(x-1, y-1), shp, sp, sy_id]
def get_ship_info(lst):
    return (decode_pos(lst[0]), lst[1], lst[2])
def decode_dir(d):
    return {"S" : 1, "W" : 2, "N" : 3, "E" : 4, "C": 5}[d]
def decode_flight_plan(plan):
    dirs = [decode_dir(i) for i in re.sub(r'[^A-Z ]+','', plan)]
    dirs += [0] * (4 - len(dirs))
    lengths = [min(31,int(i)) for i in re.split("S|W|N|E|C",plan) if(i)]
    if(len(lengths) < len(dirs)):
        lengths.append(31)
    lengths += [0] * (4 - len(lengths))
    return dirs + lengths
def decode_action(a):
    a = a.split("_")

    if("LAUNCH" in a):
        return [0, int(a[1])] + a[2]
    else:
        return [1, int(a[1])]

def get_all_actions(episode_dir, min_cnt):
    actions = list()
    paths = [path for path in Path(episode_dir).glob('*.json') if 'info' not in path.name]
    for filepath in tqdm(paths):
        with open(filepath) as f:
            json_load = json.load(f)
            ep_id = json_load['info']['EpisodeId']
            index = np.argmax([r or 0 for r in json_load['rewards']])
            for i in range(len(json_load['steps'])-1):
                if json_load['steps'][i][index]['status'] == 'ACTIVE':
                    for a in json_load['steps'][i+1][index]['action']:
                        action = json_load['steps'][i+1][index]['action'][a]
                        if "SPAWN" in action:
                            action="SPAWN_10"
                        actions.append(action)
    actions = [i[0] for i in Counter(actions).most_common() if(i[1]>=min_cnt)]
    with open("all_actions.txt", "w") as f:
        f.write(" ".join(actions))
    return actions
def create_dataset_from_json(episode_dir, possible_actions):
    ob_id = -1
    obses = {}
    samples = []
    paths = [path for path in Path(episode_dir).glob('*.json') if 'info' not in path.name]
    for filepath in tqdm(paths):
        with open(filepath) as f:
            json_load = json.load(f)
            ep_id = json_load['info']['EpisodeId']
            index = np.argmax([r or 0 for r in json_load['rewards']])
            for i in range(len(json_load['steps'])-1):
                if json_load['steps'][i][index]['status'] == 'ACTIVE':
                    ob_id += 1
                    player = json_load['steps'][i][0]['observation']['player']
                    me = json_load['steps'][i][0]['observation']['players'][player]
                    acts = json_load['steps'][i+1][index]['action']
                    actions = [[decode_pos(me[1][a][0]),acts[a]] for a in acts.keys() if a in me[1]]
                    for pos, a in actions:
                        if(a in possible_actions):
                            if "SPAWN" in a:
                                a = "SPAWN_10"
                            samples.append([ob_id, pos, possible_actions.index(a)])
                        #else:
                        #    if(random.randint(1,100)==50):
                        #        print(pos,a)
                    obses[ob_id] = json_load['steps'][i][0]['observation']
                    
                        
    return obses, samples


# In[ ]:


episode_dir = "../input/koreepisodes"


# In[ ]:


possible_actions = get_all_actions(episode_dir, 100)


# In[ ]:


obses, samples = create_dataset_from_json(episode_dir, possible_actions)
print('obses:', len(obses), 'samples:', len(samples), "possible actions:", len(possible_actions))


# # Training

# In[ ]:


def make_obs(obs, pos):
    kore = obs['kore']
    step=obs["step"]
    player = obs['player']
    me = obs['players'][player]
    opponent = obs['players'][1 - player]
    my_shipyards = [get_shipyard_info(me[1][sy_id],sy_id) for sy_id in me[1]]
    opponent_shipyards = [get_shipyard_info(opponent[1][sy_id],sy_id) for sy_id in opponent[1]]
    my_ships = [get_ship_info(me[2][sy_id]) for sy_id in me[2]]
    opponent_ships = [get_ship_info(opponent[2][sy_id]) for sy_id in opponent[2]]
    observation = np.zeros((8,21,21))
    observation[0] = np.array(kore).reshape((21,21))
    for sy in my_shipyards:
        observation[1][sy[0]] = sy[1]
    for sy in my_shipyards:
        observation[2][sy[0]] = sy[2]
    for sp in my_ships:
        observation[3][sp[0]] = sp[1]
    for sp in my_ships:
        observation[4][sp[0]] = sp[2]
    observation[5][pos] = 1
    observation[6][(0,0)] = me[0]
    observation[6][(0,1)] = len(me[1])
    observation[6][(0,2)] = len(me[2])
    observation[6][(1,0)] = opponent[0]
    observation[6][(1,1)] = len(opponent[1])
    observation[6][(1,2)] = len(opponent[2])
    observation[6][2:]=step
    observation[7] = np.random.rand(21,21)
    return observation


# In[ ]:


class KoreDataset(Dataset):
    def __init__(self, obses, samples):
        self.obses = obses
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_id, pos,  action = self.samples[idx]
        obs = make_obs(self.obses[obs_id], pos)
        
        return obs, action


# In[ ]:


class BasicConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim, 
            kernel_size=kernel_size, 
            padding=(kernel_size[0] // 2, kernel_size[1] // 2)
        )
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        return h


class KoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32
        self.conv0 = BasicConv2d(8, filters, (3, 3), True)
        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, len(possible_actions), bias=False)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        p = self.head_p(h_head)
        return p


# In[ ]:


def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs):
    best_acc = 0.0
    train_accuracy = []
    val_accuracy = []
    for epoch in range(num_epochs):
        model.cuda()
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            epoch_loss = 0.0
            epoch_acc = 0
            
            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                states = item[0].cuda().float()
                actions = item[1].cuda().long()

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    policy = model(states)
                    #print(actions)
                    loss = criterion(policy, actions)
                    _, preds = torch.max(policy, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(policy)
                    epoch_acc += torch.sum(preds == actions.data)

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = epoch_acc.double() / data_size
            
            if phase == 'train':
                train_accuracy.append(epoch_acc)
            else:
                val_accuracy.append(epoch_acc)
            
            print(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
        
        if epoch_acc > best_acc:
            traced = torch.jit.trace(model.cpu(), torch.rand(32, 8, 3, 3))
            traced.save('model.pth')
            best_acc = epoch_acc
    return train_accuracy, val_accuracy


# In[ ]:


model = KoreNet()
labels = [sample[-1] for sample in samples]
train, val = train_test_split(samples, test_size=0.1, random_state=42, stratify=labels)
train, val = balance_samples(train, 720, 720), balance_samples(val, 80, 80)
batch_size = 32
train_loader = DataLoader(
    KoreDataset(obses, train), 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=2
)
val_loader = DataLoader(
    KoreDataset(obses, val), 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=2
)
dataloaders_dict = {"train": train_loader, "val": val_loader}
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


# In[ ]:


history = [[], []]
schedule = [
    (1e-3, 2),
    (1e-4, 4),
    (1e-5, 4),
    (1e-6, 2)
]
for lr, epochs in schedule:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_accuracy, val_accuracy = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=epochs)
    history[0] += train_accuracy
    history[1] += val_accuracy


# In[ ]:


plt.plot([i.cpu().numpy() for i in history[0]])
plt.title("Train accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()


# In[ ]:


plt.plot([i.cpu().numpy() for i in history[1]])
plt.title("Val accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()


# # Submission

# In[ ]:


get_ipython().run_cell_magic('writefile', 'main.py', 'import os\nimport numpy as np\nimport torch\nfrom kaggle_environments.envs.kore_fleets.helpers import *\nfrom random import randint\nimport math\n\n\npath = \'/kaggle_simulations/agent\' if os.path.exists(\'/kaggle_simulations\') else \'.\'\nmodel = torch.jit.load(f\'{path}/model.pth\')\nwith open(f\'{path}/all_actions.txt\') as f:\n    all_actions = f.read().split()\nmodel.eval()\ndef decode_pos(p):\n    return (p - (p - 1) % 21 - 1) // 21 - 1, (p - 1) % 21 \ndef get_shipyard_info(lst, sy_id):\n    p, sp, shp = lst\n    x, y = decode_pos(p)\n    return [(x-1, y-1), shp, sp, sy_id]\ndef get_ship_info(lst):\n    return (decode_pos(lst[0]), lst[1], lst[2])\ndef submit_action(a, n, sp):\n    a = a.split("_")\n\n    if("LAUNCH" in a and n>=2 and math.floor(2 * math.log(n)) + 1 >= len(a[2])):\n        return ShipyardAction.launch_fleet_with_flight_plan(min(int(a[1]), n), a[2])\n    else:\n        #if("LAUNCH" in a):\n        #    print("Can\'t launch", *a)\n        return ShipyardAction.spawn_ships(min(int(a[1]), sp))\ndef make_obs(obs, pos):\n    kore = obs[\'kore\']\n    step=obs["step"]\n    player = obs[\'player\']\n    me = obs[\'players\'][player]\n    opponent = obs[\'players\'][1 - player]\n    my_shipyards = [get_shipyard_info(me[1][sy_id],sy_id) for sy_id in me[1]]\n    opponent_shipyards = [get_shipyard_info(opponent[1][sy_id],sy_id) for sy_id in opponent[1]]\n    my_ships = [get_ship_info(me[2][sy_id]) for sy_id in me[2]]\n    opponent_ships = [get_ship_info(opponent[2][sy_id]) for sy_id in opponent[2]]\n    observation = np.zeros((8,21,21))\n    observation[0] = np.array(kore).reshape((21,21))\n    for sy in my_shipyards:\n        observation[1][sy[0]] = sy[1]\n    for sy in my_shipyards:\n        observation[2][sy[0]] = sy[2]\n    for sp in my_ships:\n        observation[3][sp[0]] = sp[1]\n    for sp in my_ships:\n        observation[4][sp[0]] = sp[2]\n    observation[5][pos] = 1\n    observation[6][(0,0)] = me[0]\n    observation[6][(0,1)] = len(me[1])\n    observation[6][(0,2)] = len(me[2])\n    observation[6][(1,0)] = opponent[0]\n    observation[6][(1,1)] = len(opponent[1])\n    observation[6][(1,2)] = len(opponent[2])\n    observation[6][2:]=step\n    observation[7] = np.random.rand(21,21)\n    return observation\ndef agent(obs, config):\n    board = Board(obs, config)\n    me = board.current_player\n    for pos, shipyard in zip([decode_pos(i[0]) for i in obs[\'players\'][obs[\'player\']][1].values()], me.shipyards):\n        if(board.step>0):\n            state = make_obs(obs,pos)\n            with torch.no_grad():\n                p = model(torch.from_numpy(state).float().unsqueeze(0))\n\n            policy = p.squeeze(0).numpy()\n            shipyard.next_action = submit_action(all_actions[policy.argmax()], shipyard.ship_count, shipyard.max_spawn)\n        else:\n            shipyard.next_action = ShipyardAction.spawn_ships(shipyard.max_spawn)\n    return me.next_actions')


# In[ ]:


from kaggle_environments import make
env = make("kore_fleets", debug=True)
env.run(["/kaggle/working/main.py", "/kaggle/working/main.py"])
env.render(mode="ipython", width=1000, height=800)


# In[ ]:


get_ipython().system('tar -czf submission.tar.gz *')

