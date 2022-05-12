#!/usr/bin/env python
# coding: utf-8

# # Install kaggle-environments

# In[ ]:


# 1. Enable Internet in the Kernel (Settings side pane)

# 2. Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 
# !curl -X PURGE https://pypi.org/simple/kaggle-environments

# ConnectX environment was defined in v0.1.6
get_ipython().system("pip install 'kaggle-environments>=0.1.6'")


# # Create ConnectX Environment

# In[ ]:


from kaggle_environments import evaluate, make, utils

env = make("connectx", debug=True)
env.render()


# # Create an Agent
# 
# To create the submission, an agent function should be fully encapsulated (no external dependencies).  
# 
# When your agent is being evaluated against others, it will not have access to the Kaggle docker image.  Only the following can be imported: Python Standard Library Modules, gym, numpy, scipy, pytorch (1.3.1, cpu only), and more may be added later.
# 
# 

# In[ ]:


# This agent random chooses a non-empty column.
def my_agent(observation, configuration):
    from random import choice
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])


# # Test your Agent

# In[ ]:


env.reset()
# Play as the first agent against default "random" agent.
env.run([my_agent, "random"])
env.render(mode="ipython", width=500, height=450)


# # Debug/Train your Agent

# In[ ]:


# Play as first position against random agent.
trainer = env.train([None, "random"])

observation = trainer.reset()

while not env.done:
    my_action = my_agent(observation, env.configuration)
    print("My Action", my_action)
    observation, reward, done, info = trainer.step(my_action)
    # env.render(mode="ipython", width=100, height=90, header=False, controls=False)
env.render()


# # Evaluate your Agent

# In[ ]:


def mean_reward(rewards):
    return sum(r[0] for r in rewards) / float(len(rewards))

# Run multiple episodes to estimate its performance.
print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))
print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))


# # Play your Agent
# Click on any column to place a checker there ("manually select action").

# In[ ]:


# "None" represents which agent you'll manually play as (first or second player).
env.play([None, "negamax"], width=500, height=450)


# # Write Submission File
# 
# 

# In[ ]:


import inspect
import os

def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)

write_agent_to_file(my_agent, "submission.py")


# # Validate Submission
# Play your submission against itself.  This is the first episode the competition will run to weed out erroneous agents.
# 
# Why validate? This roughly verifies that your submission is fully encapsulated and can be run remotely.

# In[ ]:


# Note: Stdout replacement is a temporary workaround.
import sys
out = sys.stdout
submission = utils.read_file("/kaggle/working/submission.py")
agent = utils.get_last_callable(submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")


# # Submit to Competition
# 
# 1. Commit this kernel.
# 2. View the commited version.
# 3. Go to "Data" section and find submission.py file.
# 4. Click "Submit to Competition"
# 5. Go to [My Submissions](https://kaggle.com/c/connectx/submissions) to view your score and episodes being played.
