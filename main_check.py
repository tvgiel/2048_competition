###############################################################
# this file is used to train and evaluate any submitted agent #
###############################################################

import agents.gemini_AI as model1
import agents.random_AI as randommodel
import agents.one_direction_AI as one_direction_model
import convolutional_model as Convolutionalmodel
import numpy as np



def play_game(agent):
    # always load the game environment first so then the agent can interact with it
    env = Game2048Env()
    state = env.reset()

    # Do not change the code below, as this is the main loop of the game. The agent should interact with the environment through this loop.
    while not env.done:
        valid_actions = env.get_available_actions()
        action = agent.act(state, valid_actions)
        next_state, reward, done, info = env.step(action)
        state = next_state
    return info['score'], info['highest_tile']

######################
# training the model #
######################
print("start training")
model = model1.instantiate_and_train()
print("training finished")


#####################
# running the model #
#####################
resulting_scores_1 = []
resulting_scores_2 = []
resulting_scores_3 = []
for i in range(100):
    score, highest_tile = model1.play_game(model)
    
    resulting_scores_1.append(score)
    print(f"Game {i+1}/100 - Score: {score} - Highest Tile: {highest_tile}")
    
    score, highest_tile = randommodel.play_game(randommodel.RandomAI())
    resulting_scores_2.append(score)

    one_dir_AI = one_direction_model.SimplePriorityAI()
    score, highest_tile = one_direction_model.play_game(one_dir_AI)
    resulting_scores_3.append(score)

#########################
# printing statistics #
#########################
# model 1
print(f"Average Score over 100 games for simple agent: {sum(resulting_scores_1)/len(resulting_scores_1)}")
print(f"The standard deviation over 100 games: {np.std(resulting_scores_1)}")

# model 2
print(f"Average Score over 100 games for random agent: {sum(resulting_scores_2)/len(resulting_scores_2)}")
print(f"The standard deviation over 100 games: {np.std(resulting_scores_2)}")
# model 3
print(f"Average Score over 100 games for one-direction agent: {sum(resulting_scores_3)/len(resulting_scores_3)}")
print(f"The standard deviation over 100 games: {np.std(resulting_scores_3)}")
