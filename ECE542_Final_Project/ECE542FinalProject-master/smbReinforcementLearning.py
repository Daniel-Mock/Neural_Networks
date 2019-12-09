from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from copy import deepcopy
import time
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, [['right'], ['left'], ['up'], ['NOOP']])
env.reset()

# Batch_Size = 5000
# Epochs = 500


'''action_meanings = env.get_action_meanings()
step_array = []
for step in range(5000):
    action_arr = []
    for action in range(len(action_meanings)):
        test_env = deepcopy(env)
        state, reward, done, info = test_env.step(action)
        if reward != 0:
            print(action_meanings[action])
        step_tuple = (action, reward)
        index = 0
        for i in range(len(action_arr)):
            if action_arr[i][1] < step_tuple[1]:
                index = i
                break
        action_arr.insert(index, step_tuple)
    state, reward, done, info = env.step(action_arr[0][0])
    if done:
        print(action_arr)
        print(info)
        break
'''

