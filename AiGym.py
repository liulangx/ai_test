"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
# from gym.utils import seeding
import numpy as np
import random

class AiGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # 每走一步获得的分数,分别代表四周无墙，1面墙，2面墙，3面墙，4面墙的情况
        self.stepScore = [1, 5, 10, 15, 20]
        # 每砍敌人一刀的分数,分别代表砍一刀，两刀，三刀和四刀
        self.hurtCount = 0;
        self.hurtScore = [30, 35, 40, 45];
        # 捡到宝物的分数,分别代表当前已经有刀(盾)和无刀(盾)的奖励
        self.swordScore = [0, 20];
        self.shieldScore = [0, 15];
        # stop条件,该条件应该和安全区范围有关，一种是满足该分数条件，然后开始寻找一个有墙保护的位置，>=一面墙就可以停止了
        self.stopThresholdBase = 72
        self.stopThreshold = 72 * 12 / 12;

        self.tau = 0.02  # seconds between state updates

        # 安全区大小，我方血量，我方剑，我方盾，敌方血量，敌方剑，敌方盾，我方位置（2位），敌方位置（2位）
        high = np.array([
            13,
            500,
            2,
            2,
            500,
            2,
            2,
            12,
            12,
            12,
            12
        ])

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(-high, high)

        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

        self.selfDefinedMap = None
        self.meBlood = None
        self.meHasShield = None
        self.meHasSword = None
        self.mePos = None
        self.enemyBlood = None
        self.enemyHasShield = None
        self.enemyHasSword = None
        self.enemyPos = None
        self.inDrugArea = None
        self.intoDrugArea = None
        self.scores = None

    # 改变我方位置
    def mePosChange(self, pos, action):
        # action: 0左，1左下，2下，3右下，4右，5右上，6上，7左上
        done = False
        if action == 0:
            pos[1] -= 1
            if pos[1] < 0 or pos[1] > 11:
                done = True;
        elif action == 1:
            pos[0] += 1
            pos[1] -= 1
            if pos[1] < 0 or pos[1] > 11 or pos[0] < 0 or pos[0] > 11:
                done = True;
        elif action == 2:
            pos[0] += 1
            if  pos[0] < 0 or pos[0] > 11:
                done = True
        elif action == 3:
            pos[0] += 1
            pos[1] += 1
            if pos[1] < 0 or pos[1] > 11 or pos[0] < 0 or pos[0] > 11:
                done = True;
        elif action == 4:
            pos[1] += 1
            if pos[1] < 0 or pos[1] > 11:
                done = True;
        elif action == 5:
            pos[0] -= 1
            pos[1] += 1
            if pos[1] < 0 or pos[1] > 11 or pos[0] < 0 or pos[0] > 11:
                done = True;
        elif action == 6:
            pos[0] -= 1
            if pos[0] < 0 or pos[0] > 11:
                done = True
        elif action == 7:
            pos[0] -= 1
            pos[1] -= 1
            if pos[1] < 0 or pos[1] > 11 or pos[0] < 0 or pos[0] > 11:
                done = True;
        if done is not True:
            if self.selfDefinedMap[pos[0]][pos[1]] != 0 and self.selfDefinedMap[pos[0]][pos[1]] != 1 and self.selfDefinedMap[pos[0]][pos[1]] != 2:
                done = True
        if done is not True:
            self.selfDefinedMap[pos[0]][pos[1]] = 5
        return pos, done

    # 获取周围是否有敌人，有返回True，无返回False
    def getAroundEnemyState(self, pos):
        isAroundEnemy = False;
        left = np.array([pos[0], pos[1] - 1]);
        if left[1] >= 0 and left[1] <= 11:
            if self.selfDefinedMap[left[0]][left[1]] == 4:
                isAroundEnemy = True
        right = np.array([pos[0], pos[1] + 1]);
        if right[1] >= 0 and right[1] <= 11:
            if self.selfDefinedMap[right[0]][right[1]] == 4:
                isAroundEnemy = True
        top = np.array([pos[0] - 1, pos[1]]);
        if top[0] >= 0 and top[0] <= 11:
            if self.selfDefinedMap[top[0]][top[1]] == 4:
                isAroundEnemy = True
        bottom = np.array([pos[0] + 1, pos[1]]);
        if bottom[0] >= 0 and bottom[0] <= 11:
            if self.selfDefinedMap[bottom[0]][bottom[1]] == 4:
                isAroundEnemy = True
        return isAroundEnemy;

    # 返回周围墙的数量
    def getCountAroundWall(self, pos):
        count = 0
        left = np.array([pos[0], pos[1] - 1]);
        if left[1] >= 0 and left[1] <= 11:
            if self.selfDefinedMap[left[0]][left[1]] == 3:
                count += 1
        right = np.array([pos[0], pos[1] + 1]);
        if right[1] >= 0 and right[1] <= 11:
            if self.selfDefinedMap[right[0]][right[1]] == 3:
                count += 1
        top = np.array([pos[0] - 1, pos[1]]);
        if top[0] >= 0 and top[0] <= 11:
            if self.selfDefinedMap[top[0]][top[1]] == 3:
                count += 1
        bottom = np.array([pos[0] + 1, pos[1]]);
        if bottom[0] >= 0 and bottom[0] <= 11:
            if self.selfDefinedMap[bottom[0]][bottom[1]] == 3:
                count += 1
        return count;

    def step(self, action):
        # action: 0左，1左下，2下，3右下，4右，5右上，6上，7左上
        # 判断是否是正确的动作
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        map = self.selfDefinedMap
        mePos = self.mePos
        done = False
        reward = 0.0
        # 如果我方位置变化成功，则更新自己的状态self.mePos
        mePos, done = self.mePosChange(mePos, action)
        if done:
            return state, reward, done, self.inDrugArea, self.intoDrugArea, {}
        else:
            self.mePos = mePos
        # 判断是否在毒圈内
        if self.isInDrugArea(mePos):
            self.intoDrugArea = True
            self.inDrugArea = True
        else:
            self.inDrugArea = False
        # 判断四周是否有英雄：
        isAroundEnemy = self.getAroundEnemyState(mePos)
        if isAroundEnemy:
            print(self.hurtCount)
            reward += self.hurtScore[self.hurtCount]
            self.hurtCount += 1
        # 判断四周有几面墙：
        countAroundWall = self.getCountAroundWall(mePos)
        reward += self.stepScore[countAroundWall]
        # 判断是否有宝物
        if map[mePos[0]][mePos[1]] == 1 and self.meHasSword == 0:
            reward += self.swordScore
            self.meHasSword = 1;
        if map[mePos[0]][mePos[1]] == 2 and self.meHasShield == 0:
            reward += self.shieldScore
            self.meHasShield = 1;
        # 获取当前总分
        self.scores += reward;
        # 判断是否需要停止
        if self.scores > self.stopThreshold:
            if countAroundWall > 0:
                done = True
        self.state = np.array(
            [self.safeColRow, self.meHasSword, self.meHasShield, self.mePos[0], self.mePos[1], self.enemyPos[0], self.enemyPos[1]])
        return np.array(self.state), reward, done, self.inDrugArea, self.intoDrugArea, {}

    def isInDrugArea(self, pos):
        # 安全区的起始数，包括该行（列）
        startSafeNum = 0 + (12 - self.safeColRow) / 2
        # 安全区的截止数，包括该行（列）
        endSafeNum = 11 - (12 - self.safeColRow) / 2
        if (pos[0] >= startSafeNum and pos[0] <= endSafeNum) and (
                pos[1] >= startSafeNum and pos[1] <= endSafeNum):
            inDrugArea = False;
        else:
            inDrugArea = True;

        return inDrugArea

    def reset(self):
        self.hurtCount = 0
        self.scores = 0
        self.createNewMap()
        self.createPos()
        # 固定地图暂时不需要生成该信息
        # self.createTreasure()
        # 生成安全区大小2的倍数 <=12
        self.safeColRow = random.randrange(1, 7) * 2
        totalBlood = 200
        # 随机生成己方血量和敌方血量
        self.meBlood = random.randrange(1, totalBlood + 1)
        self.enemyBlood = random.randrange(1, totalBlood + 1)
        # 随机生成是否有剑或者盾，1代表有，0代表无
        self.meHasSword = random.randrange(0, 2)
        self.enemyHasSword = random.randrange(0, 2)
        self.meHasShield = random.randrange(0, 2)
        self.enemyHasShield = random.randrange(0, 2)
        # 安全区大小，（我方血量），我方剑，我方盾，（敌方血量，敌方剑，敌方盾），我方位置（2位），敌方位置（2位）
        self.state = np.array([self.safeColRow, self.meHasSword, self.meHasShield, self.mePos[0], self.mePos[1],
             self.enemyPos[0], self.enemyPos[1]])
        # self.state = np.array([self.safeColRow, self.meBlood, self.meHasSword, self.meHasShield, self.enemyBlood, self.enemyHasSword,
                               #self.enemyHasShield, self.mePos[0][0], self.mePos[0][1], self.enemyPos[0][0], self.enemyPos[0][1]])
        self.steps_beyond_done = None
        # ***毒圈相关***
        # 确定停止条件
        self.stopThreshold = self.stopThresholdBase * self.safeColRow / 12
        # 判断是否在毒圈内部：
        isInDrugArea = self.isInDrugArea(self.mePos)
        if isInDrugArea:
            self.inDrugArea = True
            self.intoDrugArea = True
        else:
            self.inDrugArea = False
            self.intoDrugArea = False
        return self.state

    def createTreasure(self):
        self.hasTreasure = random.randrange(0, 1)
        if self.hasTreasure == 1:
            self.swordPos = [random.randrange(0, 12), random.randrange(0, 12)]
            count = 0
            while self.selfDefinedMap[self.swordPos[0]][self.swordPos[1]] != 0:
                self.swordPos = [random.randrange(0, 12), random.randrange(0, 12)]
                count += 1
                if count % 1000 == 0:
                    print('放置剑找不到空地')
            self.shieldPos = [random.randrange(0, 12), random.randrange(0, 12)]
            count = 0
            while self.selfDefinedMap[self.shieldPos[0]][self.shieldPos[1]] != 0:
                self.swordPos = [random.randrange(0, 12), random.randrange(0, 12)]
                count += 1
                if count % 1000 == 0:
                    print('放置剑找不到空地')
            self.selfDefinedMap[self.swordPos[0]][self.swordPos[1]] = 1
            self.selfDefinedMap[self.shieldPos[0]][self.shieldPos[1]] = 2

    def createPos(self):
        # 生成敌方位置
        # self.enemyPos = np.zeros((1, 2))
        # self.enemyPos = [random.randrange(0, 12), random.randrange(0, 12)]
        # count = 0
        # while self.selfDefinedMap[self.mePos[0]][self.mePos[1]] != 0:
        #     self.mePos = [random.randrange(0, 12), random.randrange(0, 12)]
        #     count += 1
        #     if count % 1000 == 0:
        #         print('敌方找不到空地')

        self.enemyPos = [1, 10]
        self.selfDefinedMap[1][10] = 4
        # print('生成敌方位置：', self.enemyPos)
        # 生成我方位置
        # self.mePos = np.zeros((1, 2))
        # self.mePos = [random.randrange(0, 12), random.randrange(0, 12)]
        # count = 0
        # while self.selfDefinedMap[self.mePos[0]][self.mePos[1]] != 0:
        #     self.mePos = [random.randrange(0, 12), random.randrange(0, 12)]
        #     count += 1
        #     if count % 1000 == 0:
        #         print('我方找不到空地')
        self.mePos = [11, 0]
        self.selfDefinedMap[11][0] = 5
        # print('生成我方位置：', self.mePos)

    def createNewMap(self):
        # 0 空地，1 剑， 2 盾，3 墙， 4 敌人 5 已走过路径
        self.selfDefinedMap = np.zeros((12, 12), dtype=np.int)
        self.selfDefinedMap[0][5] = 2
        # self.selfDefinedMap[1][10] = 4
        self.selfDefinedMap[2][1] = 3
        self.selfDefinedMap[2][2] = 3
        self.selfDefinedMap[3][1] = 3
        self.selfDefinedMap[5][7] = 3
        self.selfDefinedMap[5][8] = 3
        self.selfDefinedMap[6][7] = 3
        self.selfDefinedMap[9][9] = 3
        self.selfDefinedMap[10][8] = 3
        self.selfDefinedMap[11][9] = 3
        self.selfDefinedMap[10][10] = 3
        self.selfDefinedMap[10][9] = 1

    def render(self, mode='human'):
        return None

    def close(self):
        return None
