import numpy as np
import matplotlib.pyplot as plt

introduction = 'Welcome to Paper/Stone/Scissors game based on HMM'
instructions = 'Input:\tp - paper, st - stone, sc - scissors, q - stop, s - stats'

winComb = ['sc', 'p', 'st']
statesDict = {'q': -1, 'p': 0, 'st': 1, 'sc': 2, 's': 3}
winrate = {'win': 0, 'loss': 0, 'tie': 0}
payout = 0
prob = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
toGraph = [payout]


def printStats():
    p = [getProbabilities(prob[0]), getProbabilities(prob[1]), getProbabilities(prob[2])]
    print(f'=======================STATS==========================\nMatrix:')
    print(np.matrix(p))
    print(f'Winrate: {winrate}\n'
          f'Payout: {payout}')


def getInput():
    tmp = str(input()).lower()
    while tmp not in statesDict.keys():
        print("Wrong input")
        tmp = str(input()).lower()
    if tmp == 'q' or tmp == 'Q':
        x = np.linspace(0, len(toGraph))
        y = np.linspace(min(toGraph) - 1, max(toGraph) + 1)
        X, Y = np.meshgrid(x, y)
        tmpY = Y.copy()
        tmpY[(tmpY > 0)] = 1
        tmpY[(tmpY < 0)] = -1
        Z = tmpY
        cs = plt.contourf(X, Y, Z, [-1, 0, 1], colors=('b', 'r'), alpha=0.3)
        plt.colorbar(cs)
        plt.plot(range(len(toGraph)), toGraph, 'b--')
        plt.plot(range(len(toGraph)), toGraph, 'bo')
        plt.xlabel('games count')
        plt.ylabel('payout')
        plt.show()
        exit(0)
    if tmp == 's':
        printStats()
        return getInput()
    return tmp

def getProbabilities(val):
    tmp = sum(val)
    return [val[0]/tmp, val[1]/tmp, val[2]/tmp]


def getPayout(pl,ai):
    print('=================',pl,ai,winComb[statesDict[ai]])
    if ai == winComb[statesDict[pl]]:
        winrate["loss"] = winrate["loss"] + 1
        return -1
    if pl == winComb[statesDict[ai]]:
        winrate["win"] = winrate["win"] + 1
        return 1
    winrate["tie"] = winrate["tie"] + 1
    return 0

def payoutToString(p):
    if p == 1:
        return 'You won'
    if p == -1:
        return 'You lost'
    if p == 0:
        return 'Tie'


print(introduction)
print(instructions)

inp = getInput()
prevInp = 'H'

while inp in statesDict.keys():
    if not prevInp == 'H':
        prob[statesDict[prevInp]][statesDict[inp]] += 1
        generated = np.random.choice(winComb, p=getProbabilities(prob[statesDict[prevInp]]))
    else:
        generated = np.random.choice(winComb, p=getProbabilities(prob[statesDict[inp]]))

    val = getPayout(inp, generated)
    payout += val
    toGraph.append(payout)
    print(f'Opponents move: {generated} \n\t{payoutToString(val)}')
    print(instructions)
    prevInp = inp
    inp = getInput()

