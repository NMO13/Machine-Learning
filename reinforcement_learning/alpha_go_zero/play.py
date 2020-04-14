from connect4_proxy import Connect4Proxy
from connect4_net_wrapper import NNetWrapper
import numpy as np
from MCTS import MCTS
from utils import dotdict

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('/media/nmo/3E26969E2696572D/Martin/Programmieren/Machine-Learning/reinforcement_learning/alpha_go_zero/temp','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__ == "__main__":
    game = Connect4Proxy(6, 7)
    net = NNetWrapper(game)
    net.load_checkpoint(folder='./temp/', filename='best.pth.tar')
    board = game.getInitBoard()

    nmcts = MCTS(game, net, args)
    curPlayer = 1
    it = 0

    def computer_move(canonical_board, valids):
        action = np.argmax(nmcts.getActionProb(canonical_board, temp=0))
        if valids[action] == 0:
            assert valids[action] > 0
        return action

    def human_move(canonical_board, valids):
        while True:
            # break if we make a legal move
            move = input("Enter column: ")
            if valids[int(move)] == 1:
                break
        return int(move)

    players = [computer_move , None, human_move]
    while game.getGameEnded(board, curPlayer)==0:
        it += 1
        valids = game.getValidMoves(game.getCanonicalForm(board, curPlayer), None)
        action = players[curPlayer + 1](game.getCanonicalForm(board, curPlayer), valids)
        board, curPlayer = game.getNextState(board, curPlayer, action)
        game.draw_board(board)




