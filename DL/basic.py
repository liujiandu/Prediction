import os
from DL.LSTM import VanillaLSTM

class PredictMethod(object):
    '''
    simple factory class for traj prediction methods
    '''
    @staticmethod
    def __init__(self, args):
        if args['Architecture'] == 'VanillaLSTM':
            self.method = VanillaLSTM(args)
        elif args['Architecture'] == 'GAN':
            pass
