import argparse
import torch.optim as optim

optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}

#code adapted from: https://github.com/declare-lab/MISA/blob/master/src/config.py
class Config(object):
    def __init__(self, **kwargs):
        '''Configuration Class: set kwargs as class attributes with setattr'''
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]

                setattr(self, key, value)

def get_config(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser("Center Loss Example")

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--lr-model', type=float, default=1e-3, help="learning rate for model")
    parser.add_argument('--lr-cent', type=float, default=1e-3, help="learning rate for center loss")
    parser.add_argument('--center', action='store_true', help='activate center loss')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--cent-weight', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=1.8)
    parser.add_argument('--run-type', type=str, default='train')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--batch-size', type=int, default=64)

    if parse:
        kwargs = parser.parse_args()

    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)


    return Config(**kwargs)










#
