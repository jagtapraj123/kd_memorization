# Ref: https://github.com/cydonia999/VGGFace2-pytorch/blob/master/utils.py

import pickle
import torch


def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, "rb") as f:
        weights = pickle.load(f, encoding="latin1")

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                if name == "fc.weight" or "fc.bias":
                    print(
                        "Exception in load_state_dict: Ignoring weights of {}".format(
                            name
                        )
                    )
                    continue

                raise RuntimeError(
                    "While copying the parameter named {}, whose dimensions in the model are {} and whose "
                    "dimensions in the checkpoint are {}.".format(
                        name, own_state[name].shape, param.shape
                    )
                )
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))
