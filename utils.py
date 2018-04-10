"""
Contains all helpers for DRCN
"""
import json
import os


def prepare_dir(args):
    # customize the output path
    if args.source_only:
        out_path = os.path.join(args.out, 'source_only')
    else:
        out_path = os.path.join(args.out, args.noise)
    try:
        os.makedirs(out_path)
    except OSError:
        pass
    # save all options for the experiment
    args_path = os.path.join(out_path, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f)
    return out_path
