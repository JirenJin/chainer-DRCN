"""
Contains all helpers for DRCN
"""
import datetime
import json
import os


def prepare_dir(args):
    # customize the output path
    date = datetime.datetime.now()
    date_str = date.strftime("%m%d%H%M%S")
    if args.source_only:
        out_path = os.path.join(args.out, 'source_only', date_str)
    else:
        out_path = os.path.join(args.out, args.noise, date_str)
    try:
        os.makedirs(out_path)
    except OSError:
        pass
    # save all options for the experiment
    args_path = os.path.join(out_path, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f)
    return out_path
