import argparse
import os

def get_parsed_files():
    parser = argparse.ArgumentParser(description='Read in a file or set of files, and return the result.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', nargs='+', help='Path of a file or a folder of files.')
    args = parser.parse_args()
    full_paths = [os.path.join(os.getcwd(), path) for path in args.path]
    files = []
    names = []
    for path in full_paths:
        if os.path.isfile(path):
            files.append(path)
            name_with_ext = os.path.basename(path)
            pure_name = os.path.splitext(name_with_ext)[0]
            names.append(pure_name)
    return files, names