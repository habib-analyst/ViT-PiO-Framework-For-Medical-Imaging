#!/usr/bin/env python3
import os
import re
import sys

def replace_in_file(path, pattern, replacement):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            data = f.read()
    except Exception:
        return False
    new_data, n = re.subn(pattern, replacement, data, flags=re.IGNORECASE)
    if n > 0:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_data)
        return True
    return False

def rename_path(old_path):
    dirname = os.path.dirname(old_path)
    basename = os.path.basename(old_path)
    if 'habib' in basename.lower():
        new_basename = re.sub('(?i)habib', 'habib', basename)
        new_path = os.path.join(dirname, new_basename)
        try:
            os.rename(old_path, new_path)
            return new_path
        except Exception:
            return old_path
    return old_path

def main(root):
    pattern = r'habib'
    replacement = 'habib'

    # First, replace inside files
    text_exts = ('.py', '.txt', '.csv', '.md', '.json', '.xml', '.html', '.htm', '.yaml', '.yml', '.ini', '.cfg', '.log')
    modified_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            # attempt to treat common text files; also try others (ignore binary errors)
            try:
                changed = replace_in_file(fpath, pattern, replacement)
                if changed:
                    modified_files.append(fpath)
            except Exception:
                continue

    # Then, rename files and directories that contain 'habib' in their name
    # Rename files first
    renamed = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in list(filenames):
            fpath = os.path.join(dirpath, fname)
            if 'habib' in fname.lower():
                newpath = rename_path(fpath)
                if newpath != fpath:
                    renamed.append((fpath, newpath))

    # Rename directories bottom-up to avoid issues
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for dname in list(dirnames):
            if 'habib' in dname.lower():
                old_d = os.path.join(dirpath, dname)
                new_d = re.sub('(?i)habib', 'habib', old_d)
                try:
                    os.rename(old_d, new_d)
                    renamed.append((old_d, new_d))
                except Exception:
                    continue

    print('Modified files:')
    for p in modified_files:
        print(' -', p)
    print('Renamed paths:')
    for a,b in renamed:
        print(' -', a, '->', b)

if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else r'd:\\FYP-Files'
    main(root)
