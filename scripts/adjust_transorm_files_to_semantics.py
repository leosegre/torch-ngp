import sys
import shutil
import os
import json

def main(argv):
    dir_name = argv[0]
    train_file_path = os.path.join(dir_name, 'transforms_train.json')
    test_file_path = os.path.join(dir_name, 'transforms_test.json')
    val_file_path = os.path.join(dir_name, 'transforms_val.json')
    backup_file_path = os.path.join(dir_name, 'transforms_backup')
    os.mkdir(backup_file_path)
    shutil.copy(train_file_path, backup_file_path)
    shutil.copy(test_file_path, backup_file_path)
    shutil.copy(val_file_path, backup_file_path)

    with open(os.path.join(train_file_path), 'r') as f:
        train = json.load(f)
    with open(os.path.join(test_file_path), 'r') as f:
        test = json.load(f)
    with open(os.path.join(val_file_path), 'r') as f:
        val = json.load(f)

    new_train = train
    new_train['frames'] = train['frames'] + test['frames']

    for frame in new_train['frames']:
        semantic_path = frame['file_path'].replace("Images", "DeepLab_seg")
        frame['semantic_path'] = semantic_path

    print(f"[INFO] writing {len(new_train['frames'])} frames to {train_file_path}")
    with open(train_file_path, "w") as outfile:
        json.dump(new_train, outfile, indent=2)

    new_test = train
    for frame in new_test['frames']:
        semantic_path = frame['file_path'].replace("Images", "Labels")
        frame['semantic_path'] = semantic_path

    print(f"[INFO] writing {len(new_test['frames'])} frames to {test_file_path}")
    with open(test_file_path, "w") as outfile:
        json.dump(new_test, outfile, indent=2)

    for frame in val['frames']:
        semantic_path = frame['file_path'].replace("Images", "DeepLab_seg")
        frame['semantic_path'] = semantic_path

    print(f"[INFO] writing {len(val['frames'])} frames to {val_file_path}")
    with open(val_file_path, "w") as outfile:
        json.dump(val, outfile, indent=2)

if __name__ == "__main__":
   main(sys.argv[1:])