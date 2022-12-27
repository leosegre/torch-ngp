import sys
import shutil
import os

def main(argv):
    for file in argv:
        shutil.move(file, file + "~")
        with open(file + "~", "r") as f:
            lines = f.readlines()
        with open(file, 'w') as f:
            for line in lines:
                f.write(line)
                if "file_path" in line:
                    semantic_path = line.replace("file_path", "semantic_path").replace("Images/rgba", "DeepLab_seg/segmentation")
                    f.write(semantic_path)
        os.remove(file + "~")

if __name__ == "__main__":
   main(sys.argv[1:])