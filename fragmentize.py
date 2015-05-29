import io
import os

BLOCK_SIZE = 512
SOURCE_DIRS = ["data/html"]
TARGET_DIRS = ["data/fragmented_html"]

def chunks(string, size):
    return [ string[i:i + size] for i in range(0, len(string), size) ]
             

def fragmentize_all_files_in_dir(source_dir, target_dir):
    for source_file in os.listdir(source_dir):
        source_path = os.path.join(source_dir, source_file)     
        content = open(source_path).read()
        name, extension = os.path.splitext(source_file)

        # Drop last chunk. I only want blocks of the same size.
        for index, chunk in enumerate( chunks(content, BLOCK_SIZE)[:-1] ):
           target_path = os.path.join(target_dir, name + "_" + str(index) + extension)
           with open(target_path, "wb") as target_file:
               target_file.write(chunk)

for source_dir in SOURCE_DIRS:
    # Put fragments from data/log into data/fragmented_log
    rest, base_name = os.path.dirname(source_dir), os.path.basename(source_dir)
    target_dir = os.path.join(rest, "fragmented_" + base_name)

    fragmentize_all_files_in_dir(source_dir, target_dir)
