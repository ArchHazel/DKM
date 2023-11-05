import pickle
from pathlib import Path

megadepth_indices_train = Path("data/Megadepth/train-data/megadepth_indices/prep_scene_info")
sceneInfoFile = list(megadepth_indices_train.iterdir())
sceneInfoFile = [x.name for x in sceneInfoFile]

file_num = len(sceneInfoFile) // 2

with open('data/megadepth_indices_part1.pkl', 'wb') as f:
    pickle.dump(sceneInfoFile[:file_num], f)


with open('data/megadepth_indices_part2.pkl', 'wb') as f:
    pickle.dump(sceneInfoFile[file_num:], f)