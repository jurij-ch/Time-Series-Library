import os
import tqdm
import pandas as pd
from matlab_data import ZoneMatData

# Parameters
DATA_PATH = os.path.join("dataset", "01_raw", "bacteria", "Zones")
CLASSES = [r"Antib", r"Inhibition_Zone", r"Out_of_Zone"]
TRAIN_RATIO = 0.9
TS_LENGTH = 1080    # 1080 = 6 Hours. Use -1 for the whole period
PREFIX = f"bact_{TS_LENGTH}_"


if __name__ == "__main__":
    x_train = []
    x_test = []
    lbl_train = []
    lbl_test = []
    for folder in tqdm.tqdm(CLASSES):
        data = ZoneMatData(root_path=os.path.join(DATA_PATH, folder))
        x_df, lbl_df = data.to_df()
        if TS_LENGTH > -1:
            x_df = x_df.iloc[:TS_LENGTH, :]
        n_train = int(x_df.shape[1] * TRAIN_RATIO)
        x_train.append(x_df.iloc[:, :n_train])
        x_test.append(x_df.iloc[:, n_train:])
        lbl_train.append(lbl_df.iloc[:n_train])
        lbl_test.append(lbl_df.iloc[n_train:])

    x_train_df = pd.concat(x_train, axis=1, ignore_index=True)
    x_test_df = pd.concat(x_test, axis=1, ignore_index=True)
    lbl_train_df = pd.concat(lbl_train, axis=0, ignore_index=True)
    lbl_test_df = pd.concat(lbl_test, axis=0, ignore_index=True)

    x_train_df.to_csv(os.path.join(DATA_PATH, f"{PREFIX}features_TRAIN.csv"), index=False)
    x_test_df.to_csv(os.path.join(DATA_PATH, f"{PREFIX}features_TEST.csv"), index=False)
    lbl_train_df.to_csv(os.path.join(DATA_PATH, f"{PREFIX}labels_TRAIN.csv"), index=False)
    lbl_test_df.to_csv(os.path.join(DATA_PATH, f"{PREFIX}labels_TEST.csv"), index=False)

    print(f"Exported to {DATA_PATH}")
