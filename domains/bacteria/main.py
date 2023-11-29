import os
from matlab_data import ZoneMatData


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_path = r"dataset/01_raw/bacteria/Zones"
    antibacterial = ZoneMatData(root_path=os.path.join(data_path, r"Antib"))
    print(antibacterial.filenames)
    print(antibacterial.T)
    antibacterial.plot_frame_sigx(time_ind=0, sample_name=antibacterial.filenames[0])
    antibacterial.plot_ts_sigx(0, 3, 0, 3, antibacterial.filenames[0], with_mean=False)
