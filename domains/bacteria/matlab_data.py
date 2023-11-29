import os
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

__version__ = "1"

class Labels:
    Label_dct = {
        "Antib": 0,
        "Inhibition_Zone": 1,
        "Out_of_Zone": 2,
    }


class MatLabData:

    def __init__(self, root_path: str):
        self.__version__ = "0"
        self.root_path = root_path
        self.set_name = ""
        self.X = 0
        self.Y = 0
        self.T = 0

    def _plot_frame(self, data_frame, title: str):
        plt.figure(figsize=(6, 6))
        plt.imshow(data_frame, cmap='gray', interpolation='none')
        plt.colorbar()
        plt.title(f'{self.set_name} - {title}')
        plt.axis('off')
        plt.show()

    def _plot_frame_ts(self, ts, time_ind: int, ts_name: str):
        if len(ts.shape) == 2:
            frame_vector = ts[:, time_ind]
            frame = frame_vector.reshape(self.Y, self.X)
        else:
            frame = ts[:, :, time_ind]
        self._plot_frame(frame, f"{ts_name} at time index {time_ind}")


class MatData(MatLabData):

    def __init__(self, root_path: str, filenames: list):
        super().__init__(root_path)
        self.__version__ = "34"
        self.filenames = filenames
        self.set_name = "merged: " + " ".join(self.filenames)
        self.mat_data_dct = {}

        for filename in self.filenames:
            filepath = os.path.join(self.root_path, filename)
            self.mat_data_dct[filename] = scipy.io.loadmat(filepath)
            if self.T == 0:
                self.matav = self.mat_data_dct[filename]["MatAv"]
                self.imgsig = self.mat_data_dct[filename]["ImgSig"]
                self.matsig = self.mat_data_dct[filename]["MatSig"]
                # self.imgav = self.mat_data_dct[fname]["ImgAv"]
                self.Y = self.matav.shape[0]
                self.X = self.matav.shape[1]
                self.T = self.mat_data_dct[filename]["tind"].shape[1]
                self.FILE_T = self.mat_data_dct[filename]["tind"].shape[1]
            else:
                self.matav = np.concatenate(
                    (self.matav, self.mat_data_dct[filename]["MatAv"]),
                    axis=1)
                self.imgsig = np.concatenate(
                    (self.imgsig, self.mat_data_dct[filename]["ImgSig"]),
                    axis=1)
                self.matsig = np.concatenate(
                    (self.matsig, self.mat_data_dct[filename]["MatSig"]),
                    axis=1)
                #                 self.imgav = np.concatenate(
                #                     (self.imgav, self.mat_data_dct[fname]["ImgAv"]),
                #                     axis=1)
                self.T += self.mat_data_dct[filename]["tind"].shape[1]

    def plot_frame_matav(self):
        self._plot_frame(data_frame=self.matav, title="MatAv")

    def plot_frame_matsig(self):
        self._plot_frame(data_frame=self.matsig, title="MatSig")

    def plot_frame_imgsig(self, time_ind):
        self._plot_frame_ts(ts=self.imgsig, time_ind=time_ind)

    def _animate(self, ts):
        def _update_plot(frame_number, z_array, plot_, ax_):
            plot_[0].remove()
            plot_[0] = ax_.imshow(z_array[:, :, frame_number], cmap="gray")

        data_3d = ts.reshape((self.Y, self.X, self.T))

        # Set up the initial plot
        fig, ax = plt.subplots(figsize=(5, 5))
        plot = [ax.imshow(data_3d[:, :, 0], cmap="gray")]  # "viridis"

        # Create the animation
        frames = data_3d.shape[2]  # The third dimension size is the number of frames
        ani = FuncAnimation(fig, _update_plot, frames=range(frames),
                            fargs=(data_3d, plot, ax), blit=False, interval=200)

        # Display the animation
        HTML(ani.to_html5_video())

    def animate_imgsig(self):
        self._animate(ts=self.imgsig)

    def _plot_timeseries(self, ts, x1, x2, y1, y2, title: str):
        frame_vector = ts[:, 0].copy()  # we are going draw a frame by modifying the data
        frame = frame_vector.reshape(self.Y, self.X)
        frame[y1:y2, (x1, x2)] = frame.max()
        frame[(y1, y2), x1:x2] = frame.max()

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

        axs[0].imshow(frame, cmap='gray', interpolation='none')
        axs[0].set_title('Frame')

        xset = list(range(x1, x2+1))
        yset = list(range(y1, y2+1))

        frame_crop = np.ndarray((0, self.T))

        # for y, x in zip(yset, xset):
        for y in yset:
            frame_crop = np.concatenate(
                (frame_crop, ts[y * self.X + x1:y * self.X + x2, :]), axis=0
            )
            for x in xset:
                point_id = y * self.X + x
                axs[1].plot(ts[point_id, :], label=f'Line {y=},{x=}')
        mean_line = np.mean(frame_crop, axis=0)
        axs[1].plot(mean_line, color="purple", linewidth=4, linestyle="--")

        axs[1].set_title(f'{self.set_name} - {title}')
        plt.tight_layout()
        # plt.xlabel('Column Index')
        # plt.ylabel('title')
        # plt.legend()
        # plt.grid(True)
        plt.show()


    def plot_ts_imgsig(self, x1, x2, y1, y2):
        self._plot_timeseries(self.imgsig, x1, x2, y1, y2, "ImgSig")

    # def plot_ts_imgav(self, x1, x2, y1, y2):
    #     self._plot_timeseries(self.imgav, x1, x2, y1, y2, "ImgAv")


class ZoneMatData(MatLabData):

    def __init__(self, root_path: str, end_time_ind=-1):
        super().__init__(root_path)
        self.__version__ = "1"
        self.filenames = self.find_mat_files()
        self.set_name = os.path.basename(self.root_path)
        self.end_time_ind = end_time_ind
        self.mat_data_dct = {}
        self.sigx_dct = {}

        self.load_data()

    def find_mat_files(self):
        mat_files = []
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.endswith(".mat"):
                    mat_files.append(file)
        return mat_files

    def load_data(self):
        for filename in self.filenames:
            filepath = os.path.join(self.root_path, filename)
            self.mat_data_dct[filename] = scipy.io.loadmat(filepath)
            if self.T == 0:
                if self.end_time_ind > -1:
                    self.T = self.end_time_ind
                else:
                    self.T = self.mat_data_dct[filename]["sigX"].shape[2]
                self.Y = self.mat_data_dct[filename]["sigX"].shape[0]
                self.X = self.mat_data_dct[filename]["sigX"].shape[1]
            else:
                assert self.X == self.mat_data_dct[filename]["sigX"].shape[1]
                assert self.Y == self.mat_data_dct[filename]["sigX"].shape[0]
            self.sigx_dct[filename] = self.mat_data_dct[filename]["sigX"][:, :, : self.T]

    def plot_frame_sigx(self, time_ind, sample_name: str):
        self._plot_frame_ts(ts=self.sigx_dct[sample_name], time_ind=time_ind, ts_name="sigX")

    def _plot_timeseries(self, ts, x1, x2, y1, y2, title: str, whith_mean: bool):
        frame = ts[y1:y2+1, x1:x2+1, 0].copy()

        fig, axs = plt.subplots(1, 2, figsize=(14, 4))  # 1 row, 2 columns

        axs[0].imshow(frame, cmap='gray', interpolation='none')
        axs[0].set_title('Frame')
        for y in range(y1, y2+1):
            for x in range(x1, x2+1):
                axs[1].plot(ts[y, x, :], label=f'Line {y=},{x=}')
        if whith_mean:
            reshaped_array = ts.reshape(-1, ts.shape[2])
            mean_line = np.mean(reshaped_array, axis=0)
            axs[1].plot(mean_line, color="purple", linewidth=4, linestyle="--")

        axs[1].set_title(f'{self.set_name} - {title}')
        plt.tight_layout()
        plt.show()

    def plot_ts_sigx(self, x1, x2, y1, y2, sample_name: str, with_mean:bool = True):
        self._plot_timeseries(self.sigx_dct[sample_name], x1, x2, y1, y2, "sigX", with_mean)

    def to_df(self):
        dfs = []
        for filename in self.filenames:
            arr = self.sigx_dct[filename].reshape(-1, self.sigx_dct[filename].shape[2]).T
            dfs.append(pd.DataFrame(arr))
        df_features = pd.concat(dfs, axis=1)
        df_labels = pd.DataFrame({'label': [Labels.Label_dct[self.set_name]] * df_features.shape[1]})
        return df_features, df_labels
