import librosa
import numpy as np
from matplotlib import pyplot as plt


class SoundAnalyser:
    def __init__(self, audio_data, sample_rate):
        #Loading audio data
        self.audio_data = audio_data
        self.sample_rate = sample_rate


        #Calculating onset envelope and peak times
        self.onset_envelope = librosa.onset.onset_strength(self.audio_data, sr=self.sample_rate)
        #TODO - making peak_pick parameters related to tempo and beat
        self.peaks = librosa.util.peak_pick(self.onset_envelope,
                                            pre_max = 3, post_max = 3, pre_avg = 10,
                                            post_avg=10, delta=0.12,wait=0)

        self.times = librosa.frames_to_time(np.arange(len(self.onset_envelope)),
                                            sr = self.sample_rate, hop_length = 512)

        self.peak_times = self.times[self.peaks]
        #Plotting the onset peak times in a graph
        plt.plot(self.times, self.onset_envelope, alpha = 0.8, label='Onset strength')
        plt.vlines(self.peak_times, 0, self.onset_envelope.max(),
                   color='r', alpha = 0.5, label = 'Detected Notes')
        plt.legend(frameon=True, framealpha=0.8)
        plt.axis('tight')
        plt.xlim(0, 150)
        plt.tight_layout()
        plt.show()

    def peak_frames(self, video_frame_rate):
        peak_frames = self.peak_times*video_frame_rate
        return(peak_frames)

