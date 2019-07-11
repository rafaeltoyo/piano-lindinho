#!/usr/bin/env python
# -*- coding: utf-8 -*-

import librosa
import numpy as np
from matplotlib import pyplot as plt


class SoundAnalyser:

    def __init__(self, filename: str):
        """
        SoundAnalyser
        :param filename: Audio filename
        """

        # Loading audio data
        audio_data, sample_rate = librosa.load(filename)
        self.audio_data: np.ndarray = audio_data
        self.sample_rate: int = sample_rate

        # Calculating onset envelope and peak times
        self.onset_envelope: np.ndarray = librosa.onset.onset_strength(self.audio_data, sr=self.sample_rate)

        # TODO - making peak_pick parameters related to tempo and beat

        self.peaks: np.ndarray = librosa.util.peak_pick(self.onset_envelope,
                                                        pre_max=1,
                                                        post_max=1,
                                                        pre_avg=3,
                                                        post_avg=3,
                                                        delta=0.1,
                                                        wait=0)

        self.times: np.ndarray = librosa.frames_to_time(np.arange(len(self.onset_envelope)),
                                                        sr=self.sample_rate,
                                                        hop_length=512)

        self.peak_times = self.times[self.peaks]

    def plot(self):
        """
        Plotting the onset peak times in a graph
        :return:
        """

        plt.plot(self.times, self.onset_envelope, alpha=0.8, label='Onset strength')
        plt.vlines(self.peak_times, 0, self.onset_envelope.max(), color='r', alpha=0.5, label='Detected Notes')
        plt.legend(frameon=True, framealpha=0.8)
        plt.axis('tight')
        plt.xlim(0, 150)
        plt.tight_layout()
        plt.show()

    def peak_frames(self, video_frame_rate: int) -> np.ndarray:
        """
        peak_frames
        :param video_frame_rate:
        :return:
        """

        return self.peak_times * video_frame_rate
