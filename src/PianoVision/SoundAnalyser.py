class SoundAnalyser:
    def __init__(self, audio_data, video_frame_rate):
        self.audio_data = audio_data
        self.frame_rate = video_frame_rate

        #Calculating onset envelope and peak times

