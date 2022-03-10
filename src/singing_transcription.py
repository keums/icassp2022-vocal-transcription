# -*- coding: utf-8 -*-
# %%
import argparse
import numpy as np
from pathlib import Path
from model import *
from featureExtraction import *
from quantization import *
from utils import *
from MIDI import *

# %%
class SingingTranscription:
    def __init__(self):

        self.PATH_PROJECT = pathlib.Path(__file__).absolute().parent.parent
        self.num_spec = 513
        self.window_size = 31
        self.note_res = 1
        self.batch_size = 64

    def load_model(self, path_weight, TF_summary=False):

        model = melody_ResNet_JDC(self.num_spec, self.window_size, self.note_res)
        model.load_weights(path_weight)
        if TF_summary == True:
            print(model.summary())
        return model

    def predict_melody(self, model_ST, filepath):
        pitch_range = np.arange(40, 95 + 1.0 / self.note_res, 1.0 / self.note_res)
        pitch_range = np.concatenate([np.zeros(1), pitch_range])

        """  Features extraction"""
        X_test, _ = spec_extraction(file_name=filepath, win_size=self.window_size)

        """  melody predict"""
        y_predict = model_ST.predict(X_test, batch_size=self.batch_size, verbose=1)
        y_predict = y_predict[0]  # [0]:pitch,  [1]:vocal
        y_shape = y_predict.shape
        num_total = y_shape[0] * y_shape[1]
        y_predict = np.reshape(y_predict, (num_total, y_shape[2]))

        est_MIDI = np.zeros(num_total)
        est_freq = np.zeros(num_total)
        for i in range(num_total):
            index_predict = np.argmax(y_predict[i])
            pitch_MIDI = pitch_range[np.int32(index_predict)]
            if pitch_MIDI >= 40 and pitch_MIDI <= 95:
                est_MIDI[i] = pitch_MIDI
                # est_freq[i] = 2 ** ((pitch_MIDI - 69) / 12.0) * 440
        return est_MIDI

    def save_output_time_pitch_axis(self, pitch, path_save):
        check_and_make_dir(Path(path_save).parent)
        f = open(path_save, "w")
        for j in range(len(pitch)):
            est = "%.2f %.4f\n" % (0.01 * j, pitch[j])
            # est = "%.2f %.4f\n" % (0.01 * j, 2 ** ((pitch[j] - 69) / 12.0) * 440)
            f.write(est)
        f.close()


def main(args):
    ST = SingingTranscription()

    """ load model """
    model_ST = ST.load_model(f"{ST.PATH_PROJECT}/data/weight_ST.hdf5", TF_summary=False)

    """ predict note (time-freq) """
    filepath = args.path_audio
    note_midi = ST.predict_melody(model_ST, filepath)

    """ refine note """
    tempo = calc_tempo(filepath)
    refined_note = refine_note(note_midi, tempo)

    """ save results """
    filename = get_filename_wo_extension(filepath)
    path_note = f"{args.path_save}/{filename}.txt"
    ST.save_output_time_pitch_axis(refined_note, path_note)

    """ change note from (time-freq) to (midi) """
    PATH_est_midi = f"{args.path_save}/{filename}.mid"

    note2Midi(path_input_note=path_note, path_output=PATH_est_midi, tempo=tempo)

    print(f"DONE! Transcription: {filepath}")


# %%
if __name__ == "__main__":
    PATH_PROJECT = pathlib.Path(__file__).absolute().parent.parent
    parser = argparse.ArgumentParser(description="Predict singing transcription")
    parser.add_argument(
        "-i",
        "--path_audio",
        type=str,
        help="Path to input audio file.",
        default=f"{PATH_PROJECT}/audio/test.wav",
    )
    parser.add_argument(
        "-o",
        "--path_save",
        type=str,
        help="Path to folder for saving note&mid files",
        default=f"{PATH_PROJECT}/output",
    )

    main(parser.parse_args())
