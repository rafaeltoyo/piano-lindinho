import numpy as np
import cv2

from pianovision.KeyData import KeyData


class Keyboard:

    def __init__(self, cropped_image: object, black_key_xarray: object, key_shift: object,
                 key_height: object) -> object:
        self.key_shift = key_shift
        self.image = cropped_image

        # Making a mask for the key numbers
        self.key_mask = np.zeros(cropped_image[:, :, 0].shape)
        key_index = key_shift

        # Font cuz unicorns exist
        font = cv2.FONT_HERSHEY_SIMPLEX

        number_lines = len(black_key_xarray)
        self.lines = black_key_xarray

        mask_image = cropped_image.copy()

        white_key_lines = []
        print(number_lines)
        for i in range(number_lines):
            if (key_index == 0 or key_index == 2 or key_index == 4 or key_index == 6 or key_index == 8 or key_index == 3
                    or key_index == 9):
                if (i < number_lines - 1):
                    midway_point = int(self.lines[i] + (self.lines[i + 1] - self.lines[i]) / 2)
                    white_key_lines.append(midway_point)
            key_index = (key_index + 1) % 10

        last_x = 0
        self.keys = []
        key_number = 0
        white_notes = ["C", "D", "E", "F", "G", "A", "B"]

        # Calculating note index
        d_notes = [0, 1]
        e_notes = [2, 3]
        g_notes = [4, 5]
        a_notes = [6, 7]
        b_notes = [8, 9]
        key_index = None
        if key_shift in d_notes:
            key_index = 1
        if key_shift in e_notes:
            key_index = 2
        if key_shift in g_notes:
            key_index = 4
        if key_shift in a_notes:
            key_index = 5
        if key_shift in b_notes:
            key_index = 6

        for line in white_key_lines:
            self.key_mask = cv2.rectangle(self.key_mask, (last_x, 0), (line, self.key_mask.shape[0]),
                                          key_number, cv2.FILLED)
            mask_image = cv2.rectangle(mask_image, (last_x, 0), (line, mask_image.shape[0]),
                                       key_number, cv2.FILLED)
            last_x = line

            note = white_notes[key_index]
            found_key = KeyData(key_number, note)
            self.keys.append(found_key)
            key_number = key_number + 1;
            key_index = (key_index + 1) % len(white_notes)

        black_notes = ["C#", "D#", "F#", "G#", "A#"]
        key_index = key_shift
        for i in range(len(self.lines)):
            if (key_index == 0 or key_index == 2 or key_index == 4 or key_index == 6 or key_index == 8):
                if (i < len(self.lines) - 1):
                    self.key_mask = cv2.rectangle(self.key_mask, (self.lines[i], 0), (self.lines[i + 1],
                                                                                      key_height), key_number,
                                                  cv2.FILLED)
                    mask_image = cv2.rectangle(mask_image, (self.lines[i], 0), (self.lines[i + 1],
                                                                                key_height), key_number,
                                               cv2.FILLED)
                    note = black_notes[int(key_index / 2)]

                    found_key = KeyData(key_number, note)
                    self.keys.append(found_key)

                    key_number = key_number + 1
            key_index = (key_index + 1) % 10

        # Assuming the fourth octave in the middle of the image, populates the notes to the right of it

        middle_point = int(cropped_image.shape[1] / 2)
        octave_number = 4
        x = middle_point
        cur_number = self.key_mask[0, x]
        x_number = self.key_mask[0, x]
        for key in self.keys:
            if (key.id == x_number):
                x_key = key
                break
        cur_key = x_key


        while (x < cropped_image.shape[1] - 1):
            # check what the x_key is
            x_number = self.key_mask[0, x]
            for key in self.keys:
                if (key.id == x_number):
                    x_key = key
                    break

            # if x_key is different, check if its a C
            # if not C, then simply put the octave number on note
            # if it is a C, then increment the octave number and put the octave number on the note

            if (x_key != cur_key):
                if (x_key.note != "C"):
                    x_key.note = x_key.note + str(octave_number)
                else:
                    octave_number = octave_number + 1
                    x_key.note = x_key.note + str(octave_number)
                if (x_key.note.find('#') != -1):
                    cv2.putText(mask_image, x_key.note, (x, 50), font, 0.4, (0, 0, 200), 1, cv2.LINE_AA)
                else:
                    cv2.putText(mask_image, x_key.note, (x, 200), font, 0.4, (0, 0, 200), 1, cv2.LINE_AA)
            cur_key = x_key
            x = x + 2
        # Doing the same, but to the left
        middle_point = int(cropped_image.shape[1] / 2)
        octave_number = 4
        x = middle_point - 1
        x_number = self.key_mask[0, x]
        for key in self.keys:
            if key.id == x_number:
                x_key = key
                break
        cur_key = x_key

        while (x >= 0):
            # check what the x_key is
            x_number = self.key_mask[0, x]
            x_key = self.keys[int(x_number)]

            # if x_key is different, check if its a C
            # if not C, then simply put the octave number on note
            # if it is a C, then increment the octave number and put the octave number on the note

            if (x_key != cur_key and not (x_key.note[len(x_key.note) - 1].isdigit())):
                if x_key.note != "C":
                    x_key.note = x_key.note + str(octave_number)
                else:
                    x_key.note = x_key.note + str(octave_number)
                    octave_number = octave_number - 1
                if x_key.note.find('#') != -1:
                    cv2.putText(mask_image, x_key.note, (x - 25, 50), font, 0.4, (0, 0, 200), 1, cv2.LINE_AA)
                else:
                    cv2.putText(mask_image, x_key.note, (x - 25, 200), font, 0.4, (0, 0, 200), 1, cv2.LINE_AA)
            cur_key = x_key
            x = x - 1

        cv2.imshow('Mask created', mask_image)
        cv2.waitKey(0)

    def get_key_mask(self):
        return (self.key_mask)
