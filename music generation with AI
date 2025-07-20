import glob
import numpy as np
import pickle
import random
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation
from tensorflow.keras.utils import to_categorical

# Step 1: Load and Parse MIDI Files
def load_midi_files(folder):
    notes = []
    for file in glob.glob(folder + "/*.mid"):
        print("Parsing:", file)
        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)
        elements = parts.parts[0].recurse() if parts else midi.flat.notes

        for element in elements:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

# Step 2: Prepare Sequences for LSTM
def prepare_sequences(notes, sequence_length=100):
    pitchnames = sorted(set(notes))
    note_to_int = {note: i for i, note in enumerate(pitchnames)}
    int_to_note = {i: note for note, i in note_to_int.items()}

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[note] for note in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_vocab = len(pitchnames)

    network_input = np.reshape(network_input, (len(network_input), sequence_length, 1)) / float(n_vocab)
    network_output = to_categorical(network_output)

    return network_input, network_output, note_to_int, int_to_note, n_vocab

# Step 3: Build the LSTM Model
def build_model(input_shape, n_vocab):
    model = Sequential()
    model.add(LSTM(512, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Step 4: Generate Music
def generate_music(model, network_input, int_to_note, n_vocab, output_file="output/generated_song.mid", length=500):
    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]
    pattern = pattern.tolist()

    prediction_output = []

    for _ in range(length):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append([index])
        pattern = pattern[1:]

    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if '.' in pattern:
            notes_in_chord = pattern.split('.')
            chord_notes = [note.Note(int(n)) for n in notes_in_chord]
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.instrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)
    print(f"✅ Music generated and saved to {output_file}")

# === MAIN PIPELINE === #
if __name__ == "__main__":
    notes = load_midi_files("midi_songs")

    with open("data/notes.pkl", "wb") as f:
        pickle.dump(notes, f)

    network_input, network_output, note_to_int, int_to_note, n_vocab = prepare_sequences(notes)

    model = build_model((network_input.shape[1], network_input.shape[2]), n_vocab)
    model.fit(network_input, network_output, epochs=50, batch_size=64)
    model.save("music_model.h5")

    generate_music(model, network_input, int_to_note, n_vocab)
