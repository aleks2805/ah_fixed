import os
import warnings
import pickle
import numpy as np
from config import * from music21 import *
from tqdm import trange
from copy import deepcopy
from model import build_model
from samplings import gamma_sampling
from loader import get_filenames, convert_files
# Обновленный современный импорт для Keras/TensorFlow
from tensorflow.keras.utils import to_categorical

# use cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings("ignore")

# Load chord types
with open(CHORD_TYPES_PATH, "rb") as filepath:
    chord_types = pickle.load(filepath)

def generate_chord(chord_model, melody_data, beat_data, key_data, segment_length=SEGMENT_LENGTH, rhythm_gamma=RHYTHM_DENSITY, chord_per_bar=CHORD_PER_BAR):

    chord_data = []

    # Process each melody sequence in the corpus
    for idx, song_melody in enumerate(melody_data):

        # Load the corresponding beat sequence
        song_melody = segment_length*[0] + song_melody + segment_length*[0]
        song_beat = segment_length*[0] + beat_data[idx] + segment_length*[0]
        song_key = segment_length*[0] + key_data[idx] + segment_length*[0]
        song_chord  = segment_length*[0]
        
        # Predict each pair
        for idx in range(segment_length, len(song_melody)-segment_length):
            
            # Create input data
            melody_left = song_melody[idx-segment_length:idx]
            melody_right = song_melody[idx:idx+segment_length][::-1]
            beat_left = song_beat[idx-segment_length:idx]
            beat_right = song_beat[idx:idx+segment_length][::-1]
            key_left = song_key[idx-segment_length:idx]
            key_right = song_key[idx:idx+segment_length][::-1]
            chord_left = song_chord[idx-segment_length:idx]
            
            # One-hot vectorization
            melody_left = to_categorical(melody_left, num_classes=128)
            melody_right = to_categorical(melody_right, num_classes=128)
            beat_left = to_categorical(beat_left, num_classes=5)
            beat_right = to_categorical(beat_right, num_classes=5)
            key_left = to_categorical(key_left, num_classes=16)
            key_right = to_categorical(key_right, num_classes=16)
            condition_left = np.concatenate((beat_left, key_left), axis=-1)
            condition_right = np.concatenate((beat_right, key_right), axis=-1)
            chord_left = to_categorical(chord_left, num_classes=len(chord_types))

            # expand dimension
            melody_left = np.expand_dims(melody_left, axis=0)
            melody_right = np.expand_dims(melody_right, axis=0)
            condition_left = np.expand_dims(condition_left, axis=0)
            condition_right = np.expand_dims(condition_right, axis=0)
            chord_left = np.expand_dims(chord_left, axis=0)
            
            # Predict the next chord
            prediction = chord_model.predict(x=[melody_left, melody_right, condition_left, condition_right, chord_left], verbose=0)[0]

            if song_melody[idx]!=0 and song_beat[idx]==4:
                prediction = gamma_sampling(prediction, [[0]], [1], return_probs=True)

            # Tuning rhythm density
            if chord_per_bar:
                if song_beat[idx]==4 and (song_melody[idx]!=song_melody[idx-1] or song_beat[idx]!=song_beat[idx-1]) and not (idx==segment_length and song_melody[idx]==0):
                    prediction = gamma_sampling(prediction, [[song_chord[-1]]], [1], return_probs=True)
                
                else:
                    prediction = gamma_sampling(prediction, [[song_chord[-1]]], [0], return_probs=True)

            else:
                prediction = gamma_sampling(prediction, [[song_chord[-1]]], [rhythm_gamma], return_probs=True)

            cho_idx = np.argmax(prediction, axis=-1)
            song_chord.append(cho_idx)
        
        # Remove the leading padding 
        chord_data.append(song_chord[segment_length:])

    return chord_data


def watermark(score, filename, water_mark=WATER_MARK):

    # Add water mark
    if water_mark:
        score.metadata = metadata.Metadata()
        score.metadata.title = filename
        score.metadata.composer = 'harmonized by AutoHarmonizer'
    
    return score


def export_music(score, beat_data, chord_data, filename, repeat_chord=REPEAT_CHORD, outputs_path=OUTPUTS_PATH, water_mark=WATER_MARK):

    filename = os.path.basename(filename)
    filename = '.'.join(filename.split('.')[:-1])

    # 1. Выделяем дорожку Мелодии (очищаем от старых символов аккордов, если они были)
    if isinstance(score, stream.Score):
        melody_part = deepcopy(score.parts[0]) if score.parts else deepcopy(score)
    else:
        melody_part = deepcopy(score)
    
    melody_part.id = 'Melody'
    for el in list(melody_part.recurse()):
        if isinstance(el, harmony.ChordSymbol):
            melody_part.remove(el, recurse=True)

    # 2. Создаем отдельную дорожку для Аккордов
    chord_part = stream.Part()
    chord_part.id = 'Chords'

    for idx, song_chord in enumerate(chord_data):
        song_chord = [chord_types[int(cho_idx)] for cho_idx in song_chord]
        song_beat = beat_data[idx]
        
        chord_events = []  # Хранит кортежи вида (индекс_шага, название_аккорда)
        pre_chord = None
        
        for t_idx, cho in enumerate(song_chord):
            cho = cho.replace('N.C.', 'R')
            cho = cho.replace('bpedal', '-pedal')
            
            # Проверяем, изменился ли аккорд или нужно ли его повторить
            if t_idx == 0 or cho != pre_chord or (repeat_chord and t_idx != 0 and song_beat[t_idx] == 4 and song_beat[t_idx-1] != 4):
                chord_events.append((t_idx, cho))
                pre_chord = cho

        # Превращаем зафиксированные гармонические события в нотные объекты с длительностями
        total_steps = len(song_chord)
        for i, (start_step, cho) in enumerate(chord_events):
            end_step = chord_events[i+1][0] if i + 1 < len(chord_events) else total_steps
            duration_quarters = (end_step - start_step) * 0.25  # Каждый шаг = 0.25 четвертной ноты
            
            if cho == 'R':
                chord_obj = note.Rest(quarterLength=duration_quarters)
            else:
                # Извлекаем реальные музыкальные ноты из буквенного обозначения для MIDI-трека
                cs = harmony.ChordSymbol(cho)
                chord_obj = chord.Chord(cs.pitches, quarterLength=duration_quarters)
            
            chord_part.insert(start_step * 0.25, chord_obj)

    # ИСПРАВЛЕНИЕ ОШИБКИ: Автоматически генерируем тактовую сетку для дорожки аккордов
    chord_part = chord_part.makeMeasures()
    chord_part.id = 'Chords'

    # 3. Объединяем обе дорожки (Мелодия и Аккорды) в один многотрековый Score
    output_score = stream.Score()
    output_score.insert(0, melody_part)
    output_score.insert(0, chord_part)

    if water_mark:
        output_score = watermark(output_score, filename)
        
    # Экспортируем в формат MIDI (.mid)
    os.makedirs(outputs_path, exist_ok=True)
    midi_file_path = os.path.join(outputs_path, filename + '.mid')
    output_score.write('midi', fp=midi_file_path)
    print(f"\nSaved 2-track MIDI to: {midi_file_path}")


if __name__ == "__main__":

    # Load data from 'inputs'
    filenames = get_filenames(input_dir=INPUTS_PATH)
    data_corpus = convert_files(filenames, fromDataset=False)

    # Build harmonic rhythm and chord model
    model = build_model(SEGMENT_LENGTH, RNN_SIZE, NUM_LAYERS, DROPOUT, WEIGHTS_PATH, training=False)
    
    # Process each melody sequence
    for idx in trange(len(data_corpus)):
        
        melody_data = data_corpus[idx][0]
        beat_data = data_corpus[idx][1]
        key_data = data_corpus[idx][2]
        score = data_corpus[idx][3]
        filename = data_corpus[idx][4]

        # Generate harmonic rhythm and chord data
        chord_data = generate_chord(model, melody_data, beat_data, key_data)
        
        # Export music file
        export_music(score, beat_data, chord_data, filename)
