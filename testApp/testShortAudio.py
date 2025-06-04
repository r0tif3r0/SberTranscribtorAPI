import os
from tqdm import tqdm

TEST_DATA_SHORT_PATH = '../testAudioShort'

def test_transcriber(model):
    num_tests = 0
    sum_similarity = 0
    for subdir in os.listdir(TEST_DATA_SHORT_PATH):
        subdir_path = os.path.join(TEST_DATA_SHORT_PATH, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for filename in tqdm(os.listdir(subdir_path)):
            if not filename.endswith('.wav'):
                continue
            wav_path = os.path.join(subdir_path, filename)
            txt_path = os.path.join(subdir_path, filename[:-4] + '.txt')
            transcription = model.transcribe(wav_path)
            with open(txt_path, 'r', encoding='utf-8') as txt_file:
                expected_transcription = txt_file.read().strip()
            sum_similarity += similarity(transcription, expected_transcription)
            num_tests += 1
    accuracy = sum_similarity / num_tests
    print(f"Accuracy: {accuracy * 100}%")
    print(f"Number of tests: {num_tests}")

def similarity(a, b):
    if len(a) < len(b):
        return similarity(b, a)

    if len(b) == 0:
        return len(a)

    previous_row = range(len(b) + 1)
    for i, c1 in enumerate(a):
        current_row = [i + 1]
        for j, c2 in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return (len(a) - previous_row[-1]) / len(a)
