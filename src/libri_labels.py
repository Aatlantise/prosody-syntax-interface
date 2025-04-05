import pdb
import os
from tqdm import tqdm

"""
This code snippet adds syntactic tag lines to LibriTTSLabel line data (one word or syllable per line).
To access code that adds syntactic tags (e.g. <NP>, </VP>) to data, see src/syntax.py.
"""

def unit_test():
    word_dir = "/home/jm3743/data/LibriTTSLabelNP/lab/word/"
    text_dir = "/home/jm3743/data/LibriTTSNP/"
    iterator = "dev-clean/1272/128104/1272_128104_000001_000000"
    text_file = text_dir + iterator + ".normalized.txt"
    word_file = word_dir + iterator + ".lab"

    text = open(text_file).read()
    word = open(word_file).read()


def add_tokens(running_text, sample_line_input):
    """
    Add <NP>, </NP>, <VP>, </VP> lines to LibriTTSLabel data.

    Requires annotated LibriTTS .normalized.txt data

    """
    running_tokens = running_text.split(" ")
    lines_with_tags = []
    lines = sample_line_input.split('\n')

    n_lines = len(lines)
    n_special_tags = 0
    n_blanks = 0
    n_words = 0

    line = 0
    for token in running_tokens:
        if token in ["<NP>", "<VP>"]:
            n_special_tags += 1

            # iterate until we see the first word of the NP as there may be empty duration lines
            while line < len(lines) and lines[line].split('\t')[-1] == "":
                n_blanks += 1
                lines_with_tags.append(lines[line])
                line += 1

            b, e, word = lines[line].split("\t")
            lines_with_tags.append("\t".join([b, b, token]))


        elif token in ["</NP>", "</VP>"]:
            n_special_tags += 1
            lines_with_tags.append("\t".join([e, e, token]))

        else:
            while line < len(lines) and lines[line].split('\t')[-1] == "":
                n_blanks += 1
                lines_with_tags.append(lines[line])
                line += 1

            n_words += 1
            b, e, word = lines[line].split("\t")

            lines_with_tags.append(lines[line])
            line += 1
    for l in lines[line:]:
        lines_with_tags.append(l)
        if l.split('\t')[-1] == "":
            n_blanks += 1
    return "\n".join(lines_with_tags), n_words + n_blanks == n_lines


def main():

    word_read_dir = "/home/jm3743/data/LibriTTSLabel/lab/word/"
    word_write_dir = "/home/jm3743/data/LibriTTSLabelNPVP/lab/word/"
    text_dir = "/home/jm3743/data/LibriTTSNPVP/"

    good = 0
    bad = 0

    for split in [
        'dev-clean',
        'test-clean',
        # 'train-clean-100'
    ]:
        for book in tqdm(os.listdir(os.path.join(word_read_dir, split))):
            for chapter in os.listdir(os.path.join(word_read_dir, split, book)):
                for sent in os.listdir(os.path.join(word_read_dir, split, book, chapter)):

                    word = open(os.path.join(word_read_dir, split, book, chapter, sent)).read()
                    text = open(os.path.join(text_dir, split, book, chapter, sent)[:-4] + ".normalized.txt").read()
                    try:
                        lines_with_tags, checker = add_tokens(text, word)
                        if checker:
                            good += 1
                            with open(os.path.join(word_write_dir, split, book, chapter, sent), "w") as f:
                                f.write(lines_with_tags)
                        else:
                            bad += 1
                    except:
                        bad += 1

    print(good)
    print(bad)
