import os
from tqdm import tqdm

def add_tokens(running_text, sample_line_input):
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
            # pdb.set_trace()

            # iterate until we see the first word of the NP as there may be empty duration lines
            while line < len(lines) and lines[line].split('\t')[-1] == "":
                n_blanks += 1
                # pdb.set_trace()
                # print(lines[line])
                lines_with_tags.append(lines[line])
                line += 1

            # pdb.set_trace()
            b, e, word = lines[line].split("\t")
            # print("\t".join([b, b, token]))
            lines_with_tags.append("\t".join([b, b, token]))


        elif token in ["</NP>", "</VP>"]:
            n_special_tags += 1
            # pdb.set_trace()
            # print("\t".join([e, e, token]))
            lines_with_tags.append("\t".join([e, e, token]))

        else:
            # pdb.set_trace()
            while line < len(lines) and lines[line].split('\t')[-1] == "":
                n_blanks += 1
                # pdb.set_trace()
                # print(lines[line])
                lines_with_tags.append(lines[line])
                line += 1

            n_words += 1
            b, e, word = lines[line].split("\t")

            # print(lines[line])
            lines_with_tags.append(lines[line])
            line += 1
    for l in lines[line:]:
        # print(l)
        lines_with_tags.append(l)
        if l.split('\t')[-1] == "":
            n_blanks += 1
    return "\n".join(lines_with_tags), n_words + n_blanks == n_lines,


word_read_dir = "/home/jm3743/data/LibriTTSLabel/lab/word/"
word_write_dir = "/home/jm3743/data/LibriTTSLabelNP/lab/word/"
text_dir = "/home/jm3743/data/LibriTTSNP/"
# word_write_dir = "/home/jm3743/data/LibriTTSLabelNPVP/lab/word/"
# text_dir = "/home/jm3743/data/LibriTTSNPVP/"

good = 0
bad = 0

for split in [
    'dev-clean',
              'test-clean',
              'train-clean-100'
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

                if checker:
                    good += 1
                else:
                    bad += 1


print(good)
print(bad)
