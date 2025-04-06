import pdb
import os
import stanza
from ngram.ngram import annotate_phrases


def add_tokens(split, model):
    """
    Add <NP>, </NP>, <VP>, </VP> lines to LibriTTSLabel data. Modified from libri_labels.add_tokens

    Requires annotated LibriTTS .normalized.txt data

    """
    assert split in ['train_100', 'dev', 'test', 'train_360']

    helsinki_file_lines = open(f"/home/jm3743/data/helsinki-prosody/data/{split}.txt")

    print(f"Helsinki {split} file read.")

    f = open(f"/home/jm3743/data/helsinki-prosody-np/data/{split}.txt", 'w')
    g = open(f"/home/jm3743/data/helsinki-prosody-npvp/data/{split}.txt", 'w')
    local_lines = []

    print("Looping...")
    for line in helsinki_file_lines:
        if line.startswith('<file>'):
            if local_lines:
                # record

                local_sentence = ' '.join([line.split('\t')[0] for line in local_lines])
                analysis = model(local_sentence)
                annotated_output = annotate_phrases(analysis.sentences[0].constituency)

                i = 0
                for token in annotated_output.split():
                    if token in ["<NP>", "</NP>"]:
                        f.write("\t".join([token, "NA", "NA", "NA", "NA"]) + "\n")
                        g.write("\t".join([token, "NA", "NA", "NA", "NA"]) + "\n")
                    elif token in ["<VP>", "</VP>"]:
                        g.write("\t".join([token, "NA", "NA", "NA", "NA"]) + "\n")
                    else:
                        f.write(local_lines[i])
                        g.write(local_lines[i])
                        i += 1

                local_lines = []
            f.write(line)
            g.write(line)

        else:
            local_lines.append(line)


def main():
    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,pos,constituency",
        tokenize_pretokenized=True,
        tokenize_nossplit=True
    )

    for split in [
        # 'dev',
        # 'test',
        # 'train_100',
        'train_360'
    ]:
        add_tokens(split, nlp)

if __name__ == "__main__":
    main()