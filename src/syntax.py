import stanza
from ngram.ngram import annotate_phrases
import os
from tqdm import tqdm

"""
This code snippet adds syntactic tags (e.g. <NP>, </VP>) to data.
To access code that adds syntactic tags (generated here) to line data (one word or syllable per line), see libri_labels.py.
"""


def process_content(norm, syn, model):
    """
    Use ngram.ngram.annotate_phrases to annotate syntactic tags.

    :param norm: Normalized.txt filename
    :param syn: Syntactic.txt filename
    :param model: stanza model
    """
    analysis = model(open(norm).read())
    annotated_output = [annotate_phrases(s.constituency) for s in analysis.sentences]
    with open(syn, "w") as f:
        f.write(' '.join(annotated_output))



# Get all .origianl.txt files in the directory
def main():
    for split in [
        # "dev-clean",
        # "test-clean",
        "train-clean-100"]:
        books = os.listdir(os.path.join(data_dir, split))
        for book in tqdm(books):
            chapters = os.listdir(os.path.join(data_dir, split, book))
            for chapter in chapters:
                original_files = [t for t in os.listdir(os.path.join(data_dir, split, book, chapter)) if t.endswith(".original.txt")]
                for original_file in original_files:
                    # Construct the corresponding .syntactic.txt filename
                    base_name = original_file.replace(".original.txt", "")
                    syntactic_file = f"{base_name}.syntactic.txt"
                    norm = os.path.join(data_dir, split, book, chapter, original_file)
                    syn = os.path.join(data_dir, split, book, chapter, syntactic_file)
                    process_content(norm, syn, nlp)

    print("Syntactic files generated successfully.")

if __name__ == "__main__":
    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,pos,constituency",
        tokenize_pretokenized=True
    )

    data_dir = "/home/jm3743/data/LibriTTSNPVP"
    main()
