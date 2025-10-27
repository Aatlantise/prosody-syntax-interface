import os
import glob
import json
from tqdm import tqdm
import stanza
import re
from transformers import GPT2TokenizerFast, T5Tokenizer

class TokenizerBuilder:
    def __init__(self, model_name, scratch=True, filename="/home/jm3743/prosody-syntax-interface/data/constituency_corpus.json"):
        self.model_name = model_name
        self.scratch = scratch
        self.filename = filename

        # POS tags (PTB set) and brackets as regular tokens
        self.pos_tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS",
                         "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$",
                         "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG",
                         "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "#", "$", ".", ","]

        # True special tokens
        self.special_tokens = {
            "pad_token": "<pad>",
            "bos_token": "<bos>",
            "eos_token": "<eos>"
        }

        # Collect POS and phrase labels from corpus
        all_pos = set()
        all_phrases = set()
        with open(self.filename) as f:
            for line in f:
                data = json.loads(line)
                parse_line = data["parse"]
                pos, phrases = self.extract_labels_from_linearized(parse_line)
                all_pos.update(pos)
                all_phrases.update(phrases)

        print(f"Found {len(all_pos)} POS tags: {sorted(all_pos)}")
        print(f"Found {len(all_phrases)} phrase labels: {sorted(all_phrases)}")

        # Final token list (regular tokens include parentheses)
        self.all_tokens = sorted(all_pos) + sorted(all_phrases) + ["(", ")"]

        # Build the tokenizer
        self.tokenizer = self.build_tokenizer()

    def extract_labels_from_linearized(self, line):
        tokens = re.findall(r'\(|\)|[^\s()]+', line)
        pos_set = set()
        phrase_set = set()
        for t in tokens:
            if t in ("(", ")"):
                continue
            elif t.isupper():  # heuristic: all PTB labels uppercase
                if t in self.pos_tags:
                    pos_set.add(t)
                else:
                    phrase_set.add(t)
        return pos_set, phrase_set

    def build_tokenizer(self):
        if 'gpt' in self.model_name:
            tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
        elif 't5' in self.model_name:
                tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        else:
            raise ValueError(f"{self.model_name} is not supported.")

        # Add regular tokens (POS, phrase labels, parentheses)
        tokenizer.add_tokens(self.all_tokens)

        # Add special tokens
        tokenizer.add_special_tokens(self.special_tokens)

        # Save tokenizer
        tokenizer_name = f"tokenizers/{self.model_name}_from_scratch" if self.scratch else f"tokenizers/{self.model_name}"
        tokenizer.save_pretrained(tokenizer_name)

        return tokenizer



class CorpusBuilder:
    def __init__(self):
        # --- CONFIG ---
        self.root_dir = os.path.expanduser('~/data/LibriTTS')
        self.splits = ['train-clean-100', 'dev-clean', 'test-clean']
        self.output_path = os.path.expanduser('data/constituency_corpus.json')
        self.batch_size = 64  # tune based on GPU memory and sentence length

        # --- SETUP ---
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        self.nlp = stanza.Pipeline(
            lang='en',
            processors='tokenize,pos,constituency',
            tokenize_no_ssplit=True,   # keep single-sentence mode per text
            use_gpu=True
        )

    # --- HELPER ---
    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def strip_words(self, tree):
        """
        Recursively remove lexical items from a Stanza constituency tree,
        keeping only POS tags and brackets.
        """
        # Check if this is a preterminal: one child, which is a leaf
        if len(tree.children) == 1 and not tree.children[0].children:
            # Preterminal: return just the POS tag
            return f"{tree.label}"

        # Internal node: recurse over children
        children_str = " ".join(self.strip_words(c) for c in tree.children)
        return f"({tree.label} {children_str})"

    def __call__(self):
        with open(self.output_path, 'w', encoding='utf-8') as out_f:
            for split in self.splits:
                files = glob.glob(os.path.join(self.root_dir, split, '*', '*', '*.original.txt'))
                print(f"Found {len(files)} sentence files in {split}.")

                for batch_files in tqdm(list(self.chunks(files, self.batch_size)), desc=f"Parsing {split}"):
                    texts = []
                    metadata = []

                    for file_path in batch_files:
                        try:
                            text = open(file_path, encoding='utf-8').read().strip()
                            if text:
                                texts.append(text)
                                metadata.append({
                                    "split": split,
                                    "file": file_path,
                                    "text": text
                                })
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")

                    if not texts:
                        continue

                    try:
                        # Batch process
                        docs = self.nlp("\n\n".join(texts))  # Stanza treats blank lines as sentence breaks
                        all_linearized = []
                        all_tagsonly = []
                        for doc_idx, doc in enumerate(docs.sentences):
                            tree = doc.constituency
                            linearized = " ".join(str(tree).split())
                            tags_only = " ".join(self.strip_words(tree).split())
                            all_linearized.append(linearized)
                            all_tagsonly.append(tags_only)

                        # Safety: ensure alignment between metadata and outputs
                        # Stanza sometimes merges/splits sentences, so guard with zip
                        for meta, lin, tags in zip(metadata, all_linearized, all_tagsonly):
                            meta["full_parse"] = lin
                            meta["parse"] = tags
                            out_f.write(json.dumps(meta) + "\n")

                    except Exception as e:
                        print(f"Error parsing batch: {e}")

if __name__ == '__main__':
    corpus = CorpusBuilder()
    corpus()