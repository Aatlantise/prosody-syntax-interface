import os
import glob
import json
from tqdm import tqdm
import stanza
import re
from transformers import GPT2TokenizerFast, T5TokenizerFast
from constituency.data import load_data, extract_examples_from_sent


class ParseTokenizer:
    def __init__(self, vocab_list, special_tokens):

        self.special_tokens = special_tokens

        self.token2id = {}
        self.ids = []
        self.id2token = {}

        # 1. Add special tokens FIRST
        for name, tok in self.special_tokens.items():
            self.token2id[tok] = len(self.ids)
            self.ids.append(tok)

        # 2. Add regular tokens AFTER
        for tok in vocab_list:
            if tok not in self.token2id:
                self.token2id[tok] = len(self.ids)
                self.ids.append(tok)

        # expose HF-like attributes
        self.pad_token = self.special_tokens["pad_token"]
        self.bos_token = self.special_tokens["bos_token"]
        self.eos_token = self.special_tokens["eos_token"]
        self.unk_token = self.special_tokens["unk_token"]

        self.pad_token_id = self.token2id[self.pad_token]
        self.bos_token_id = self.token2id[self.bos_token]
        self.eos_token_id = self.token2id[self.eos_token]
        self.unk_token_id = self.token2id[self.unk_token]

        self.id2token = {i: tok for tok, i in self.token2id.items()}

    def tokenize(self, text):
        return re.findall(r'\(|\)|[^\s()]+', text)

    def encode(self, text):
        # no <unk> should exist, though
        return [self.token2id.get(tok, self.token2id["<unk>"]) for tok in self.tokenize(text)]

    def decode(self, ids):
        return " ".join(self.id2token[i] for i in ids)

    def batch_encode(self, texts):
        return {"input_ids": [self.encode(t) for t in texts]}

    def batch_decode(self, batch_ids):
        return [self.decode(ids) for ids in batch_ids]

    def pad(self, encoded_inputs, max_length=None, return_tensors=None, **kwargs):
        """
        HuggingFace-style pad() implementation.
        encoded_inputs: list of dicts with "input_ids" and optionally "attention_mask"
        """
        # List[dict] case → HF always passes a list from DataCollator
        if isinstance(encoded_inputs, list):
            # Determine max sequence length
            if max_length is None:
                max_length = max(len(feat["input_ids"]) for feat in encoded_inputs)

            padded_input_ids = []
            padded_attention_masks = []

            for feat in encoded_inputs:
                ids = feat["input_ids"]
                pad_len = max_length - len(ids)

                padded_input_ids.append(ids + [self.pad_token_id] * pad_len)
                padded_attention_masks.append([1] * len(ids) + [0] * pad_len)

            result = {
                "input_ids": padded_input_ids,
                "attention_mask": padded_attention_masks
            }

        else:
            raise ValueError("pad() expects a list of feature dicts.")

        # Convert to tensors
        if return_tensors == "pt":
            import torch
            result = {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}

        return result

    def save_pretrained(self, directory):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "vocab.json"), "w") as f:
            json.dump(self.token2id, f, indent=2)

    @classmethod
    def from_pretrained(cls, directory):
        with open(os.path.join(directory, "vocab.json")) as f:
            token2id = json.load(f)
        inv = {i: tok for tok, i in token2id.items()}
        special = {k: v for k, v in token2id.items()
                   if k in ["<pad>", "<bos>", "<eos>", "<unk>"]}
        tok = cls([], special)
        tok.token2id = token2id
        tok.id2token = inv
        return tok

    def __len__(self):
        return len(self.token2id)


class TokenizerBuilder:
    def __init__(self, model_name, filename="/home/jm3743/prosody-syntax-interface/data/constituency_corpus.json"):
        self.model_name = model_name
        self.filename = filename

        # POS tags (PTB set) and brackets as regular tokens
        self.pos_tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS",
                         "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$",
                         "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG",
                         "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "#", "$", ".", ",", "''"]

        # True special tokens
        self.special_tokens = {
            "pad_token": "<pad>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
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
        manual_addn = ' '.join(["(", ")", "#", "$", ".", ",", "''"])
        print(f"Adding manual additions: {manual_addn}")

        # Final token list (regular tokens include parentheses)
        self.all_tokens = sorted(all_pos) + sorted(all_phrases) + ["(", ")", "#", "$", ".", ",", "''"]

        # Build the tokenizer
        self.tokenizer = self.build_tokenizer()

    def extract_labels_from_linearized(self, line):
        tokens = re.findall(r'\(|\)|[^\s()]+', line)
        pos_set = set()
        phrase_set = set()
        for t in tokens:
            if t in ("(", ")", "#", "$", ".", ",", "''"):
                continue
            elif t.isupper():  # heuristic: all PTB labels uppercase
                if t in self.pos_tags:
                    pos_set.add(t)
                else:
                    phrase_set.add(t)
        return pos_set, phrase_set

    def build_tokenizer(self):
        if 'gpt' in self.model_name:
            tokenizer = ParseTokenizer(self.all_tokens, self.special_tokens)
        elif 't5' in self.model_name:
                tokenizer = T5TokenizerFast.from_pretrained(self.model_name)
                # Add regular tokens (POS, phrase labels, parentheses)
                tokenizer.add_tokens(self.all_tokens)

                # Add special tokens
                tokenizer.add_special_tokens(self.special_tokens)
        else:
            raise ValueError(f"{self.model_name} is not supported.")

        # Save tokenizer
        tokenizer_name = f"tokenizers/{self.model_name}"
        tokenizer.save_pretrained(tokenizer_name)

        return tokenizer



class CorpusBuilder:
    def __init__(self):
        # --- CONFIG ---
        self.root_dir = os.path.expanduser('/home/jm3743/data/LibriTTS')
        self.splits = ['train-clean-100', 'dev-clean', 'test-clean']
        self.output_path = os.path.expanduser('/home/jm3743/prosody-syntax-interface/data/constituency_corpus.json')
        self.output_with_prosody_path = os.path.expanduser('/home/jm3743/prosody-syntax-interface/data/constituency_corpus_prosody.json')
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

                                prosody_path = file_path.replace("LibriTTS/", "LibriTTSLabel/lab/word/").replace("original.txt", "lab")
                                prosody_df = load_data(prosody_path)
                                prosody_dict, _, _ = extract_examples_from_sent(prosody_df)
                                _1 = text
                                _2 = ' '.join([w['text'] for w in prosody_dict])
                                metadata.append({
                                    "split": split,
                                    "file": file_path,
                                    "text": text,
                                    "pause": [w['pause'] for w in prosody_dict],
                                    "duration": [w['duration'] for w in prosody_dict],
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


def tokenizer_test():
    t = TokenizerBuilder("gpt2")
    tokenizer = t.tokenizer
    tokens = tokenizer.encode("(ROOT (S (SBAR IN (S (NP PRP) (VP VBP RB (VP VB (PRT RP)))))"
                           " (NP PRP) (VP MD RB (VP VB (PRT RP))) , '' '' (VP VBZ (NP NNP) ,"
                           " (ADVP RB . CC (S (NP PRP) (VP VBZ (VP VBN (PRT RP)))))) .))")
    print([tokenizer.decode(k) for k in tokens["input_ids"]])

def corpus_test():
    corpus = CorpusBuilder()
    corpus()


def load_jsonl_data(path="/home/jm3743/prosody-syntax-interface/data/constituency_corpus.json", debug=False):
    items = []
    i = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "text" not in obj or "pause" not in obj or "duration" not in obj or "parse" not in obj:
                continue
            items.append({
                "text": obj["text"],
                "pause": obj["pause"],
                "duration": obj["duration"],
                "parse": obj["parse"],
            })
            i += 1
            if debug and i == 100:
                break
    return items


def preprocess(tokenizer, examples, max_source_length, max_target_length):
    inputs = [_.lower() for _ in examples["text"]]
    targets = examples["parse"]

    model_inputs = tokenizer(
        inputs,
        truncation=True,
        max_length=max_source_length,
        padding="max_length"
    )

    token_level_durations = []
    token_level_pauses = []
    for i in range(len(inputs)):
        input_ids = model_inputs["input_ids"][i]
        p = examples["pause"][i]
        d = examples["duration"][i]
        word_id = model_inputs.word_ids(i)
        _token_level_pause = []
        _token_level_duration = []
        punc = set()
        for j, w in enumerate(word_id):
            # jth sub-word token is part of kth word, including punctuation
            t = tokenizer.decode(input_ids[j])

            # skip punctuation marks, add to set of punctuation-words
            if not t.isalpha():
                punc.add(w)
                continue

            # kth word, discounting punctuation marks
            _token_level_pause.append(p[w - len(punc)])
            _token_level_duration.append(d[w - len(punc)])

        padding_zeros = [0.0] * (max_source_length - len(_token_level_pause))
        _token_level_duration += padding_zeros
        _token_level_pause += padding_zeros

        token_level_durations.append(_token_level_duration)
        token_level_pauses.append(_token_level_pause)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            truncation=True,
            max_length=max_target_length,
            padding="max_length"
        )

    # Replace pad token id in labels with -100 for loss masking
    label_pad_token_id = -100
    labels_ids = [
        [(x if x != tokenizer.pad_token_id else label_pad_token_id) for x in seq]
        for seq in labels["input_ids"]
    ]

    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "pause": token_level_pauses,
        "duration": token_level_durations,
        "labels": labels_ids,
    }

if __name__ == '__main__':
    corpus_test()
