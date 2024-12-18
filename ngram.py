import stanza
import json
from stanza.models.constituency.parse_tree import Tree
from tqdm import tqdm
from collections import defaultdict
import math


def annotate_phrases(tree):
    """
    Annotate phrases in the tree with nested <label> tags.

    Args:
        tree (Tree): A Stanza constituency parse Tree object.

    Returns:
        str: Annotated sentence with nested <label> tags.
    """
    def traverse(node):
        # If it's a leaf node, return the word
        if node.is_leaf():
            return node.label

        # If it's a target label (NP or VP), wrap in tags
        if node.label in {"NP", "VP"}:
            children_text = " ".join(traverse(child) for child in node.children)
            return f"<{node.label}> {children_text} </{node.label}>"
        else:
            # For other nodes, just process their children
            return " ".join(traverse(child) for child in node.children)

    return traverse(tree)


def file2json(filename):
    corpus = []
    sent = {
                "words": [],
                "labels": []
            }
    with open(filename) as f:
        for line in f:
            if "<file>" in line:
                if sent["words"]:
                    corpus.append(sent)
                    sent = {
                                "words": [],
                                "labels": []
                    }
                continue
            else:
                word, _, boundary, _, _ = line.split('\t')
                sent["words"].append(word)
                sent["labels"].append(boundary)

    return corpus


def preprocess_data(inputs, current_labels, special_tokens = ['<NP>', '</NP>', '<VP>', '</VP>']):
    """
    Preprocesses data by adding 'NA' labels for special tokens and aligning labels.

    Args:
        inputs (list of str): Input sequence of tokens (e.g., ['<NP>', 'A', ...]).
        current_labels (list of str): Current labels for lexical items only.
        special_tokens (set): Set of special tokens (e.g., {'<NP>', '</NP>'}).

    Returns:
        list of str: Aligned labels, with 'NA' for special tokens.
    """
    # Initialize result
    aligned_labels = []
    label_index = 0  # Pointer for current_labels

    for token in inputs:
        if token in special_tokens:
            aligned_labels.append("NA")  # Assign 'NA' for special tokens
        else:
            # Assign the next available label from current_labels
            aligned_labels.append(current_labels[label_index])
            label_index += 1

    # Ensure all labels are used
    assert label_index == len(current_labels), "Mismatch between input and labels."
    return aligned_labels


def prepare_splits(filename, _nlp):
    split_data = file2json(filename)

    sents = [sent["words"] for sent in split_data]
    split_doc = _nlp(sents)

    for const_sent, _sent in zip(split_doc.sentences, split_data):
        annotated_output = annotate_phrases(const_sent.constituency)
        _sent["anno"] = annotated_output
        # Move boundary label to next salient token

    aligned_labels_list = [
        preprocess_data(_sent["anno"].split(' '), _sent["labels"])
        for _sent in split_data
    ]

    input_list = [_sent["anno"].split(' ') for _sent in split_data]

    return input_list, aligned_labels_list


def prepare_dataset():
    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,pos,constituency",
        tokenize_pretokenized=True
    )

    test_input, test_labels = prepare_splits("data/prosody/data/test.txt", nlp)
    dev_input, dev_labels = prepare_splits("data/prosody/data/dev.txt", nlp)
    train_input, train_labels = prepare_splits("data/prosody/data/train_360.txt", nlp)

    with open("data/test_input.json", "w", encoding="utf-8") as f:
        json.dump(test_input, f, ensure_ascii=False)
    with open("data/test_labels.json", "w", encoding="utf-8") as f:
        json.dump(test_input, f, ensure_ascii=False)
    with open("data/dev_input.json", "w", encoding="utf-8") as f:
        json.dump(dev_input, f, ensure_ascii=False)
    with open("data/dev_labels.json", "w", encoding="utf-8") as f:
        json.dump(dev_labels, f, ensure_ascii=False)
    with open("data/train_input.json", "w", encoding="utf-8") as f:
        json.dump(train_input, f, ensure_ascii=False)
    with open("data/train_labels.json", "w", encoding="utf-8") as f:
        json.dump(train_labels, f, ensure_ascii=False)


class Context:
    def __init__(self):
        self.free_context = []
        self.np_context = []
        self.npvp_context = []
        self.len = 0

    def __len__(self):
        return self.len

    def get_free_context_str(self):
        return ' '.join(self.free_context)

    def get_np_context_str(self):
        return ' '.join([word for n in self.np_context for word in n])

    def get_npvp_context_str(self):
        return ' '.join([word for n in self.npvp_context for word in n])


def update_4gram_dict(context_str_func, target_dict, l):
    """
    Updates the target dictionary for the given context string function and label.

    Args:
        context_str_func: Function to retrieve the context string.
        target_dict: Dictionary to update.
        l: Label to update the dictionary with.
    """
    context_str = context_str_func()
    if context_str in target_dict:
        if l in target_dict[context_str]:
            target_dict[context_str][l] += 1
        else:
            target_dict[context_str][l] = 1
    else:
        target_dict[context_str] = {l: 1}


def get_4_gram():
    _input = []
    labels = []
    with open("data/dev_input.json") as f:
        _input += json.load(f)
    with open("data/test_input.json") as f:
        _input += json.load(f)
    with open("data/train_input.json") as f:
        _input += json.load(f)
    with open("data/dev_labels.json") as f:
        labels += json.load(f)
    with open("data/test_labels.json") as f:
        labels += json.load(f)
    with open("data/train_labels.json") as f:
        labels += json.load(f)

    _1gram = {}
    free_4gram = {}
    np_4gram = {}
    npvp_4gram = {}
    context = Context()
    np_current_token = [] # current token span, with np tags
    npvp_current_token = [] # current token span, with np, vp tags
    for tt, ll in tqdm(zip(_input, labels)):
        for t, l in zip(tt, ll):
            if t in ["<VP>", "</VP>"]:
                assert l == "NA"

                # update current token
                npvp_current_token.append(t)
            elif t in ["<NP>", "</NP>"]:
                assert l == "NA"

                # update current token
                npvp_current_token.append(t)
                np_current_token.append(t)
            # lexical item or punctuation here
            else:
                # update current token
                npvp_current_token.append(t)
                np_current_token.append(t)

                # pop oldest context if window is full
                if context.len == 3:
                    context.free_context.pop(0)
                    context.np_context.pop(0)
                    context.npvp_context.pop(0)
                    context.len -= 1
                # increase length is window not full
                # update context
                context.free_context.append(t)
                context.np_context.append(np_current_token)
                context.npvp_context.append(npvp_current_token)
                context.len += 1

                # Update dictionaries using the function
                if context.len == 3:
                    update_4gram_dict(context.get_free_context_str, free_4gram, l)
                    update_4gram_dict(context.get_np_context_str, np_4gram, l)
                    update_4gram_dict(context.get_npvp_context_str, npvp_4gram, l)

                # reset current token
                np_current_token = []
                npvp_current_token = []

    with open("free_4gram.json", "w", encoding="utf-8") as f:
        json.dump(free_4gram, f, indent=4)
    with open("np_4gram.json", "w", encoding="utf-8") as f:
        json.dump(np_4gram, f, indent=4)
    with open("npvp_4gram.json", "w", encoding="utf-8") as f:
        json.dump(npvp_4gram, f, indent=4)


def calculate_entropies(four_gram_counts):
    """
    Calculates conditional entropy H(w|C) and marginal entropy H(w) from 4-gram counts.

    Args:
        four_gram_counts: Dictionary with 4-gram counts
                          {('w1', 'w2', 'w3'): {'w4': count, ...}}.

    Returns:
        H_w_given_C: Conditional entropy H(w|C).
        H_w: Marginal entropy H(w).
    """
    context_totals = defaultdict(int)  # Total counts for each context
    total_count = 0  # Total count of all 4-grams
    word_totals = defaultdict(int)  # Total counts for each word w4 across all contexts

    # Step 1: Calculate totals for contexts and words
    for context, next_word_counts in four_gram_counts.items():
        for next_word, count in next_word_counts.items():
            context_totals[context] += count
            word_totals[next_word] += count
            total_count += count

    # Step 2: Calculate H(w|C) (Conditional Entropy)
    H_w_given_C = 0.0
    for context, next_word_counts in four_gram_counts.items():
        context_prob = context_totals[context] / total_count  # p(C)
        for next_word, count in next_word_counts.items():
            p_w_given_C = count / context_totals[context]  # p(w|C)
            if p_w_given_C > 0:
                H_w_given_C -= context_prob * p_w_given_C * math.log(p_w_given_C, 2)

    # Step 3: Calculate H(w) (Marginal Entropy)
    H_w = 0.0
    for word, count in word_totals.items():
        p_w = count / total_count  # p(w)
        if p_w > 0:
            H_w -= p_w * math.log(p_w, 2)

    return H_w_given_C, H_w


def main():
    prepare_dataset()
    get_4_gram()
    free_4gram = json.load(open("free_4gram.json"))
    np_4gram = json.load(open("np_4gram.json"))
    npvp_4gram = json.load(open("npvp_4gram.json"))

    free_conditional, free_marginal = calculate_entropies(free_4gram)
    print(f"H(P) = {free_marginal}, H(P|W) = {free_conditional}. Redundancy = {free_marginal - free_conditional}")

    np_conditional, np_marginal = calculate_entropies(np_4gram)
    print(f"H(P) = {np_marginal}, H(P|W) = {np_conditional}. Redundancy = {np_marginal - np_conditional}")

    npvp_conditional, npvp_marginal = calculate_entropies(npvp_4gram)
    print(f"H(P) = {npvp_marginal}, H(P|W) = {npvp_conditional}. Redundancy = {npvp_marginal - npvp_conditional}")


if __name__ == "__main__":
    get_4_gram()