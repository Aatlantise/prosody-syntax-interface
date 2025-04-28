import stanza
from datasets import load_dataset
from ngram.ngram import annotate_phrases


def process_content(model, text):
    analysis = model(text)
    annotated_output = [annotate_phrases(s.constituency) for s in analysis.sentences]
    return ' '.join(annotated_output)

def main():
    print("Downloading Stanza models...")
    stanza.download('en')  # Only downloads once
    nlp = stanza.Pipeline('en', processors='tokenize,pos,constituency', verbose=False)

    print("Loading Wikipedia dataset...")
    dataset = load_dataset('wikipedia', '20220301.en', split='train', streaming=True, trust_remote_code=True)

    output_file = open('wikipedia_tagged_sample.txt', 'w', encoding='utf-8')

    for idx, example in enumerate(dataset):
        text = example['text']
        if not text.strip():
            continue

        print(f"Processing article {idx}...")
        try:
            tagged = process_content(nlp, text)
            output_file.write(tagged + "\n\n")
        except Exception as e:
            print(f"Error at article {idx}: {e}")

        if idx >= 9999:  # 0–9 = 10 articles
            break

    output_file.close()
    print("✅ Done! Output saved to wikipedia_tagged_sample.txt")

if __name__ == "__main__":
    main()