from datasets import load_dataset, interleave_datasets

"""
load_dataset("TheBritishLibrary/blbooks", streaming=True)
    .filter(lambda x: x["date"]<=1899 and x["ocr_confidence"]>0.92),
load_dataset("dell-research-harvard/AmericanStories", "subset_years",
                year_list=[str(y) for y in range(1800,1900)])


# royal society
# TODO - no year - need regex filtering
https://huggingface.co/datasets/bigscience-data/roots_en_royal_society_corpus
"""

def main():
    ds = load_dataset("emozilla/pg19", num_proc=5, split="train")
    ds = ds.filter(lambda x: x["publication_date"] < 1900)
    ds = ds.remove_columns(["short_book_title", "url"])
    # columns at this point are ("text", "publication_date")

    # mix = interleave_datasets(sources, probabilities=[0.4,0.3,0.3], seed=42)
    mix = ds

    def clean(example):
        example["text"] = example["text"].replace("\r\n","\n").strip()
        return example

    cleaned = mix.map(clean, num_proc=8)
    cleaned = cleaned.shuffle(seed=42)
    cleaned.save_to_disk("pre1900_corpus.arrow")

if __name__ == "__main__":
    main()