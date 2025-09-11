# scripts/create_medical_seed.py
import json
import os

import datasets
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

datasets.config.HF_HUB_TIMEOUT = 30


def _with_ext(path, fmt):
    ext = ".json" if fmt == "json" else ".parquet"
    return path if path.endswith(ext) else path + ext


def save_sample(sample_data, output_path, output_format):
    out = _with_ext(output_path, output_format)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if output_format == "json":
        with open(out, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
    elif output_format == "parquet":
        table = pa.Table.from_pylist(sample_data)  # list of dicts
        pq.write_table(table, out, compression="snappy")
    else:
        raise ValueError(f"Unknown output_format: {output_format}")

    return out


def filter_animal_studies(text_chunk):
    """Returns True if content contains animal studies (should be filtered out)"""
    animal_indicators = [
        "mice",
        "mouse",
        "rats",
        "rat",
        "murine",
        "porcine",
        "bovine",
        "canine",
        "feline",
        "in vivo",
        "animal model",
        "laboratory animals",
        "C57BL/6",
        "Sprague-Dawley",
        "Wistar",
        "xenograft",
        "transgenic mice",
    ]

    text_lower = text_chunk.lower()
    return any(term in text_lower for term in animal_indicators)


def create_medical_seed(
    dataset_path,
    seed_size=None,
    min_content_len=100,
    output_path="data/medical_wikipedia_seed.json",
    output_format=["parquet", "json"],
    source=["pubmed", "wikipeida", "textbook"],
):
    """Create medical Wikipedia seed dataset"""

    print(f"Loading MedRAG {dataset_path} dataset...")
    ds = load_dataset(dataset_path, streaming=True)

    sample_data = []

    for i, item in enumerate(ds["train"]):
        if seed_size != None:
            if len(sample_data) >= seed_size:
                break

        text = item.get("content", "")
        if source == "pubmed":
            if filter_animal_studies(text):
                continue  #! Skip this chunk - it contains animal study content
        if len(text) > min_content_len:
            sample_data.append(
                {
                    "id": item.get("id", str(i)),
                    "title": item.get("title", ""),
                    "text": text,
                    #'url': item.get('url', ''),
                    "wiki_id": item.get("wiki_id", ""),
                    "source": source,
                }
            )

        if i % 1000 == 0:
            print(f"Processed {i} medical articles, collected {len(sample_data)}")

    full_out_path = save_sample(sample_data, output_path, output_format)

    print(f"Created medical seed dataset with {len(sample_data)} articles")
    print(f"Saved to: {full_out_path}")
    print(f"File size: {os.path.getsize(full_out_path) / (1024 * 1024):.1f} MB")

    return sample_data


if __name__ == "__main__":
    # create_medical_seed(dataset_path="MedRAG/wikipedia", seed_size=2000, output_path=f"data/med_wiki_seed", output_format="parquet")
    # create_medical_seed(dataset_path="MedRAG/wikipedia", seed_size=2000, output_path=f"data/small_seed/medical_wiki_seed_small", output_format="json", source='wikipedia')
    #! Data is way too generalized and not focused on medical
    # create_medical_seed(dataset_path="MedRAG/textbooks", seed_size=2000, output_path=f"data/small_seed/medical_textbook_seed_small", output_format="json", source='textbook')
    # create_medical_seed(dataset_path="MedRAG/pubmed", seed_size=2000, output_path=f"data/small_seed/medical_pubmed_seed_small", output_format="json", source='pubmed')
    # create_medical_seed(dataset_path="MedRAG/textbooks", seed_size=60000, output_path=f"../data/medium_seed/medical_textbook_seed_medium", output_format="json", source='textbook') #max is 125,847 rows
    # create_medical_seed(dataset_path="MedRAG/pubmed", seed_size=600000, output_path=f"../data/medium_seed/medical_pubmed_seed_medium", output_format="json", source='pubmed')  #max is 2 million rows

    create_medical_seed(
        dataset_path="MedRAG/pubmed",
        seed_size=6,
        output_path=f"../data/medium_seed/test",
        output_format="json",
        source="pubmed",
    )  # max is 2 million rows
