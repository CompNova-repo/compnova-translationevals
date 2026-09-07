from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "jonavila/DRAL"
OUT_DIR = Path("data/dral")
NUM_PAIRS = 30

api = HfApi()

# Ask Hugging Face for filenames only -- this does not download
# the whole 22 GB dataset.
files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset",
)

wav_files = [
    f for f in files
    if f.lower().endswith(".wav")
    and "fragments-long" in f.lower()
]

# Map files by their language-independent pair identifier.
english = {}
spanish = {}

for path in wav_files:
    name = Path(path).name

    if name.startswith("EN_"):
        english[name[3:]] = path
    elif name.startswith("ES_"):
        spanish[name[3:]] = path

pair_ids = sorted(set(english) & set(spanish))

if not pair_ids:
    raise RuntimeError(
        "No matched EN/ES fragment pairs found. "
        "Print `wav_files[:50]` to inspect the current DRAL layout."
    )

selected = pair_ids[:NUM_PAIRS]

print(f"Found {len(pair_ids)} matched pairs.")
print(f"Downloading {len(selected)} pairs...")

OUT_DIR.mkdir(parents=True, exist_ok=True)

for pair_id in selected:
    for lang, source_path in (
        ("EN", english[pair_id]),
        ("ES", spanish[pair_id]),
    ):
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=source_path,
        )

        destination = OUT_DIR / f"{lang}_{pair_id}"
        destination.write_bytes(Path(downloaded).read_bytes())

        print(destination)

print(f"\nDone: {len(selected) * 2} WAV files in {OUT_DIR}")