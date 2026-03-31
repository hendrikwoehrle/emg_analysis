"""
Download NinaPro EMG datasets.

All datasets are freely downloadable without registration.
DB1-DB3 are hosted on ninapro.hevs.ch; DB4, DB5 on Zenodo.

Usage:
    python get_ninapro.py --db 1
    python get_ninapro.py --db 4 --output ../data/ninapro
    python get_ninapro.py --db 1 --subjects 1 3 5
"""

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Dataset metadata: number of subjects and base download URL
DATASETS = {
    1: {
        "n_subjects": 27,
        "url_template": "https://ninapro.hevs.ch/files/DB1/Preprocessed/s{subject}.zip",

        "description": "DB1 — 27 intact subjects, 52 movements, 10 Otto Bock electrodes",
    },
    2: {
        "n_subjects": 40,
        "url_template": "https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s{subject}.zip",

        "description": "DB2 — 40 intact subjects, 49 movements, 12 Delsys Trigno electrodes",
    },
    3: {
        "n_subjects": 11,
        "url_template": "https://ninapro.hevs.ch/files/db3_Preproc/s{subject}_0.zip",

        "description": "DB3 — 11 trans-radial amputees, 49 movements, 12 Delsys Trigno electrodes",
    },
    4: {
        "n_subjects": 10,
        "url_template": "https://zenodo.org/records/1000138/files/s{subject}.zip",

        "description": "DB4 — 10 intact subjects, 53 movements, Cometa Wave Plus + Dormo",
    },
    5: {
        "n_subjects": 10,
        "url_template": "https://zenodo.org/records/1000116/files/s{subject}.zip",

        "description": "DB5 — 10 intact subjects, 53 movements, 2x Thalmic Myo armband",
    },
}


def download_file(url: str, dest: Path, session: requests.Session) -> bool:
    """Download a single file with a progress bar. Returns True on success."""
    try:
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()
    except requests.HTTPError as e:
        print(f"  HTTP error {e.response.status_code} for {url}", file=sys.stderr)
        return False
    except requests.RequestException as e:
        print(f"  Request failed: {e}", file=sys.stderr)
        return False

    total = int(response.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    return True


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    print(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NinaPro EMG datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  DB{k}: {v['description']}" for k, v in DATASETS.items()
        ),
    )
    parser.add_argument(
        "--db",
        type=int,
        required=True,
        choices=list(DATASETS.keys()),
        metavar="N",
        help=f"Dataset number ({', '.join(str(k) for k in DATASETS)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ninapro"),
        help="Root output directory (default: ../data/ninapro)",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        metavar="S",
        help="Specific subject numbers to download (default: all)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Keep zip files instead of extracting them",
    )
    args = parser.parse_args()

    db = DATASETS[args.db]
    out_dir = args.output / f"DB{args.db}"
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects or list(range(1, db["n_subjects"] + 1))

    # Validate subject numbers
    invalid = [s for s in subjects if s < 1 or s > db["n_subjects"]]
    if invalid:
        parser.error(
            f"DB{args.db} has subjects 1–{db['n_subjects']}. "
            f"Invalid: {invalid}"
        )

    print(f"Downloading NinaPro DB{args.db}")
    print(f"  {db['description']}")
    print(f"  Subjects : {subjects}")
    print(f"  Output   : {out_dir.resolve()}")

    session = requests.Session()

    failed = []
    for subject in subjects:
        url = db["url_template"].format(subject=subject)
        zip_path = out_dir / f"s{subject}.zip"

        if zip_path.exists():
            print(f"  s{subject}: zip already exists, skipping download")
        else:
            # Check if already extracted (any file starting with s{subject})
            existing = list(out_dir.glob(f"S{subject}_*")) + list(out_dir.glob(f"s{subject}_*"))
            if existing and not args.no_extract:
                print(f"  s{subject}: already extracted, skipping")
                continue

            print(f"  s{subject}: downloading from {url}")
            ok = download_file(url, zip_path, session)
            if not ok:
                failed.append(subject)
                continue

        if not args.no_extract and zip_path.exists():
            extract_zip(zip_path, out_dir)

    print(f"\nDone. Data saved to: {out_dir.resolve()}")
    if failed:
        print(f"Failed subjects: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
