import os
import shutil
import zipfile
import urllib.request
import ssl
import argparse

# Direct mirrors for the PKDD'99 Czech Financial Dataset (Berka)
BERKA_MIRROR_URLS = [
    "https://sorry.vse.cz/~berka/challenge/pkdd1999/data_berka.zip",
    "http://sorry.vse.cz/~berka/challenge/pkdd1999/data_berka.zip",
    "https://web.archive.org/web/20220120000000id_/http://sorry.vse.cz/~berka/challenge/pkdd1999/data_berka.zip",
]

REQUIRED_TABLES = [
    "account.asc",
    "client.asc",
    "disp.asc",
    "trans.asc",
    "loan.asc",
    "card.asc",
    "district.asc",
]


def _extract_asc_files(zip_path: str, target_dir: str) -> None:
    """Extracts all .asc files from a zip archive directly into target_dir (flattening nested folders)."""
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            fname = os.path.basename(member.filename).lower()
            if fname.endswith(".asc"):
                dest_path = os.path.join(target_dir, fname)
                with zf.open(member) as src, open(dest_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"  [EXTRACTED] {fname} -> {dest_path}")


def download_and_extract_berka(data_dir: str = "data/raw", zip_path: str = None) -> dict:
    """
    Ensures all required Berka ASCII tables (*.asc) exist in `data_dir`.
    1. Checks if tables are already extracted in `data_dir` (or `./berka`).
    2. Checks if a local `berka.zip` or `data_berka.zip` exists in the repo and extracts it.
    3. Otherwise, downloads `data_berka.zip` from the PKDD'99 mirrors and extracts all `.asc` files.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Check if files already exist in data_dir
    missing = [f for f in REQUIRED_TABLES if not os.path.exists(os.path.join(data_dir, f))]
    if not missing:
        print(f"All {len(REQUIRED_TABLES)} required Berka .asc tables are already present in '{data_dir}'.")
        return {f: True for f in REQUIRED_TABLES}

    # Check if a local zip archive already exists in common locations
    candidate_zips = [
        zip_path,
        os.path.join(data_dir, "data_berka.zip"),
        os.path.join(data_dir, "berka.zip"),
        "data/data_berka.zip",
        "data/berka.zip",
        "data_berka.zip",
        "berka.zip",
    ]
    local_zip = next((p for p in candidate_zips if p and os.path.exists(p)), None)

    if local_zip is not None:
        print(f"Found local archive '{local_zip}'. Extracting .asc tables into '{data_dir}'...")
        _extract_asc_files(local_zip, data_dir)
    else:
        target_zip = os.path.join(data_dir, "data_berka.zip")
        downloaded = False

        # Allow unverified SSL context for legacy academic mirror (sorry.vse.cz)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        for url in BERKA_MIRROR_URLS:
            print(f"Downloading Berka dataset from: {url} ...")
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Python-urllib/3"},
                )
                with urllib.request.urlopen(req, context=ctx, timeout=60) as response, open(target_zip, "wb") as out_file:
                    total_bytes = int(response.info().get("Content-Length", 0))
                    downloaded_bytes = 0
                    chunk_size = 1024 * 256
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded_bytes += len(chunk)
                        if total_bytes > 0:
                            pct = (downloaded_bytes / total_bytes) * 100
                            print(f"\r  Progress: {downloaded_bytes / (1024*1024):.1f} MB / {total_bytes / (1024*1024):.1f} MB ({pct:.1f}%)", end="", flush=True)
                print("\nDownload complete. Extracting .asc files...")
                _extract_asc_files(target_zip, data_dir)
                downloaded = True
                break
            except Exception as e:
                print(f"\n  [WARNING] Mirror failed ({url}): {e}")
                if os.path.exists(target_zip):
                    os.remove(target_zip)

        if not downloaded:
            print(
                "\n[ERROR] Could not automatically download data_berka.zip from mirrors.\n"
                "Please place 'berka.zip' / 'data_berka.zip' in the project root or pass --zip_path."
            )

    status = {f: os.path.exists(os.path.join(data_dir, f)) for f in REQUIRED_TABLES}
    for k, v in status.items():
        print(f"  {k:15s}: {'READY' if v else 'MISSING'}")
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and auto-extract Berka PKDD'99 ASCII tables.")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Destination folder for .asc files")
    parser.add_argument("--zip_path", type=str, default=None, help="Optional path to a local berka.zip / data_berka.zip")
    args = parser.parse_args()
    download_and_extract_berka(data_dir=args.data_dir, zip_path=args.zip_path)
