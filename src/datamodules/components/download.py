import ssl
import urllib
from pathlib import Path

from torchvision.datasets.utils import check_integrity, extract_archive
from tqdm import tqdm


def download_and_extract_archive(url, path, md5=None):
    # Works when the SSL certificate is expired for the link
    path = Path(path)
    extract_path = path
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / Path(url).name
        if not file_path.exists() or not check_integrity(file_path, md5):
            print(f"{file_path} not found or corrupted")
            print(f"downloading from {url}")
            context = ssl.SSLContext()
            with urllib.request.urlopen(url, context=context) as response:
                with tqdm(total=response.length) as pbar:
                    with open(file_path, "wb") as file:
                        chunk_size = 1024
                        chunks = iter(lambda: response.read(chunk_size), "")
                        for chunk in chunks:
                            if not chunk:
                                break
                            pbar.update(chunk_size)
                            file.write(chunk)
            extract_archive(str(file_path), str(extract_path))
    return extract_path
