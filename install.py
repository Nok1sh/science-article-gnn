import subprocess
import sys
import urllib.request


def pip_install(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)


def find_pyg_wheel_index() -> str | None:
    import torch

    base = torch.__version__.split("+")[0]
    major, minor = base.split(".")[:2]
    cuda = "cu" + torch.version.cuda.replace(".", "") if torch.version.cuda else "cpu"
    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | {cuda}")

    for name in (f"torch-{base}+{cuda}", f"torch-{major}.{minor}.0+{cuda}"):
        url = f"https://data.pyg.org/whl/{name}.html"
        try:
            page = urllib.request.urlopen(url).read().decode()
        except OSError:
            continue
        if "pyg_lib" in page:
            return url
    return None


def main() -> None:
    pip_install("-e", ".")

    url = find_pyg_wheel_index()
    if url is None:
        sys.exit(
            "No prebuilt pyg-lib for this torch/CUDA build. Build it from source:\n"
            "  pip install git+https://github.com/pyg-team/pyg-lib.git"
        )
    print(f"installing pyg-lib from {url}")
    pip_install("pyg-lib", "-f", url)

    from torch_geometric.typing import WITH_PYG_LIB

    print(f"pyg-lib available: {WITH_PYG_LIB}")


if __name__ == "__main__":
    main()
