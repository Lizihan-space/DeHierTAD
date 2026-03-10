# DeHierTAD
DeHierTAD: A novel algorithm for decoding deeply hierarchical TADs to uncover repressive genomic units.
## System Requirements

**Operating Systems:**
DeHierTAD is supported on macOS, Linux, and Windows systems. It has been tested on macOS and Windows 10.

**Software Dependencies:**
* Python (>= 3.8)
## Installation
We highly recommend using `conda` to create an isolated environment for DeHierTAD to avoid version conflicts, especially for bioinformatics packages like `cooler`.
```bash
# 1. Create and activate a conda environment (Python 3.9 is recommended)
conda create -n DeHierTAD python=3.9 -y
conda activate DeHierTAD

# 2. Install required dependencies
pip install -r requirements.txt
Download the DeHierTAD.py file to your local directory.
## Usage
DeHierTAD.py [-h] -c COOL_PATH -chr CHROMOSOME -r RESOLUTION [-A AREA] -o OUTPUT_FOLDER [-w WORKERS] -p
                    PROMINENCE [-q] [-m M] [-g G]

Identify the multi-level TAD organization from Hi-C contact matrices.

optional arguments:
  -h, --help           show this help message and exit
  -c, --COOL_PATH      Path to the cool file.
  -chr, --CHROMOSOME   Chromosome number.
  -r, --resolution     Resolution of the cool file.
  -A, --Area           Hi-C data area for first-level iteration. default=1000000.
  -o, --output_folder  Output directory.
  -w, --workers        Number of worker threads. default=4
  -p, --prominence     Prominence threshold.
  -q, --quality        Output TAD quality scores(Optional).
  -m, --m              Calculation interval: Area/m. default=10.
  -g, --g              Second-level iteration area: Area/g. default=2.
The algorithm's default output includes two directories and one BED file. The directories contain data files for boundary score curves derived from the iteration layers, with each file providing bin coordinates and their corresponding boundary scores. The BED file records the chromosome number, start and end positions, X, L and id for each TAD. Additionally, users can choose to export a supplemental file for TAD quality scores, containing the chromosome number, start and end positions, L, and the TAD quality scores.
