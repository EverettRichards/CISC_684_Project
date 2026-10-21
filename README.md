# Instructions:

## Setup the Repository

**Clone the GitHub repository to your machine:**

`git clone https://github.com/EverettRichards/CISC_684_Project.git`

**Navigate into the directory:**

`cd CISC_684_Project`

**Create a new Venv:** *(If you prefer Conda, skip this step and go do that... I don't judge.)*

`python -m venv .venv`

**Activate the Venv:**

Linux:

`source .venv/bin/activate`

Windows:

`.venv/Scripts/Activate`

**Install required dependencies**

`pip install -r requirements.txt`

**Download the dataset**

This script will automatically download the dataset, downsample it, and perform test/train/validation split, putting the resulting files in the `data` directory:

`python download_dataset.py`


*Note: In Jupyter Notebooks, select the Venv as the execution environment.*

**Download OLLama**

Linux:

`curl -fsSL https://ollama.com/install.sh | sh`

`ollama pull llama3.2`
