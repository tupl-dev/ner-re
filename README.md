# NER-RE
Playing around with NER and RE in Python 3.14.

## Setup
### Windows
```shell
# setup
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# run
docker compose up -d
python graph.py -f 'data/obama.pdf'
```

### Unix
```shell
# setup
python -m venv .venv
. ./.venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# run
docker compose up -d
python graph.py -f 'data/obama.pdf'
```
Visit `http://localhost:7474/` (username: `neo4j`, password: `password`) to view Neo4j instance.
