# ino-sgd
Supplementary programmes for INO-SGD.


Save requirements:
```
pip list --format=freeze | grep -i -E -f <(conda list --json | python -c 'import sys,json; print("\n".join("^"+p["name"].lower().replace("_","-")+"==" for p in json.load(sys.stdin) if p.get("channel")=="pypi"))') > requirements.txt
```
