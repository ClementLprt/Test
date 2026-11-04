#!/usr/bin/env python3
"""Outil de développement : écrit manifest.json (empreintes des fichiers) avant chaque publication.
L'app installée compare ces empreintes et télécharge seulement ce qui a changé."""
import datetime, hashlib, json, pathlib, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parent
FILES = ["intercom_live.py", "requirements.txt", "lancer.command", "config.toml", "web/index.html"]
man = {"version": dt if (dt := sys.argv[1] if len(sys.argv) > 1 else "") else datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
       "commit": "main",
       "files": {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in FILES}}
(HERE / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
print(json.dumps(man, indent=1))
