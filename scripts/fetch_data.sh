#!/bin/bash
# Public text-to-SQL data for Luthor. Run from the repo root; everything lands in data/raw/.
set -euo pipefail
R=data/raw; mkdir -p $R
# Spider 1.0 (train + dev + test, 200+ SQLite databases; CC BY-SA 4.0)
[ -d $R/spider_data ] || { curl -sL -o $R/spider.zip "https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t"; unzip -q $R/spider.zip -d $R; rm $R/spider.zip; rm -rf $R/__MACOSX; }
# Spider-Syn: Spider train/dev with schema words replaced by synonyms (MIT)
mkdir -p $R/spider_syn
for f in train_spider.json dev.json; do [ -f $R/spider_syn/$f ] || curl -sL -o $R/spider_syn/$f https://raw.githubusercontent.com/ygan/Spider-Syn/main/Spider-Syn/$f; done
# WikiSQL (single-table, BSD-3)
[ -d $R/wikisql ] || { curl -sL https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2 | tar -xj -C $R; mv $R/data $R/wikisql; }
# SynSQL-2.5M (Apache-2.0): databases + tables; questions streamed and trimmed (drops the CoT text)
mkdir -p $R/synsql
[ -f $R/synsql/tables.json ] || curl -sL -o $R/synsql/tables.json https://huggingface.co/datasets/seeklhy/SynSQL-2.5M/resolve/main/tables.json
[ -d $R/synsql/databases ] || { curl -sL -o $R/synsql/databases.zip https://huggingface.co/datasets/seeklhy/SynSQL-2.5M/resolve/main/databases.zip; unzip -q $R/synsql/databases.zip -d $R/synsql; rm $R/synsql/databases.zip; }
[ -f $R/synsql/data.json ] || curl -sL https://huggingface.co/datasets/seeklhy/SynSQL-2.5M/resolve/main/data.json \
  | .venv/bin/python scripts/trim_synsql.py > $R/synsql/data.json
echo done
