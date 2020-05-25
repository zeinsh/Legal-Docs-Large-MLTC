#!/bin/bash

# Donwload jrc-acquis data
#uncomment to install jrc-aquis data usning links provided in conf/jrc_aquis_links.txt
input="./conf/jrc_aquis_links.txt"

mkdir tmp
while IFS= read -r url
do
  echo "$url"
  filename="$(basename -- $url)"
  wget -P tmp/ $url
  tar  -C tmp/ -xzvf tmp/$filename
done < "$input"

# download and prepare eurovoc data and eurovoc analysis tool
wget http://publications.europa.eu/resource/distribution/eurovoc/20190329-1/zip/eurovoc_dtd/eurovoc_xml.zip -O tmp/eurovoc_xml.zip
unzip tmp/eurovoc_xml.zip -d tmp/EuroVoc
python prepare_eurovoc.py

# prepare jrc-aquis datasets
python prepare_jrc_data.py --languages "en" --save_path "datasets/jrc_en_basic.csv"            # monolingual English
python prepare_jrc_data.py --languages "en,de,fr" --save_path "datasets/jrc_3langs_basic.csv"  # multi-lingual

################# EURLEX57K ####################
wget -O tmp/datasets.zip http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip
unzip tmp/datasets.zip -d tmp/EURLEX57K

python prepare_eurlex57k_data.py --dataset_path "./tmp/EURLEX57K/dataset/" --save_path "datasets/EurLex57K.csv"