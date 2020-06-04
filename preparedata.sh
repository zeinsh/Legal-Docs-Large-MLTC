#!/bin/bash

# Donwload jrc-acquis data
#uncomment to install jrc-aquis data usning links provided in conf/jrc_aquis_links.txt
input="./conf/jrc_aquis_links.txt"
dataset_split="./Iterative_Split/JRC_Aquis"

mkdir tmp

echo "Download JRC-Acquis data ..."
while IFS= read -r url
do
  echo "$url"
  filename="$(basename -- $url)"
  wget -P tmp/ $url
  tar  -C tmp/ -xzvf tmp/$filename
done < "$input"

echo "Download and prepare eurovoc data and eurovoc analysis tool ..."
wget http://publications.europa.eu/resource/distribution/eurovoc/20190329-1/zip/eurovoc_dtd/eurovoc_xml.zip -O tmp/eurovoc_xml.zip
unzip tmp/eurovoc_xml.zip -d tmp/EuroVoc
python prepare_eurovoc.py

# prepare jrc-aquis datasets
echo "Prepare JRC-Acquis monolingual dataset"
python prepare_jrc_data.py --languages "en" \
                           --save_path "datasets/jrc_en_basic.csv" \
                           --dataset_split $dataset_split

echo "Prepare JRC-Acquis multilingual dataset (English, French, German)"
python prepare_jrc_data.py --languages "en,de,fr" \
                            --save_path "datasets/jrc_3langs_basic.csv" \
                            --dataset_split $dataset_split

################# EURLEX57K ####################
echo "Download and prepare EURLEX57K dataset"
wget -O tmp/datasets.zip http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip
unzip tmp/datasets.zip -d tmp/EURLEX57K

python prepare_eurlex57k_data.py --dataset_path "./tmp/EURLEX57K/dataset/" --save_path "datasets/EurLex57K.csv"

echo "Delete tmp directory ..."
rm -r tmp

echo "Done!"