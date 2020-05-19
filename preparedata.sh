#!/bin/bash

#uncomment to install jrc-aquis data usning links provided in conf/jrc_aquis_links.txt
#input="./conf/jrc_aquis_links.txt"
#
#mkdir tmp
#while IFS= read -r url
#do
#  echo "$url"
#  filename="$(basename -- $url)"
#  wget -P tmp/ $url
#  tar  -C tmp/ -xzvf tmp/$filename
#done < "$input"

wget http://publications.europa.eu/resource/distribution/eurovoc/20190329-1/zip/eurovoc_dtd/eurovoc_xml.zip -O tmp/eurovoc_xml.zip
unzip tmp/eurovoc_xml.zip -d tmp/EuroVoc
python perpare_eurovoc.py

python download_jrc_data.py --languages "en" --save_path "datasets/jrc_en_basic.csv"
python download_jrc_data.py --languages "en,de,fr" --save_path "datasets/jrc_3langs_basic.csv"
