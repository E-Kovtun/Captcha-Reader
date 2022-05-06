wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/file/d/1LFOP-t7xHSs1eOp86rsfuboJULsKys6V/view?usp=sharing' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LFOP-t7xHSs1eOp86rsfuboJULsKys6V" -O '../prepared_datasets.zip' && rm -rf /tmp/cookies.txt
cd ../
unzip prepared_datasets.zip
