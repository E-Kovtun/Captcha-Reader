wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/file/d/1uZdE8d4fTV-tUMrYMTyfaCxmGb6CiE_p/view?usp=sharing' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uZdE8d4fTV-tUMrYMTyfaCxmGb6CiE_p" -O '../dataset.tar.xz' && rm -rf /tmp/cookies.txt
cd ../
tar -xvf dataset.tar.xz
