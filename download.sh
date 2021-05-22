# Ce script tèlécharge les données de la base ClaimsKG
# depuis le lien suivant: http://tinyurl.com/3smozlx6
#
# Le fichier est sauvgardé sous le dossier data/,
# si il n'existe pas, il est alors créer

mkdir -p data
wget http://tinyurl.com/3smozlx6 -O data/claimKG.zip
unzip data/claimKG.zip -d data/ -f
mv data/claim_extraction_18_10_2019_annotated.csv data/claimKG.csv
rm data/claimKG.zip
