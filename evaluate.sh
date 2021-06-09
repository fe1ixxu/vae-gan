TRAIN="./save/Jun08011647/ckpts/775_out.txt"
TEST="./data/cs-norepeat/test.neg"
DATA="../VACS/language_model/temp/" 
mkdir ${DATA}

cp ${TRAIN} ${DATA}train.txt
cp ${TEST} ${DATA}test.txt
cp ${TEST} ${DATA}valid.txt
cd ../VACS/language_model

CUDA_VISIBLE_DEVICES=0 python train.py

cd ../../vae-gan/
rm -rf ${DATA}
