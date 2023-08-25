#!/usr/bin/env bash

# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh
# Adapted from https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh

git clone https://github.com/moses-smt/mosesdecoder.git

MOSES=`pwd`/mosesdecoder

SCRIPTS=${MOSES}/scripts
TOKENIZER=${SCRIPTS}/tokenizer/tokenizer.perl
LC=${SCRIPTS}/tokenizer/lowercase.perl
CLEAN=${SCRIPTS}/training/clean-corpus-n.perl
URL="http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz"
GZ=de-en.tgz

merge_ops=6000
src=de
tgt=en
lang=de-en
prep="../test/data/iwslt14"
tmp=${prep}/tmp
orig=orig

mkdir -p ${orig} ${tmp} ${prep}

echo "Downloading data from ${URL}..."
cd ${orig}
curl -O "${URL}"

if [ -f ${GZ} ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf ${GZ}
cd ..

echo "pre-processing train data..."
for l in ${src} ${tgt}; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat ${orig}/${lang}/${f} | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl ${TOKENIZER} -threads 8 -l $l > ${tmp}/${tok}
    echo ""
done
perl ${CLEAN} -ratio 1.5 ${tmp}/train.tags.${lang}.tok ${src} ${tgt} ${tmp}/train.tags.${lang}.clean 1 80
for l in ${src} ${tgt}; do
    perl ${LC} < ${tmp}/train.tags.${lang}.clean.${l} > ${tmp}/train.tags.${lang}.${l}
done

echo "pre-processing valid/test data..."
for l in ${src} ${tgt}; do
    for o in `ls ${orig}/${lang}/IWSLT14.TED*.${l}.xml`; do
    fname=${o##*/}
    f=${tmp}/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl ${TOKENIZER} -threads 8 -l ${l} | \
    perl ${LC} > ${f}
    echo ""
    done
done

echo "creating train, valid, test..."
for l in ${src} ${tgt}; do
    awk '{if (NR%23 == 0)  print $0; }' ${tmp}/train.tags.de-en.${l} > ${tmp}/valid.${l}
    awk '{if (NR%23 != 0)  print $0; }' ${tmp}/train.tags.de-en.${l} > ${tmp}/train.${l}

    cat ${tmp}/IWSLT14.TED.dev2010.de-en.${l} \
        ${tmp}/IWSLT14.TEDX.dev2012.de-en.${l} \
        ${tmp}/IWSLT14.TED.tst2010.de-en.${l} \
        ${tmp}/IWSLT14.TED.tst2011.de-en.${l} \
        ${tmp}/IWSLT14.TED.tst2012.de-en.${l} \
        > ${tmp}/test.${l}
done

for l in ${src} ${tgt}; do
    echo "learning * ${l} * BPE..."
    codes_file="${tmp}/bpe.${merge_ops}.${l}"
    vocab_file="${tmp}/vocab.${l}"
    echo "codes file ${codes_file}"
    # python3 -m subword_nmt.learn_bpe -s "${merge_ops}" -i "${tmp}/train.${l}" -o "${codes_file}"
    subword-nmt learn-joint-bpe-and-vocab -i "${tmp}/train.${l}" -s "${merge_ops}" -o "${codes_file}" --write-vocabulary "${vocab_file}"
done

echo "applying BPE..."
for l in ${src} ${tgt}; do
    codes_file="${tmp}/bpe.${merge_ops}.${l}"
    for p in train valid test; do
        python3 -m subword_nmt.apply_bpe -c "${codes_file}" -i "${tmp}/${p}.${l}" -o "${prep}/${p}.bpe.${merge_ops}.${l}"
    done
done

for l in ${src} ${tgt}; do
    codes_file="${tmp}/bpe.${merge_ops}.${l}"
    vocab_file="${tmp}/vocab.${l}"
    mv "${codes_file}" "${prep}/"
    mv "${vocab_file}" "${prep}/"
    for p in train valid test; do
        mv ${tmp}/${p}.${l} ${prep}/
    done
done

rm -rf ${MOSES}
rm -rf ${tmp}
