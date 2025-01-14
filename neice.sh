ROOT_DIR=$1
DATASET=${ROOT_DIR}/dataset/deezer_podcast_dataset.tsv
OUTPUT_DIR=${ROOT_DIR}/output
MODEL_DIR=${ROOT_DIR}/models
RESULTS_DIR=${ROOT_DIR}/results
BATCH_SIZE=64
T=10
k_nearest=500

mkdir -p $MODEL_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $RESULTS_DIR

# Download the Wikipedia2Vec model if it does not exist
if [ ! -f "${MODEL_DIR}/enwiki_20180420_300d.pkl" ]; then
    wget http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_300d.pkl.bz2 -P $MODEL_DIR
    bzip2 -d ${MODEL_DIR}/enwiki_20180420_300d.pkl.bz2
    echo "Download and decompression of model completed successfully!"
else
    echo "File already exists at ${MODEL_DIR}/enwiki_20180420_300d.pkl"
fi

cd ${ROOT_DIR}/ptm/entity_linking

# Download the required files (pre-trained Entity Linker models) if they do not exist
if [ -d "${OUTPUT_DIR}/wiki_2019" ] && [ -d "${OUTPUT_DIR}/generic" ] && [ -d "${OUTPUT_DIR}/ed-wiki-2019" ]; then
    echo "All required directories exist. No download needed."
else
    echo "One or more directories missing. Running download script..."
    ./download-files.sh $OUTPUT_DIR
fi

# Generate a JSON file which contains all the NEs extracted and linked from the podcast metadata
if [ -f "${OUTPUT_DIR}/linked_entities.json" ]; then
    echo "File already exists at ${OUTPUT_DIR}/linked_entities.json"
else
    echo "Running entity linking script..."
    ./launch-rel-batch.sh $OUTPUT_DIR $DATASET $OUTPUT_DIR $BATCH_SIZE
fi

cd ${ROOT_DIR}/ptm/data_preprocessing
if [ -f "${OUTPUT_DIR}/prepro.txt" ]; then
    echo "File already exists at ${OUTPUT_DIR}/prepro.txt"
else
    python main_prepro.py --examples_file $DATASET \
    --annotated_file ${OUTPUT_DIR}/linked_entities.json \
    --embeddings_file_path ${MODEL_DIR}/enwiki_20180420_300d.pkl \
    --path_to_save_results ${OUTPUT_DIR}
fi

# Obtain T most similar words to the entities for each alpha_ent in [0.3, 0.4]
for alpha_ent in 0.30 0.40; do

    if [ -f "${OUTPUT_DIR}/prepro_enrich_entities_th${alpha_ent}_k${k_nearest}.txt" ]; then
        echo "File already exists at ${OUTPUT_DIR}/prepro_enrich_entities_th${alpha_ent}_k${k_nearest}.txt"
    else
        python main_enrich_corpus_using_entities.py --prepro_file ${OUTPUT_DIR}/prepro.txt \
            --prepro_le_file ${OUTPUT_DIR}/prepro_le.txt \
            --vocab_file ${OUTPUT_DIR}/vocab.txt \
            --vocab_le_file ${OUTPUT_DIR}/vocab_le.txt \
            --embeddings_file_path ${MODEL_DIR}/enwiki_20180420_300d.pkl \
            --path_to_save_results $OUTPUT_DIR \
            --alpha_ent ${alpha_ent} \
            --k ${k_nearest} # ${T}
    fi
done

# Download Palmetto and wikipedia_bd for evaluation
cd $ROOT_DIR/ptm/evaluation

if [ -f "${ROOT_DIR}/ptm/evaluation/palmetto-0.1.0-jar-with-dependencies.jar" ]; then
    echo "File already exists at ${ROOT_DIR}/ptm/evaluation/palmetto-0.1.0-jar-with-dependencies.jar"
else
    wget --no-check-certificate https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/palmetto-0.1.0-jar-with-dependencies.jar -P $ROOT_DIR/ptm/evaluation
fi

if [ -d "${OUTPUT_DIR}/wikipedia_bd" ]; then
    echo "Directory already exists at ${OUTPUT_DIR}/wikipedia_bd"
else
    wget --no-check-certificate https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/Wikipedia_bd.zip -P $OUTPUT_DIR
    unzip $OUTPUT_DIR/Wikipedia_bd.zip -d $OUTPUT_DIR
    rm -rf $OUTPUT_DIR/Wikipedia_bd.zip
fi

WIKIPEDIA_DB_PATH=${OUTPUT_DIR}/wikipedia_bd

echo "run,K,alpha_word,alpha_ent,evaluation_result" > ${RESULTS_DIR}/results_summary.txt

# Extract K topics from the podcast metadata corpus and for each topic its T tops words 
# Do experiments for K in [20, 50, 100, 200] and alpha_word in [0.2, 0.3, 0.4, 0.5]
# Do experiments for N runs for statistical significance
RUNS=20
for i in $(seq 1 $RUNS); do
    for K in 20 50 100 200; do
        for alpha_word in 0.2 0.3 0.4 0.5; do
            for alpha_ent in 0.30 0.40; do

                cd $ROOT_DIR
                python ptm/neice_model/main.py \
                    --corpus ${OUTPUT_DIR}/prepro_enrich_entities_th${alpha_ent}_k${k_nearest}.txt \
                    --embeddings ${MODEL_DIR}/enwiki_20180420_300d.pkl \
                    --output_dir ${OUTPUT_DIR} \
                    --mask_entities_file $OUTPUT_DIR/mask_enrich_entities_th${alpha_ent}_k${k_nearest}.npz \
                    --vocab $OUTPUT_DIR/new_vocab_th${alpha_ent}_k${k_nearest}.txt \
                    --n_topics $K \
                    --n_neighbours ${T} \
                    --alpha_word $alpha_word \
                    --NED

                cd $ROOT_DIR/ptm/evaluation

                ./evaluate-topics.sh ${OUTPUT_DIR}/top_words.txt ${T} $WIKIPEDIA_DB_PATH > ${OUTPUT_DIR}/evaluation_result.txt
                EVALUATION_RESULT=$(cat ${OUTPUT_DIR}/evaluation_result.txt)
                echo "${i},${K},${alpha_word},${alpha_ent},${EVALUATION_RESULT}" >> ${RESULTS_DIR}/results_summary.txt

            done
        done
    done
done


