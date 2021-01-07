for model in ./model_weights/*
do
    if test -d $model
    then
        echo $model
        for name in 'appa' # 'megaage_asian' 'morph' 'UTKFace' 'balanced'
            do
            CUDA_VISIBLE_DEVICES=0 \
            /home/david/anaconda3/envs/david37/bin/python test.py -test_path ./data/original/${name}_test.tsv -trained_model ${model} -test_split 1 -result_folder test_results
            done
    fi
done