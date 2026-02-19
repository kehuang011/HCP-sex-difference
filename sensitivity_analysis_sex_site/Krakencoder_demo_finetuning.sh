#these commands will generate a new krakencoder model for LS3 data,
# fine-tuned to AGE + SEX - SITE(-AGE) conditional prediction
# Steps 1-4 only need to run once for LS3 (not specific to which demographics are predicted, conditional, etc)
# Steps 5-7 need to be run for each new set of demographics, weights, conditional choices, etc.
# (e.g., if you want to compare AGE+SEX to AGE+SEX-SITE, you need to run the following steps twice with different options)

# you need to setup own path for krakencoder

module load miniconda3
source activate torch

scriptdir=$HOME/krakencoder/krakencoder_new_20250502
scriptdir_paper=$HOME/krakenpaper/krakencoder_paper

d=$HOME/cocolab_scratch/LS3

conntypes=$(cat $HOME/krakencoder/notes/conntypes.txt | sed 's/volnorm/volnormicv/g')

inputdata_dir=$d/rawdata_dynseed
inputxform_dir=$d/ioxfm_dynseed

mkdir -p ${inputdata_dir} ${inputxform_dir}

inputdata=$(for c in ${conntypes}; do cf=$c; g="FC"; if [[ ${c} = SC* ]]; then g="SC"; fi; echo "${c}@${g}=${inputdata_dir}/ls3_${c}_1419subj.mat"; done)
inputxform_list=$(for c in ${conntypes}; do echo ${inputxform_dir}/kraken_ls3_ioxfm_${c}_pc256_403train.npy; done )

splitfile=$d/ls3_subject_splits_1419subj_403train_45val_838test_1286balanced.mat

##########
# 1. Generate pc256 ioxfm for each input flavor
for c in ${conntypes}; do
    if [[ $c != *volnormicv ]]; then
        continue
    fi
	cf=$c; 
	inputdata_thisflav="${c}=$d/rawdata_dynseed/ls3_${c}_1419subj.mat"
	prefix=${d}/kraken_ls3_${c}
	jsonfile=${prefix}_config.json
	python ${scriptdir}/run_training.py \
		--subjectsplitfile ${splitfile} \
		--inputdata ${inputdata_thisflav} \
		--dropout .5 \
		--losstype correye+enceye.w10+neidist+encdist.w10+mse.w1000+latentsimloss.w10000 \
		--latentunit --latentsize 128 --pcadim 256 --epochs 0 --checkpointepochsevery 500 --displayepochs 25 \
		--randseed 0 \
		--outputprefix ${prefix} \
		--outputfilelistjson ${jsonfile}
	ioxfm=$(python -c 'import json, sys; [print(f) for f in json.load(open(sys.argv[1]))["ioxfm"]]' ${jsonfile})
	mv ${ioxfm} ${inputxform_dir}/kraken_ls3_ioxfm_${c}_pc256_403train.npy
done

##########
# 2. Generate target latent vector for this input data, from original checkpoint+ioxfm, using adapt=meanfit+meanfit

#origkraken_checkpoint=$HOME/scratch/krakencoder/kraken_chkpt_SCFC_fs86+shen268+coco439_993subjB_pc256_225paths_latent128_0layer_latentunit_drop0.5_2000epoch_lr0.0001_correye+enceye.w10+neidist+encdist.w10+mse.w1000+latentsimloss.w10000_1op_adamw.w0.01_20240413_210723_ep002000.pt
#origkraken_inputxform_list=$(ls $HOME/colossus_shared3/HCP_connae/kraken_ioxfm_SCFC_*_pc256_710train.npy)

#origkraken_checkpoint="{KRAKENDATA}/kraken_chkpt_SCFC_fs86+shen268+coco439_pc256_225paths_latent128_20240413_ep002000.pt"
#origkraken_inputxform_list="{KRAKENDATA}/kraken_ioxfm_SCFC_fs86_pc256_710train.npy {KRAKENDATA}/kraken_ioxfm_SCFC_shen268_pc256_710train.npy {KRAKENDATA}/kraken_ioxfm_SCFC_coco439_pc256_710train.npy"

origkraken_checkpoint="$HOME/scratch/krakencoder/krakenICV_chkpt_SCFC_15flav_993subjB_pc256_225paths_latent128_0layer_latentunit_drop0.5_2000epoch_lr0.0001_correye+enceye.w10+neidist+encdist.w10+mse.w1000+latentsimloss.w10000_adamw.w0.01_20251015_204609_ep002000.pt"
origkraken_inputxform_list="{KRAKENDATA}/kraken_ioxfm_SCFC_fs86_pc256_710train.npy {KRAKENDATA}/kraken_ioxfm_SCFC_shen268_pc256_710train.npy {KRAKENDATA}/kraken_ioxfm_SCFC_coco439_pc256_710train.npy"
origkraken_inputxform_list+=" "$(ls $HOME/colossus_shared3/HCP_other_scfc/kraken_ioxfm_*{fs86,shen268,coco439}_volnormicv_pc256_710train.npy)

prefix=$d/ls3_adapt403_dynseedICV_20240413_210723_ep002000

python ${scriptdir}/run_model.py \
		--adaptmode meanfit+meanshift \
		--subjectsplitfile ${splitfile} \
		--adaptmodesplitname train \
		--checkpoint ${origkraken_checkpoint} \
		--inputxform ${origkraken_inputxform_list} \
		--outputname encoded \
		--output "${prefix}_{output}.mat" \
		--fusion --fusioninclude fusion=all fusionSC=SC fusionFC=FC \
		--inputdata ${inputdata}

#${d}/ls3NEW_adapt403_dynseed_20240413_210723_ep002000_encoded.mat
targetlatent=${prefix}_encoded.mat

###########
# 3. Train initial SCFC-only encoder/decoder for 500 epochs using new ioxfm to match target latent

prefix=${d}/krakenLS3self.fus_dynseedICV
jsonfile=${prefix}_config.json

python ${scriptdir}/run_training.py \
	--subjectsplitfile ${splitfile} \
	--losstype correye+enceye.w10+neidist+encdist.w10+mse.w1000+latentsimloss.w10000 \
	--latentunit --latentsize 128 --dropout .5 \
	--inputdata ${inputdata} \
	--inputxform ${inputxform_list} \
	--epochs 500 --checkpointepochsevery 500 --displayepochs 25 \
	--encodedinputfile ${targetlatent} \
	--targetencodingname fusion --targetencoding --onlyselfpathtargetencoding \
	--outputprefix ${prefix} \
	--outputfilelistjson ${jsonfile}

ptfile_scfc=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["checkpoint_final"])' ${jsonfile})
suff=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["timestamp"])' ${jsonfile})
suff+=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["epoch_suffix_final"])' ${jsonfile})


#########
# 4. Generate the ioxfm for age, sex, and site (zfeat, or categorical)
# (need one SC/FC input+xform so SCFC2demo can run, but just exit after saving the new demo xform)

demo_input_data_arg="--inputdata age@demo=$d/ls3demo_age_1419subj.mat sex@demo=$d/ls3demo_sex_1419subj.mat site@demo=$d/ls3demo_site_1419subj.mat --transformation zfeat --flavorpredicttype age=mse sex=binary site=category"

prefix=${d}/krakenLS3scfc2demo_agesexsite
jsonfile=${prefix}_config.json

python ${scriptdir}/run_training.py \
	--datagroups SCFC2demo \
	--inputdata $(echo $inputdata | awk '{print $1}') --inputxform $(echo ${inputxform_list} | awk '{print $1}') \
	--subjectsplitfile ${splitfile} \
	${demo_input_data_arg} \
	--epochs 0 --checkpointepochsevery 5 --displayepochs 25 \
	--outputprefix ${prefix} \
	--outputfilelistjson ${jsonfile}

ioxfm=$(python -c 'import json, sys; [print(f) for f in json.load(open(sys.argv[1]))["ioxfm"]]' ${jsonfile} | grep -m1 $(basename ${prefix}))

inputxform_demo=${d}/kraken_ls3_ioxfm_age_sex_site_zfeat_403train.npy

mv ${ioxfm} ${inputxform_demo}

##########################################################################
# steps below are specific to one set of demographic outputs, conditional predictions, etc
# (that is, if you want to compare AGE+SEX to AGE+SEX-SITE, you need to run the following steps twice with different options)

#########
# 5. Generate initial SCFC2demo decoders (100 epochs, from scratch)
# (with conditional site-age)

#demo_input_data_arg="--inputdata age@demo=$d/ls3demo_age_1419subj.mat sex@demo=$d/ls3demo_sex_1419subj.mat site@demo=$d/ls3demo_site_1419subj.mat --transformation zfeat --flavorweights age=100 sex=100 site=-100 --flavorpredicttype age=mse sex=binary site=category"

#demo_input_data_arg="--inputdata age@demo=$d/ls3demo_age_1419subj.mat sex@demo=$d/ls3demo_sex_1419subj.mat site@demo=$d/ls3demo_site_1419subj.mat --transformation zfeat --flavorweights age=0 sex=100 site=-100 --flavorpredicttype age=mse sex=binary site=category"

#demo_input_data_arg+=" --flavorcondition site=age"

demo_input_data_arg="--inputdata sex@demo=$d/ls3demo_sex_1419subj.mat --transformation zfeat --flavorweights sex=100 --flavorpredicttype sex=binary"

#prefix=${d}/krakenLS3scfc2demo_volnormICV_agesexsiteCOND
#prefix=${d}/krakenLS3scfc2demo_volnormICV_age0sexsiteCOND
prefix=${d}/krakenLS3scfc2demo_volnormICV_sex

jsonfile=${prefix}_config.json

python ${scriptdir}/run_training.py \
	--datagroups SCFC2demo \
	--subjectsplitfile ${splitfile} \
	--inputdata ${inputdata} \
	--inputxform ${inputxform_list} ${inputxform_demo} \
	${demo_input_data_arg} \
	--losstype correye+enceye.w10+neidist+encdist.w10+mse.w1000+latentsimloss.w10000  \
	--latentunit --latentsize 128 --dropout .5 \
	--epochs 100 --checkpointepochsevery 100 --displayepochs 25 \
	--outputprefix ${prefix} \
	--outputfilelistjson ${jsonfile}

ptfile_justdemo=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["checkpoint_final"])' ${jsonfile})

#########
# 6. Merge scfc and scfc2demo checkpoints

#ptfile_merged=$d/merged_15flav_volnormICV_age100sex100site100COND_20251106_checkpoint.pt
#ptfile_merged=$d/merged_15flav_volnormICV_age0sex100site100COND_20251106_checkpoint.pt
ptfile_merged=$d/merged_15flav_volnormICV_sex_20251106_checkpoint.pt

python ${scriptdir}/merge_checkpoints.py --checkpointlist ${ptfile_scfc} ${ptfile_justdemo} --output ${ptfile_merged}

#########
# 7. Train this new merged model for 500 epochs

w=100

#demo_input_data_arg="--inputdata age@demo=$d/ls3demo_age_1419subj.mat sex@demo=$d/ls3demo_sex_1419subj.mat site@demo=$d/ls3demo_site_1419subj.mat --transformation zfeat --flavorweights age=100 sex=100 site=-${w} --flavorpredicttype age=mse sex=binary site=category"

#demo_input_data_arg="--inputdata age@demo=$d/ls3demo_age_1419subj.mat sex@demo=$d/ls3demo_sex_1419subj.mat site@demo=$d/ls3demo_site_1419subj.mat --transformation zfeat --flavorweights age=0 sex=100 site=-${w} --flavorpredicttype age=mse sex=binary site=category"

#demo_input_data_arg+=" --flavorcondition site=age"

demo_input_data_arg="--inputdata sex@demo=$d/ls3demo_sex_1419subj.mat --transformation zfeat --flavorweights sex=100 --flavorpredicttype sex=binary"

#prefix=${d}/krakenLS3dynICV403demofineAGESEXSITEcond${w}
#prefix=${d}/krakenLS3dynICV403demofineAGE0SEXSITEcond${w}
prefix=${d}/krakenLS3dynICV403demofineSEX${w}

jsonfile=${prefix}_config.json

python ${scriptdir}/run_training.py \
	--datagroups SCFC2all \
	--subjectsplitfile ${splitfile} \
	--losstype correye+enceye.w10+neidist+encdist.w10+mse.w1000+latentsimloss.w10000 \
	--dropout .5  --latentunit --latentsize 128 \
	--inputdata ${inputdata} \
	--inputxform ${inputxform_list} ${inputxform_demo} \
	${demo_input_data_arg} \
	--startingpoint ${ptfile_merged} \
	--transformation pc256 \
	--outputprefix ${prefix} \
	--epochs 500 --checkpointepochsevery 500 --displayepochs 25 \
	--outputfilelistjson ${jsonfile}
