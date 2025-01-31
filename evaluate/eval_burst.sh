cd third_party/burst_tools
# first convert ytvis format results to TAO/BURST results
python3 convert_ytvis2tao.py  --results ../../exp/Final_Stage2_baseline_new_enc3_dec3_evaclipl_64g_L2/inference/results.json  --refer ../../datasets/TAO/burst_annotations/val/all_classes.json 
cd BURST-benchmark
# pip3 install -e .
export TRACKEVAL_DIR=../TrackEval/
python3 burstapi/eval/run.py --pred ../converted_tao_results.json   --gt ../../../datasets/TAO/burst_annotations/val/  --task class_guided