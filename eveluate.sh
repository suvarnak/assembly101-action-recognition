conda activate assembly101_env


# dataset_config.py has configurations for which files to use for validation
# modify line 29 to the txt file that contain file list. For example, TSM-action-recognition/data/custom_validation_combined.txt  
#        filename_imglist_val = f'{path_to_data}/custom_validation_combined.txt'

cd TSM-action-recognition
python test_models.py Assembly101 --weights=pretrained_models/TSM_Assembly101_combined_resnet50_shift8_blockres_avg_segment8_e50.pth.tar --test_segments 8 --batch_size 64 -j 16 --test_crops 1


# results after evaluation are stored in preds.npy
# to see the results
python view_results.py

