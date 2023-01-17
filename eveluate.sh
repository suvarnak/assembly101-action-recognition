conda activate assembly101_env

cd TSM-action-recognition
python test_models.py Assembly101 --weights=pretrained_models/TSM_Assembly101_combined_resnet50_shift8_blockres_avg_segment8_e50.pth.tar --test_segments 8 --batch_size 64 -j 16 --test_crops 1