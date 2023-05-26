cd word_detection

python $HOME/version_2_Word_detection_STR_product/rotated_yolov5/detect.py
--source $HOME/Word_detection_STR_product/word_detection/test_STR_images/images/
--weights $HOME/Word_detection_STR_product/word_detection/test_STR_images/weights/img_2560.pt

python $HOME/Word_detection_STR_product/word_detection/yolov5_sample_Signed_v2.py --source $HOME/Word_detection_STR_product/word_detection/test_STR_images/jerry_samples/ --weights $HOME/Word_detection_STR_product/word_detection/test_STR_images/weights/img_2560.pt
cd ..
python stroke_recovery_offline.py


python merge_words_full_page.py
python write_svg_merge_words_full_page