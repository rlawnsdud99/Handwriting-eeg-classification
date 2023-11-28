from data_preprocessing import load_eeg_data, preprocess_data, split_data, apply_tsne
from model_definition import EEGNet, EEG_LSTM, Simple1DCNN, TCNN
from train_eval import train_model
from test import test_model
import numpy as np


def main():
    filename = f"./data/subject_1.set"
    # preprocess
    raw = load_eeg_data(filename)
    data, labels = preprocess_data(raw)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, labels)

    # load train / test dataset
    num_channels = X_train.shape[1]  # 데이터에 따라 변경
    model = Simple1DCNN(num_channels=num_channels, num_classes=12)  # 데이터에 따라 변경
    train_model(model, X_train, y_train, X_val, y_val, 30)

    # test
    test_model(model, X_test, y_test)


if __name__ == "__main__":
    main()

# python main.py --train_data_file "230714_gd_jang_aiImage_1.vhdr" --infer_data_file "230714_gd_jang_aiImage_2" --content_type "video"
# 230714_gd_jang_aiImage_1, 230714_gd_jang_aiImage_2, 230714_gd_jang_Image_1, 230714_gd_jang_Image_2, 230814_js_kim_aiImage_1
# gd_jang, js_kim, kc_jeong, hs_oh
