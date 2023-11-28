from loso_data_preprocessing import (
    load_eeg_data,
    preprocess_data,
    split_data,
    apply_tsne,
)
from model_definition import EEGNet, EEG_LSTM, Simple1DCNN, TCNN
from loso_train_eval import train_model
from test import test_model
import numpy as np
import mne.preprocessing as mne


def main():
    subjects = [
        "./data/subject_1.set",
        "./data/subject_2.set",
        "./data/subject_3.set",
        "./data/subject_4.set",
        "./data/subject_5.set",
    ]

    for test_subject in subjects:
        # 데이터 로딩 및 전처리
        train_data = []
        train_labels = []
        for subject in subjects:
            if subject != test_subject:
                raw = load_eeg_data(subject)
                data, labels = preprocess_data(raw)
                train_data.append(data)
                train_labels.append(labels)

        # 훈련 데이터 병합
        X_train, y_train = merge_train_data(train_data, train_labels)

        # 테스트 데이터
        raw = load_eeg_data(test_subject)
        X_test, y_test = preprocess_data(raw)

        # 모델 정의 및 훈련
        num_channels = X_train.shape[1]
        model = Simple1DCNN(num_channels=num_channels, num_classes=12)
        train_model(model, X_train, y_train, 30)

        # 테스트
        test_model(model, X_test, y_test)


def merge_train_data(data_list, label_list):
    # 병합 로직을 여기에 추가
    pass


if __name__ == "__main__":
    main()
