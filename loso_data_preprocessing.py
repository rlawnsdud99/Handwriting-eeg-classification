import mne
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mne.decoding import CSP


def load_eeg_data(filename):
    raw = mne.io.read_raw_eeglab(filename, preload=True)
    return raw


def preprocess_data(raw):
    raw.filter(l_freq=1, h_freq=45)

    ica = mne.preprocessing.ICA(n_components=10, random_state=97, max_iter=800)
    ica.fit(raw)
    raw.load_data()
    ica.apply(raw)

    raw.filter(l_freq=1, h_freq=8)

    # Annotation에서 이벤트 생성
    events, event_id_from_anno = mne.events_from_annotations(raw)

    # 원하는 이벤트 ID만 선택
    selected_event_ids = {
        key: value
        for key, value in event_id_from_anno.items()
        if key
        in [
            "S  1",
            "S  2",
            "S  3",
            "S  4",
            "S  5",
            "S  6",
            "S  7",
            "S  8",
            "S  9",
            "S 10",
            "S 11",
            "S 12",
        ]
    }
    # "H" "E" "L" "L" "O" "," "W" "O" "R" "L" "D" "!"
    # Epoch 생성
    epoch_tmin, epoch_tmax = -1, 1  # 시작과 끝을 -1s ~ 1s로 지정
    baseline = (-0.3, 0)  # baseline correction을 위한 시간 범위
    epochs = mne.Epochs(
        raw,
        events,
        selected_event_ids,
        tmin=epoch_tmin,
        tmax=epoch_tmax,
        picks="eeg",
        baseline=baseline,
        detrend=1,
        preload=True,
    )

    # 선택적으로 Epoch 데이터를 별도의 변수에 저장
    data = epochs.get_data()

    labels = epochs.events[:, -1]

    # # CSP 적용
    # csp = CSP(n_components=csp_components, transform_into="average_power")
    # data_csp = csp.fit_transform(data, labels)

    # Normalization
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)

    return data, labels


def split_data(data, labels):
    # 셔플하기
    data, labels = shuffle(data, labels, random_state=42)

    # 레이블을 0부터 시작하게 조정
    labels -= 1

    # 각 세트의 크기 출력
    print(f"Training set size: {data.shape[0]}")

    # 레이블 분포 확인
    print("\nLabel distribution in training set:")
    print(np.bincount(labels))

    # 데이터 차원 확인
    print("\nData dimensions:")
    print(f"Training data shape: {data.shape}")

    # 레이블 차원 확인
    print("\nLabel dimensions:")
    print(f"Training label shape: {labels.shape}")

    return data, labels


def apply_tsne(x_data, y_labels, title="t-SNE plot", perplexity=50):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    x_tsne = tsne.fit_transform(x_data.reshape(x_data.shape[0], -1))

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_labels, cmap="viridis")

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    # 라벨 추가
    legend_labels = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=str(i),
            markersize=10,
            markerfacecolor=plt.cm.viridis(i / np.max(y_labels)),
        )
        for i in np.unique(y_labels)
    ]
    plt.legend(handles=legend_labels, title="Labels")

    plt.show()
