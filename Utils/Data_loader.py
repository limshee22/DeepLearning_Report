from d2l import torch as d2l

def load_data(batch_size):
    """바나나 객체 검출 데이터셋을 로드합니다."""
    train_iter, val_iter = d2l.load_data_bananas(batch_size)
    return train_iter, val_iter




