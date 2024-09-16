# https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch/71999355#71999355
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        # 몇 번이나 Update되지 않았을 때 종료할지 (patience이상 count되었으면 조기 종료)
        self.patience = patience 
        # delta 값 이상으로 차이가 나지 않으면 Update되지 않은 것으로 간주
        self.min_delta = min_delta 
        # Update되지 않은 횟수 (일종의 경고 횟수)
        self.counter = 0
        # 가장 작은 loss값
        self.min_validation_loss = float("inf")
        
    def early_stop(self, validation_loss):
        # loss값이 가장 작게 update 되었다면
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        # loss값이 여태까지 가장 작은 loss + delta 보다 크다면
        elif (self.min_validation_loss + self.min_delta) < validation_loss:
            self.counter += 1 # 경고 count
            if self.counter >= self.patience: # 경고한 횟수가 일정 횟수 이상이면 조기 종료
                return True

        # min_validation_loss가 갱신되었거나 delta 범위 안에 있다면 계속 학습
        return False
        