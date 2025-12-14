class TwoStageRegressor:
    def __init__(self, stage1, stage2):
        self.stage1 = stage1
        self.stage2 = stage2

    def predict(self, X):
        stage1_pred = self.stage1.predict(X)
        return self.stage2.predict(stage1_pred.reshape(-1, 1))
