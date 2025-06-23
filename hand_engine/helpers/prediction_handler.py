class PredictionHandler:
    def __init__(self):
        self._pred = None
        self._last_pred = None

    @property
    def pred(self):
        return self._pred

    @pred.setter
    def pred(self, pred):
        self._last_pred = self._pred
        self._pred = pred
        self.on_pred_update()

    def on_pred_update(self):
        match self._pred:
            case _:
                pass
