class PredictionHandler:
    def __init__(self):
        self._pred = None

    def __call__(self, pred):
        if pred == self._pred:
            return
        self._pred = pred
        match self._pred:
            case "hand_raise":
                pass
            case "call_simple":
                pass
