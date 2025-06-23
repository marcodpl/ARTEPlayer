
class ConflictResolver:
    def __init__(self):
        self.conflict_gestures = ["thumbs_up", "call_simple"]

    def should_be_resolved(self, gesture):
        return gesture in self.conflict_gestures

    def generic_resolve(self, gesture_label, landmarks, handedness):
        assert gesture_label in self.conflict_gestures
        match gesture_label:
            case "thumbs_up" | "call_simple":
                return self.resolve_thumbs_and_call(landmarks, handedness)
        return None

    @staticmethod
    def resolve_thumbs_and_call(landmarks, handedness) -> str:
        return "call_simple"

