"""Live experimental model adapter; the validation quality guard fails closed.

Quality is an explicit experiment declaration, not independently authenticated.
Unknown native units and acquisition timestamps remain absent from references.
"""
from copy import deepcopy

from ..haconfig.settings import class_labels
from .baseline import TrainingIndex
from .health_models import health_features, nearest_distances, rejection_mask


class StudyTransport:
    def __init__(self, model, training_features, training_records, rejection,
                 *, validation_quality_target_met: bool):
        if type(validation_quality_target_met) is not bool:
            raise ValueError("Quality gate must be an explicit boolean")
        self.model = model
        self.index = TrainingIndex(model, training_features, training_records)
        self.training = model.transform(training_features)
        self.rejection = deepcopy(rejection)
        self.quality_met = validation_quality_target_met
        self.calls = 0

    def __call__(self, method, url, **kwargs):
        self.calls += 1
        if method != "POST":
            raise ValueError("Unexpected study route")
        feature = health_features(kwargs["payload"]["signal"], self.model.variant)
        if url.endswith("/v1/search/text"):
            response = self.index.search(feature, kwargs["payload"]["top_k"])
            for case in response["results"]:
                case["meta"].pop("units", None)
                case["meta"]["preprocessing_id"] = "hust-health-" + self.model.variant + "-v1"
            return "ok", response, ""
        if not url.endswith("/diagnose"):
            raise ValueError("Unexpected study route")
        probability = self.model.probabilities(feature)[0]
        distance = nearest_distances(self.model.transform(feature)[None, :], self.training)[0]
        accepted = bool(rejection_mask(probability[None, :], [distance], self.rejection)[0])
        reason = None
        if not self.quality_met:
            reason = "validation_quality_unmet"
        elif not accepted:
            reason = ("outside_training_support" if self.rejection["distance_max"] is not None
                      and distance > self.rejection["distance_max"] else "below_selection_threshold")
        if reason:
            result = {"abstained": True, "abstention_reason": reason,
                      "predicted_class": None, "confidence": None}
        else:
            label = int(probability.argmax())
            result = {"predicted_class": label, "confidence": float(probability[label]),
                      "predicted_fault": class_labels("bearing-health-v1")[label],
                      "class_probabilities": {str(i): float(p) for i, p in enumerate(probability)}}
        return "ok", {"success": True, "diagnosis": result}, ""
