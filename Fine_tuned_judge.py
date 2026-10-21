"""Inference adapter for the fine-tuned binary DistilKoBERT student.

The checkpoint is a PyTorch ``state_dict`` trained with the custom
``DistilKoBERTClassifier`` architecture from the distillation project. A long
dialogue is split only at turn boundaries and P(voice_phishing) is aggregated
from the two strongest chunks. No rubric prompt is used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


STUDENT_HF_ID = "monologg/distilkobert"
STUDENT_HF_REVISION = "4a1fd77263770962fa9d846fc672da47f4da9eaa"
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_MANIFEST_PATH = SCRIPT_DIR / "model_manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_checkpoint_sha256(path: Path) -> str:
    manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    student = manifest["realism_student"]
    published_path = (SCRIPT_DIR / student["path"]).resolve()
    return str(student["sha256"]) if path.resolve() == published_path else ""


class DistilKoBERTJudge:
    """Load a DistilKoBERT state_dict and return P(voice-phishing)."""

    backend = "distilkobert_student"

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = "",
        device: str = "auto",
        max_length: int = 512,
        threshold: float = 0.5,
        base_model: str = STUDENT_HF_ID,
        base_revision: str = STUDENT_HF_REVISION,
    ) -> None:
        if not model_path.strip():
            raise ValueError("현실성 평가에는 DISTILKOBERT_MODEL_PATH가 필요합니다.")
        path = Path(model_path).expanduser()
        if not path.is_absolute():
            path = SCRIPT_DIR / path
        if not path.is_file():
            raise FileNotFoundError(f"DistilKoBERT checkpoint를 찾을 수 없습니다: {path}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("DistilKoBERT threshold는 0과 1 사이여야 합니다.")

        self.model_path = str(path.resolve())
        self.checkpoint_sha256 = file_sha256(path)
        expected_sha256 = expected_checkpoint_sha256(path)
        if expected_sha256 and self.checkpoint_sha256 != expected_sha256:
            raise ValueError(
                "DistilKoBERT checkpoint SHA-256이 model_manifest.json과 일치하지 않습니다."
            )
        try:
            self._model_name = str(path.resolve().relative_to(SCRIPT_DIR))
        except ValueError:
            self._model_name = self.model_path
        self.tokenizer_path = tokenizer_path or base_model
        self.base_model = base_model
        self.base_revision = base_revision
        self.device_name = device
        self.max_length = max_length
        self.threshold = threshold
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            import torch.nn as nn
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "DistilKoBERT 추론 의존성이 없습니다. `pip install -r requirements.txt`를 실행하세요."
            ) from exc

        if self.device_name == "auto":
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = self.device_name

        class DistilKoBERTClassifier(nn.Module):
            def __init__(self, encoder: Any) -> None:
                super().__init__()
                self.encoder = encoder
                self.classifier = nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(int(encoder.config.hidden_size), 2),
                )

            def forward(self, input_ids: Any, attention_mask: Any) -> Any:
                output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                return self.classifier(output.last_hidden_state[:, 0, :])

        self._torch = torch
        self._device = torch.device(resolved_device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True,
            revision=self.base_revision,
        )
        encoder = AutoModel.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            revision=self.base_revision,
        )
        model = DistilKoBERTClassifier(encoder)
        state = torch.load(self.model_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("DistilKoBERT checkpoint는 state_dict 형식이어야 합니다.")
        model.load_state_dict(state, strict=True)
        model.to(self._device)
        model.eval()
        self._model = model

    def _predict_text(self, text: str) -> dict[str, Any]:
        self._load()
        if not text.strip():
            raise ValueError("DistilKoBERT가 판정할 대화가 비어 있습니다.")

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            logits = self._model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )[0]

        probabilities = self._torch.softmax(logits, dim=-1).detach().cpu().tolist()
        normal_probability = float(probabilities[0])
        vp_probability = float(probabilities[1])
        prediction = int(vp_probability >= self.threshold)
        confidence = vp_probability if prediction == 1 else normal_probability
        return {
            "backend": self.backend,
            "model": self.model_name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "base_model": self.base_model,
            "base_revision": self.base_revision,
            "prediction": prediction,
            "label": "voice_phishing" if prediction else "normal",
            "vp_probability": round(vp_probability, 6),
            "confidence": round(confidence, 6),
            "probabilities": {
                "normal": round(normal_probability, 6),
                "voice_phishing": round(vp_probability, 6),
            },
            "threshold": self.threshold,
            "prompt_used": False,
        }

    def predict(self, scenario: str) -> dict[str, Any]:
        return self._predict_text(scenario)

    def _dialogue_chunks(self, dialogue: list[dict[str, Any]], overlap_turns: int = 3) -> list[dict[str, Any]]:
        self._load()
        lines = [f"{turn['speaker']}: {turn['text']}" for turn in dialogue]
        if not lines:
            raise ValueError("DistilKoBERT가 판정할 dialogue가 비어 있습니다.")

        max_content_tokens = max(1, self.max_length - 2)
        chunks = []
        start = 0
        while start < len(lines):
            end = start
            text = ""
            while end < len(lines):
                candidate = "\n".join(lines[start:end + 1])
                token_count = len(
                    self._tokenizer(candidate, add_special_tokens=False, verbose=False)["input_ids"]
                )
                if token_count > max_content_tokens and end > start:
                    break
                text = candidate
                end += 1
                if token_count > max_content_tokens:
                    break

            chunks.append({
                "start_turn": start + 1,
                "end_turn": end,
                "text": text,
            })
            if end >= len(lines):
                break
            start = max(start + 1, end - overlap_turns)
        return chunks

    def predict_dialogue(self, dialogue: list[dict[str, Any]]) -> dict[str, Any]:
        chunks = self._dialogue_chunks(dialogue)
        judgments = [self._predict_text(chunk["text"]) for chunk in chunks]
        probabilities = [float(item["vp_probability"]) for item in judgments]
        top = sorted(probabilities, reverse=True)[:2]
        vp_probability = sum(top) / len(top)
        prediction = int(vp_probability >= self.threshold)
        confidence = vp_probability if prediction else 1.0 - vp_probability

        result = dict(judgments[0])
        result.update({
            "prediction": prediction,
            "label": "voice_phishing" if prediction else "normal",
            "vp_probability": round(vp_probability, 6),
            "confidence": round(confidence, 6),
            "probabilities": {
                "normal": round(1.0 - vp_probability, 6),
                "voice_phishing": round(vp_probability, 6),
            },
            "aggregation": "top_2_mean",
            "chunk_probabilities": [round(value, 6) for value in probabilities],
            "chunks": [
                {
                    "start_turn": chunk["start_turn"],
                    "end_turn": chunk["end_turn"],
                }
                for chunk in chunks
            ],
        })
        return result


# Backward-compatible import name used by older scripts in this repository.
DistilBertJudge = DistilKoBERTJudge
