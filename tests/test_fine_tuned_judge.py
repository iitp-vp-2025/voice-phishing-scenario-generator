import unittest

from Fine_tuned_judge import DistilKoBERTJudge


class FineTunedDialogueTest(unittest.TestCase):
    def test_dialogue_chunks_keep_turn_boundaries(self) -> None:
        judge = object.__new__(DistilKoBERTJudge)
        judge.max_length = 6
        judge._load = lambda: None
        judge._tokenizer = lambda text, **_: {"input_ids": text.split()}
        dialogue = [
            {"speaker": "A", "text": f"발화{i}"}
            for i in range(1, 6)
        ]

        chunks = judge._dialogue_chunks(dialogue, overlap_turns=1)

        self.assertEqual(
            [(chunk["start_turn"], chunk["end_turn"]) for chunk in chunks],
            [(1, 2), (2, 3), (3, 4), (4, 5)],
        )
        self.assertTrue(all("A: 발화" in chunk["text"] for chunk in chunks))

    def test_dialogue_probability_uses_top_two_mean(self) -> None:
        judge = object.__new__(DistilKoBERTJudge)
        judge.threshold = 0.5
        judge._dialogue_chunks = lambda _: [
            {"start_turn": 1, "end_turn": 10, "text": "one"},
            {"start_turn": 8, "end_turn": 20, "text": "two"},
            {"start_turn": 18, "end_turn": 30, "text": "three"},
        ]
        probabilities = iter((0.1, 0.9, 0.7))

        def predict(_: str) -> dict:
            probability = next(probabilities)
            return {
                "prediction": int(probability >= 0.5),
                "label": "voice_phishing" if probability >= 0.5 else "normal",
                "vp_probability": probability,
                "confidence": max(probability, 1 - probability),
            }

        judge._predict_text = predict
        result = judge.predict_dialogue([{"speaker": "A", "text": "발화"}])

        self.assertEqual(result["vp_probability"], 0.8)
        self.assertEqual(result["prediction"], 1)
        self.assertEqual(result["chunk_probabilities"], [0.1, 0.9, 0.7])
        self.assertEqual(result["aggregation"], "top_2_mean")


if __name__ == "__main__":
    unittest.main()
