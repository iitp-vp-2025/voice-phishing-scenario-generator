import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import Collect_keyword as collector


class KeywordCurationTest(unittest.TestCase):
    def test_curation_keeps_distinct_keywords_without_llm_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            collector, "OUTPUT_DIR", Path(directory)
        ), patch.object(collector, "call_llm", side_effect=AssertionError("curation must be deterministic")):
            collector.write_lines(
                collector.path_for("Tech", "raw"),
                [
                    "Tech | voice clone",
                    "Tech | Voice Clone",
                    "Finance | biometric wallet",
                    "Tech | account security",
                ],
            )

            curated = collector.curate_axis("Tech")

        self.assertEqual(curated, ["Tech | voice clone", "Finance | biometric wallet"])


if __name__ == "__main__":
    unittest.main()
