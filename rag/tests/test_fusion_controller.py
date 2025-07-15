"""Unit tests for Fusion Controller."""
import unittest

from rag.core.retrieval.fusion_controller import create_default_controller


class FusionControllerSmokeTest(unittest.TestCase):
    def test_create_default(self):
        controller = create_default_controller()
        self.assertIsNotNone(controller)

    def test_search_from_empty_query(self):
        controller = create_default_controller()
        result = controller.search_from_preprocessed_query({}, top_k=1)
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
