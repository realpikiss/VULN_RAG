"""Unit tests for Retrieval Orchestrator."""
import unittest

from rag.core.retrieval.orchestrator import Vuln_RAGRetrievalOrchestrator


class OrchestratorSmokeTest(unittest.TestCase):
    def test_basic_flow(self):
        orch = Vuln_RAGRetrievalOrchestrator()
        dummy_code = "int main() { return 0; }"
        query_data = {
            "kb1_purpose": "return 0", "kb1_function": "main",
            "kb2_vector": [], "kb3_vector": []
        }
        result = orch.retrieve_context(original_code=dummy_code, query_data=query_data)
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, "detection_context"))


if __name__ == "__main__":
    unittest.main()
