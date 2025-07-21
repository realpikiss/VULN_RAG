"""Unit tests for pipeline orchestrator and Ollama Qwen wrapper."""
import unittest

from rag.core.pipeline.orchestrator import Vuln_RAGRetrievalOrchestrator


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


class QwenWrapperTest(unittest.TestCase):
    def test_to_messages(self):
        from rag.core.generation.ollama_qwen import to_messages
        ctx = "Sample context"
        msgs = to_messages(ctx)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[1]["content"], ctx)

    def test_generate_mock(self):
        """Mock ollama.chat so the unit test is offline."""
        from unittest.mock import patch
        from rag.core.generation import ollama_qwen as oq

        dummy_ctx = "void main() {}"
        fake_resp = {"message": {"content": "SAFE"}}

        with patch("ollama.chat", return_value=fake_resp) as mock_chat:
            out = oq.generate(dummy_ctx, model="qwen:fake")
            mock_chat.assert_called_once()
            self.assertEqual(out, "SAFE")
