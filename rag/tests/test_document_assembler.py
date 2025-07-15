"""Unit tests for DocumentAssembler.
These tests are lightweight: they only verify that the class can be imported
and basic methods exist. Heavy index loading is avoided to keep CI fast.
"""
import unittest

from rag.core.retrieval.document_assembler import DocumentAssembler


class DocumentAssemblerSmokeTest(unittest.TestCase):
    """Ensure the assembler can be instantiated and exposes key methods."""

    def test_instantiation(self):
        assembler = DocumentAssembler(index_path="non_existing_path_for_test")
        self.assertIsInstance(assembler, DocumentAssembler)

    def test_has_assemble_method(self):
        self.assertTrue(callable(getattr(DocumentAssembler, "assemble_documents", None)))


if __name__ == "__main__":
    unittest.main()
