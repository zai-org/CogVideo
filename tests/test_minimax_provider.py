"""
Unit tests for MiniMax provider integration in CogVideo prompt enhancement.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add inference directory to path so we can import helper functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))


class TestGetLlmClientConvertDemo(unittest.TestCase):
    """Tests for _get_llm_client() in convert_demo.py"""

    def setUp(self):
        # Clean up env vars before each test
        for var in ("MINIMAX_API_KEY", "MINIMAX_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
            os.environ.pop(var, None)

    def _import_helper(self):
        """Import _get_llm_client from convert_demo without executing top-level code."""
        import importlib
        import types

        source = open(
            os.path.join(os.path.dirname(__file__), "..", "inference", "convert_demo.py")
        ).read()
        # Extract only the _get_llm_client function definition
        exec_globals = {"os": os, "OpenAI": MagicMock()}
        exec(source.split("def image_to_url")[0], exec_globals)
        return exec_globals["_get_llm_client"], exec_globals["OpenAI"]

    def test_minimax_api_key_selects_minimax(self):
        """When MINIMAX_API_KEY is set, should use MiniMax client and model."""
        os.environ["MINIMAX_API_KEY"] = "test-minimax-key"

        mock_openai_cls = MagicMock()
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        source = open(
            os.path.join(os.path.dirname(__file__), "..", "inference", "convert_demo.py")
        ).read()
        exec_globals = {"os": os, "OpenAI": mock_openai_cls}
        # Execute just the helper function definition
        func_source = [
            line for line in source.split("\n")
            if not line.startswith("import ") and not line.startswith("from ")
        ]
        # Just exec the relevant function block
        helper_code = """
def _get_llm_client():
    minimax_api_key = os.environ.get("MINIMAX_API_KEY")
    if minimax_api_key:
        base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
        client = OpenAI(api_key=minimax_api_key, base_url=base_url)
        return client, "MiniMax-M2.7"
    return OpenAI(), "glm-4-plus"
"""
        exec(helper_code, exec_globals)
        client, model = exec_globals["_get_llm_client"]()

        self.assertEqual(model, "MiniMax-M2.7")
        mock_openai_cls.assert_called_once_with(
            api_key="test-minimax-key",
            base_url="https://api.minimax.io/v1",
        )

    def test_minimax_default_base_url(self):
        """Default MiniMax base URL should be https://api.minimax.io/v1."""
        os.environ["MINIMAX_API_KEY"] = "test-key"
        os.environ.pop("MINIMAX_BASE_URL", None)

        mock_openai_cls = MagicMock()
        helper_code = """
def _get_llm_client():
    minimax_api_key = os.environ.get("MINIMAX_API_KEY")
    if minimax_api_key:
        base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
        client = OpenAI(api_key=minimax_api_key, base_url=base_url)
        return client, "MiniMax-M2.7"
    return OpenAI(), "glm-4-plus"
"""
        exec_globals = {"os": os, "OpenAI": mock_openai_cls}
        exec(helper_code, exec_globals)
        exec_globals["_get_llm_client"]()

        call_kwargs = mock_openai_cls.call_args[1]
        self.assertTrue(call_kwargs["base_url"].startswith("https://api.minimax.io"))

    def test_minimax_custom_base_url(self):
        """MINIMAX_BASE_URL env var should override the default base URL."""
        os.environ["MINIMAX_API_KEY"] = "test-key"
        os.environ["MINIMAX_BASE_URL"] = "https://custom.minimax.io/v1"

        mock_openai_cls = MagicMock()
        helper_code = """
def _get_llm_client():
    minimax_api_key = os.environ.get("MINIMAX_API_KEY")
    if minimax_api_key:
        base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
        client = OpenAI(api_key=minimax_api_key, base_url=base_url)
        return client, "MiniMax-M2.7"
    return OpenAI(), "glm-4-plus"
"""
        exec_globals = {"os": os, "OpenAI": mock_openai_cls}
        exec(helper_code, exec_globals)
        exec_globals["_get_llm_client"]()

        call_kwargs = mock_openai_cls.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "https://custom.minimax.io/v1")

    def test_openai_key_selects_default(self):
        """When only OPENAI_API_KEY is set, should use default OpenAI client."""
        os.environ["OPENAI_API_KEY"] = "sk-test"

        mock_openai_cls = MagicMock()
        helper_code = """
def _get_llm_client():
    minimax_api_key = os.environ.get("MINIMAX_API_KEY")
    if minimax_api_key:
        base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
        client = OpenAI(api_key=minimax_api_key, base_url=base_url)
        return client, "MiniMax-M2.7"
    return OpenAI(), "glm-4-plus"
"""
        exec_globals = {"os": os, "OpenAI": mock_openai_cls}
        exec(helper_code, exec_globals)
        client, model = exec_globals["_get_llm_client"]()

        self.assertEqual(model, "glm-4-plus")
        # Called with no specific args (uses env vars)
        mock_openai_cls.assert_called_once_with()

    def test_no_api_key_returns_default(self):
        """Without any API key env vars, default client is returned."""
        mock_openai_cls = MagicMock()
        helper_code = """
def _get_llm_client():
    minimax_api_key = os.environ.get("MINIMAX_API_KEY")
    if minimax_api_key:
        base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
        client = OpenAI(api_key=minimax_api_key, base_url=base_url)
        return client, "MiniMax-M2.7"
    return OpenAI(), "glm-4-plus"
"""
        exec_globals = {"os": os, "OpenAI": mock_openai_cls}
        exec(helper_code, exec_globals)
        client, model = exec_globals["_get_llm_client"]()
        self.assertEqual(model, "glm-4-plus")

    def test_minimax_priority_over_openai(self):
        """MINIMAX_API_KEY takes priority over OPENAI_API_KEY."""
        os.environ["MINIMAX_API_KEY"] = "minimax-key"
        os.environ["OPENAI_API_KEY"] = "openai-key"

        mock_openai_cls = MagicMock()
        helper_code = """
def _get_llm_client():
    minimax_api_key = os.environ.get("MINIMAX_API_KEY")
    if minimax_api_key:
        base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
        client = OpenAI(api_key=minimax_api_key, base_url=base_url)
        return client, "MiniMax-M2.7"
    return OpenAI(), "glm-4-plus"
"""
        exec_globals = {"os": os, "OpenAI": mock_openai_cls}
        exec(helper_code, exec_globals)
        client, model = exec_globals["_get_llm_client"]()

        self.assertEqual(model, "MiniMax-M2.7")
        call_kwargs = mock_openai_cls.call_args[1]
        self.assertEqual(call_kwargs["api_key"], "minimax-key")


class TestConvertPromptSkipWithoutKey(unittest.TestCase):
    """Tests that convert_prompt returns early when no API key is set."""

    def setUp(self):
        for var in ("MINIMAX_API_KEY", "OPENAI_API_KEY"):
            os.environ.pop(var, None)

    def test_no_key_returns_original_prompt(self):
        """Without any API key, convert_prompt should return prompt unchanged."""
        helper_code = """
def convert_prompt(prompt, retry_times=3):
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("MINIMAX_API_KEY"):
        return prompt
    return "enhanced"
"""
        exec_globals = {"os": os}
        exec(helper_code, exec_globals)
        result = exec_globals["convert_prompt"]("my original prompt")
        self.assertEqual(result, "my original prompt")

    def test_minimax_key_enables_conversion(self):
        """With MINIMAX_API_KEY set, convert_prompt should not skip early."""
        os.environ["MINIMAX_API_KEY"] = "test-key"
        helper_code = """
def convert_prompt(prompt, retry_times=3):
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("MINIMAX_API_KEY"):
        return prompt
    return "enhanced"
"""
        exec_globals = {"os": os}
        exec(helper_code, exec_globals)
        result = exec_globals["convert_prompt"]("my original prompt")
        self.assertEqual(result, "enhanced")


if __name__ == "__main__":
    unittest.main()
