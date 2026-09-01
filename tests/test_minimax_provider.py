"""Tests for MiniMax LLM provider in javisdit/utils/inference_utils.py.

Run with: python tests/test_minimax_provider.py
Or: pytest tests/test_minimax_provider.py (requires repo to be installed)
"""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def _stub_heavy_imports():
    """Stub torch and javisdit sub-packages before the module under test is loaded."""
    # Stub torch (36s to import on this machine)
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.device = type("device", (), {})
        torch_stub.dtype = type("dtype", (), {})
        torch_stub.Tensor = type("Tensor", (), {})
        sys.modules["torch"] = torch_stub

    # Stub pandas (8s to import)
    if "pandas" not in sys.modules:
        pd_stub = types.ModuleType("pandas")
        pd_stub.read_csv = MagicMock(return_value=MagicMock(tolist=lambda: []))
        sys.modules["pandas"] = pd_stub

    # Stub all javisdit sub-packages that inference_utils imports
    for mod_name in [
        "javisdit",
        "javisdit.datasets",
        "javisdit.datasets.utils",
        "javisdit.datasets.read_video",
        "javisdit.datasets.read_audio",
    ]:
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            sys.modules[mod_name] = stub

    # Add required attributes
    sys.modules["javisdit.datasets"].IMG_FPS = 24
    sys.modules["javisdit.datasets.utils"].read_from_path = MagicMock()
    sys.modules["javisdit.datasets.utils"].get_transforms_video = MagicMock()
    sys.modules["javisdit.datasets.utils"].get_transforms_audio = MagicMock()
    sys.modules["javisdit.datasets.read_video"].read_video = MagicMock()
    sys.modules["javisdit.datasets.read_audio"].read_audio = MagicMock()

    # Stub javisdit.utils as a real package (not a stub) so the module loads
    if "javisdit.utils" not in sys.modules:
        util_pkg = types.ModuleType("javisdit.utils")
        sys.modules["javisdit.utils"] = util_pkg


_stub_heavy_imports()

# Add repo root to sys.path so we can import javisdit directly
import importlib.util
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Load inference_utils from file to bypass any package __init__ issues
_spec = importlib.util.spec_from_file_location(
    "javisdit.utils.inference_utils",
    os.path.join(_REPO_ROOT, "javisdit", "utils", "inference_utils.py"),
)
_iu = importlib.util.module_from_spec(_spec)
sys.modules["javisdit.utils.inference_utils"] = _iu
_spec.loader.exec_module(_iu)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestHasMinimaxKey(unittest.TestCase):
    def test_returns_true_when_key_set(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"}):
            self.assertTrue(_iu.has_minimax_key())

    def test_returns_false_when_key_absent(self):
        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(_iu.has_minimax_key())


class TestHasOpenaiKey(unittest.TestCase):
    def test_returns_true_when_key_set(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            self.assertTrue(_iu.has_openai_key())

    def test_returns_false_when_key_absent(self):
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(_iu.has_openai_key())


class TestMinimaxConstants(unittest.TestCase):
    def test_api_base_url(self):
        self.assertEqual(_iu.MINIMAX_API_BASE, "https://api.minimax.io/v1")

    def test_default_model_is_m27(self):
        self.assertIn("M2.7", _iu.MINIMAX_DEFAULT_MODEL)


class TestGetMinimaxResponse(unittest.TestCase):
    def _make_mock_client(self, content="Response text."):
        mock_msg = MagicMock()
        mock_msg.content = content
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        return mock_client

    def setUp(self):
        _iu.MINIMAX_CLIENT = None

    def test_returns_content_string(self):
        mock_client = self._make_mock_client("A serene mountain lake at sunset.")
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "key"}):
            with patch("openai.OpenAI", return_value=mock_client):
                _iu.MINIMAX_CLIENT = None
                result = _iu.get_minimax_response("sys", "usr")
        self.assertEqual(result, "A serene mountain lake at sunset.")

    def test_uses_minimax_api_base_url(self):
        mock_client = self._make_mock_client()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "key123"}):
            with patch("openai.OpenAI", return_value=mock_client) as mock_cls:
                _iu.MINIMAX_CLIENT = None
                _iu.get_minimax_response("sys", "usr")
        mock_cls.assert_called_once_with(api_key="key123", base_url="https://api.minimax.io/v1")

    def test_uses_default_m27_model(self):
        mock_client = self._make_mock_client()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "key"}):
            with patch("openai.OpenAI", return_value=mock_client):
                _iu.MINIMAX_CLIENT = None
                _iu.get_minimax_response("sys", "usr")
        kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(kwargs["model"], "MiniMax-M2.7")

    def test_temperature_in_valid_range(self):
        """MiniMax requires temperature in (0.0, 1.0]."""
        mock_client = self._make_mock_client()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "key"}):
            with patch("openai.OpenAI", return_value=mock_client):
                _iu.MINIMAX_CLIENT = None
                _iu.get_minimax_response("sys", "usr")
        kwargs = mock_client.chat.completions.create.call_args[1]
        temp = kwargs["temperature"]
        self.assertGreater(temp, 0.0)
        self.assertLessEqual(temp, 1.0)

    def test_passes_system_and_user_messages(self):
        mock_client = self._make_mock_client()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "key"}):
            with patch("openai.OpenAI", return_value=mock_client):
                _iu.MINIMAX_CLIENT = None
                _iu.get_minimax_response("SYSTEM PROMPT", "USER PROMPT")
        kwargs = mock_client.chat.completions.create.call_args[1]
        msgs = kwargs["messages"]
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "SYSTEM PROMPT")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[1]["content"], "USER PROMPT")


class TestGetLlmResponse(unittest.TestCase):
    def test_prefers_minimax_over_openai(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "m", "OPENAI_API_KEY": "o"}):
            with patch.object(_iu, "get_minimax_response", return_value="minimax") as mm:
                with patch.object(_iu, "get_openai_response", return_value="openai") as oo:
                    result = _iu.get_llm_response("sys", "usr")
        self.assertEqual(result, "minimax")
        mm.assert_called_once()
        oo.assert_not_called()

    def test_uses_openai_when_no_minimax_key(self):
        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
        env["OPENAI_API_KEY"] = "oai"
        with patch.dict(os.environ, env, clear=True):
            with patch.object(_iu, "get_minimax_response", return_value="minimax") as mm:
                with patch.object(_iu, "get_openai_response", return_value="openai") as oo:
                    result = _iu.get_llm_response("sys", "usr")
        self.assertEqual(result, "openai")
        oo.assert_called_once()
        mm.assert_not_called()


class TestRefinePromptByLlm(unittest.TestCase):
    def setUp(self):
        _iu.REFINE_PROMPTS = "system prompt template"

    def test_returns_llm_response(self):
        with patch.object(_iu, "get_llm_response", return_value="refined"):
            result = _iu.refine_prompt_by_llm("raw prompt")
        self.assertEqual(result, "refined")

    def test_passes_user_prompt(self):
        with patch.object(_iu, "get_llm_response", return_value="ok") as mock_llm:
            _iu.refine_prompt_by_llm("my input")
        self.assertEqual(mock_llm.call_args[0][1], "my input")


class TestGetRandomPromptByLlm(unittest.TestCase):
    def setUp(self):
        _iu.RANDOM_PROMPTS = "random system template"

    def test_returns_llm_response(self):
        with patch.object(_iu, "get_llm_response", return_value="random prompt"):
            result = _iu.get_random_prompt_by_llm()
        self.assertEqual(result, "random prompt")

    def test_sends_generate_one_example(self):
        with patch.object(_iu, "get_llm_response", return_value="ok") as mock_llm:
            _iu.get_random_prompt_by_llm()
        self.assertEqual(mock_llm.call_args[0][1], "Generate one example.")


class TestRefinePromptsByLlm(unittest.TestCase):
    def setUp(self):
        _iu.REFINE_PROMPTS = "system prompt"
        _iu.RANDOM_PROMPTS = "random system"

    def test_refines_multiple_prompts(self):
        with patch.object(_iu, "get_llm_response", side_effect=["out1", "out2"]):
            results = _iu.refine_prompts_by_llm(["p1", "p2"])
        self.assertEqual(results, ["out1", "out2"])

    def test_empty_prompt_calls_random(self):
        with patch.object(_iu, "get_llm_response", return_value="random") as mock_llm:
            results = _iu.refine_prompts_by_llm([""])
        self.assertEqual(results, ["random"])
        # Ensure "Generate one example" was the user message
        call_usr = mock_llm.call_args[0][1]
        self.assertEqual(call_usr, "Generate one example.")

    def test_exception_returns_original_prompt(self):
        with patch.object(_iu, "get_llm_response", side_effect=Exception("error")):
            results = _iu.refine_prompts_by_llm(["original"])
        self.assertEqual(results, ["original"])

    def test_logs_minimax_label_when_key_present(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "key"}):
            with patch.object(_iu, "get_llm_response", return_value="refined"):
                with patch("builtins.print") as mock_print:
                    _iu.refine_prompts_by_llm(["test"])
        output = " ".join(str(a) for call in mock_print.call_args_list for a in call[0])
        self.assertIn("MiniMax", output)

    def test_logs_openai_label_when_no_minimax(self):
        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(_iu, "get_llm_response", return_value="refined"):
                with patch("builtins.print") as mock_print:
                    _iu.refine_prompts_by_llm(["test"])
        output = " ".join(str(a) for call in mock_print.call_args_list for a in call[0])
        self.assertIn("OpenAI", output)


# ---------------------------------------------------------------------------
# Integration-style tests
# ---------------------------------------------------------------------------

class TestIntegrationClientReuse(unittest.TestCase):
    def _make_mock_client(self, content="result"):
        mock_msg = MagicMock()
        mock_msg.content = content
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        return mock_client

    def test_minimax_client_instantiated_once(self):
        mock_client = self._make_mock_client()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"}):
            with patch("openai.OpenAI", return_value=mock_client) as mock_cls:
                _iu.MINIMAX_CLIENT = None
                _iu.get_minimax_response("s", "u")
                _iu.get_minimax_response("s2", "u2")
        self.assertEqual(mock_cls.call_count, 1)
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    def test_refine_prompts_by_llm_uses_minimax_end_to_end(self):
        mock_client = self._make_mock_client("Refined: waves at sunset on the coast.")
        env = {k: v for k, v in os.environ.items() if k not in ("MINIMAX_API_KEY", "OPENAI_API_KEY")}
        env["MINIMAX_API_KEY"] = "test_key"
        with patch.dict(os.environ, env, clear=True):
            with patch("openai.OpenAI", return_value=mock_client):
                _iu.MINIMAX_CLIENT = None
                _iu.REFINE_PROMPTS = "refine system"
                results = _iu.refine_prompts_by_llm(["sunset"])
        self.assertEqual(results[0], "Refined: waves at sunset on the coast.")

    def test_refine_prompts_by_llm_uses_openai_end_to_end(self):
        mock_client = self._make_mock_client("Refined: forest in morning mist.")
        env = {k: v for k, v in os.environ.items() if k not in ("MINIMAX_API_KEY", "OPENAI_API_KEY")}
        env["OPENAI_API_KEY"] = "oai_key"
        with patch.dict(os.environ, env, clear=True):
            with patch("openai.OpenAI", return_value=mock_client):
                _iu.OPENAI_CLIENT = None
                _iu.REFINE_PROMPTS = "refine system"
                results = _iu.refine_prompts_by_llm(["forest"])
        self.assertEqual(results[0], "Refined: forest in morning mist.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
