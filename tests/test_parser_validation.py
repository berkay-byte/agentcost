"""Tests for parser token validation (negative, non-numeric, None values)."""

import json
import logging
from pathlib import Path

import pytest

from agentcost.parsers import (
    ClaudeCodeParser,
    CodexParser,
    HermesParser,
    OpenCodeParser,
    _validate_token_count,
)


class TestValidateTokenCount:
    """Unit tests for the _validate_token_count helper."""

    def test_valid_positive_int(self):
        assert _validate_token_count(100, "input_tokens") == 100

    def test_valid_zero(self):
        assert _validate_token_count(0, "input_tokens") == 0

    def test_valid_float_truncated(self):
        assert _validate_token_count(100.7, "input_tokens") == 100

    def test_negative_int_returns_zero(self, caplog):
        assert _validate_token_count(-5, "input_tokens") == 0
        assert "negative" in caplog.text.lower()

    def test_negative_float_returns_zero(self, caplog):
        assert _validate_token_count(-0.5, "input_tokens") == 0
        assert "negative" in caplog.text.lower()

    def test_string_returns_zero(self, caplog):
        assert _validate_token_count("abc", "input_tokens") == 0
        assert "non-numeric" in caplog.text.lower()

    def test_none_returns_zero(self):
        assert _validate_token_count(None, "input_tokens") == 0

    def test_bool_true_returns_zero(self, caplog):
        assert _validate_token_count(True, "input_tokens") == 0
        assert "non-numeric" in caplog.text.lower()

    def test_bool_false_returns_zero(self, caplog):
        assert _validate_token_count(False, "input_tokens") == 0
        assert "non-numeric" in caplog.text.lower()

    def test_inf_returns_zero(self, caplog):
        import math
        assert _validate_token_count(float("inf"), "input_tokens") == 0
        assert "non-finite" in caplog.text.lower()

    def test_neg_inf_returns_zero(self, caplog):
        import math
        assert _validate_token_count(float("-inf"), "input_tokens") == 0
        assert "non-finite" in caplog.text.lower()

    def test_nan_returns_zero(self, caplog):
        import math
        assert _validate_token_count(float("nan"), "input_tokens") == 0
        assert "non-finite" in caplog.text.lower()


class TestClaudeCodeParserValidation:
    """Integration tests for ClaudeCodeParser token validation."""

    @pytest.fixture
    def parser(self):
        return ClaudeCodeParser()

    @pytest.fixture
    def temp_file(self, tmp_path):
        return tmp_path / "test.jsonl"

    def _write_entries(self, path, entries):
        path.write_text("\n".join(json.dumps(e) for e in entries))

    def test_negative_input_tokens_replaced_with_zero(self, parser, temp_file, caplog):
        entries = [{
            "message": {"usage": {"input_tokens": -10, "output_tokens": 50}},
            "timestamp": "2024-01-01T00:00:00Z"
        }]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert len(result) == 1
        assert result[0].input_tokens == 0
        assert result[0].output_tokens == 50
        assert "negative" in caplog.text.lower()

    def test_negative_output_tokens_replaced_with_zero(self, parser, temp_file, caplog):
        entries = [{
            "message": {"usage": {"input_tokens": 100, "output_tokens": -20}},
            "timestamp": "2024-01-01T00:00:00Z"
        }]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].output_tokens == 0
        assert "negative" in caplog.text.lower()

    def test_negative_cache_write_replaced_with_zero(self, parser, temp_file, caplog):
        entries = [{
            "message": {"usage": {"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": -5}},
            "timestamp": "2024-01-01T00:00:00Z"
        }]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].cache_write_tokens == 0
        assert "negative" in caplog.text.lower()

    def test_string_token_replaced_with_zero(self, parser, temp_file, caplog):
        entries = [{
            "message": {"usage": {"input_tokens": "abc", "output_tokens": 50}},
            "timestamp": "2024-01-01T00:00:00Z"
        }]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].input_tokens == 0
        assert "non-numeric" in caplog.text.lower()

    def test_none_token_replaced_with_zero(self, parser, temp_file):
        entries = [{
            "message": {"usage": {"input_tokens": None, "output_tokens": 50}},
            "timestamp": "2024-01-01T00:00:00Z"
        }]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].input_tokens == 0

    def test_inf_token_replaced_with_zero(self, parser, temp_file, caplog):
        entries = [{
            "message": {"usage": {"input_tokens": float("inf"), "output_tokens": 50}},
            "timestamp": "2024-01-01T00:00:00Z"
        }]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].input_tokens == 0
        assert "non-finite" in caplog.text.lower()

    def test_bool_token_replaced_with_zero(self, parser, temp_file, caplog):
        entries = [{
            "message": {"usage": {"input_tokens": True, "output_tokens": 50}},
            "timestamp": "2024-01-01T00:00:00Z"
        }]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].input_tokens == 0
        assert "non-numeric" in caplog.text.lower()

    def test_valid_values_unchanged(self, parser, temp_file):
        entries = [{
            "message": {"usage": {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 25, "cache_creation_input_tokens": 10}},
            "timestamp": "2024-01-01T00:00:00Z"
        }]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].input_tokens == 100
        assert result[0].output_tokens == 50
        assert result[0].cache_read_tokens == 25
        assert result[0].cache_write_tokens == 10


class TestCodexParserValidation:
    """Integration tests for CodexParser token validation."""

    @pytest.fixture
    def parser(self):
        return CodexParser()

    @pytest.fixture
    def temp_file(self, tmp_path):
        return tmp_path / "test.jsonl"

    def _write_entries(self, path, entries):
        path.write_text("\n".join(json.dumps(e) for e in entries))

    def test_negative_input_tokens_replaced_with_zero(self, parser, temp_file, caplog):
        entries = [{"usage": {"input_tokens": -10, "output_tokens": 50}}]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].input_tokens == 0
        assert "negative" in caplog.text.lower()

    def test_string_token_replaced_with_zero(self, parser, temp_file, caplog):
        entries = [{"usage": {"input_tokens": "xyz", "output_tokens": 50}}]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].input_tokens == 0
        assert "non-numeric" in caplog.text.lower()

    def test_none_token_replaced_with_zero(self, parser, temp_file):
        entries = [{"usage": {"input_tokens": None, "output_tokens": 50}}]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].input_tokens == 0

    def test_valid_values_unchanged(self, parser, temp_file):
        entries = [{"usage": {"input_tokens": 200, "output_tokens": 75, "cache_read_input_tokens": 30, "cache_creation_input_tokens": 15}}]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].input_tokens == 200
        assert result[0].output_tokens == 75
        assert result[0].cache_read_tokens == 30
        assert result[0].cache_write_tokens == 15


class TestHermesParserValidation:
    """Integration tests for HermesParser token validation."""

    @pytest.fixture
    def parser(self):
        return HermesParser()

    @pytest.fixture
    def temp_file(self, tmp_path):
        return tmp_path / "test.jsonl"

    def _write_entries(self, path, entries):
        path.write_text("\n".join(json.dumps(e) for e in entries))

    def test_negative_input_tokens_replaced_with_zero(self, parser, temp_file, caplog):
        entries = [{"usage": {"input_tokens": -10, "output_tokens": 50}}]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].input_tokens == 0
        assert "negative" in caplog.text.lower()

    def test_string_token_replaced_with_zero(self, parser, temp_file, caplog):
        entries = [{"usage": {"input_tokens": "bad", "output_tokens": 50}}]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].input_tokens == 0
        assert "non-numeric" in caplog.text.lower()

    def test_none_token_replaced_with_zero(self, parser, temp_file):
        entries = [{"usage": {"input_tokens": None, "output_tokens": 50}}]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].input_tokens == 0

    def test_hermes_tokens_subobject_negative(self, parser, temp_file, caplog):
        entries = [{"tokens": {"input_tokens": -5, "output_tokens": 20}}]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].input_tokens == 0
        assert "negative" in caplog.text.lower()

    def test_valid_values_unchanged(self, parser, temp_file):
        entries = [{"usage": {"input_tokens": 150, "output_tokens": 60, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 8}}]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].input_tokens == 150
        assert result[0].output_tokens == 60
        assert result[0].cache_read_tokens == 20
        assert result[0].cache_write_tokens == 8


class TestOpenCodeParserValidation:
    """Integration tests for OpenCodeParser token validation."""

    @pytest.fixture
    def parser(self):
        return OpenCodeParser()

    @pytest.fixture
    def temp_file(self, tmp_path):
        return tmp_path / "test.jsonl"

    def _write_entries(self, path, entries):
        path.write_text("\n".join(json.dumps(e) for e in entries))

    def test_negative_input_tokens_replaced_with_zero(self, parser, temp_file, caplog):
        entries = [{"usage": {"input_tokens": -10, "output_tokens": 50}}]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].input_tokens == 0
        assert "negative" in caplog.text.lower()

    def test_string_token_replaced_with_zero(self, parser, temp_file, caplog):
        entries = [{"usage": {"input_tokens": "invalid", "output_tokens": 50}}]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].input_tokens == 0
        assert "non-numeric" in caplog.text.lower()

    def test_none_token_replaced_with_zero(self, parser, temp_file):
        entries = [{"usage": {"input_tokens": None, "output_tokens": 50}}]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].input_tokens == 0

    def test_inf_token_replaced_with_zero(self, parser, temp_file, caplog):
        entries = [{"usage": {"input_tokens": float("inf"), "output_tokens": 50}}]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].input_tokens == 0
        assert "non-finite" in caplog.text.lower()

    def test_valid_values_unchanged(self, parser, temp_file):
        entries = [{"usage": {"input_tokens": 175, "output_tokens": 80, "cache_read_input_tokens": 35, "cache_creation_input_tokens": 12}}]
        self._write_entries(temp_file, entries)
        result = parser.parse(temp_file)
        assert result[0].input_tokens == 175
        assert result[0].output_tokens == 80
        assert result[0].cache_read_tokens == 35
        assert result[0].cache_write_tokens == 12