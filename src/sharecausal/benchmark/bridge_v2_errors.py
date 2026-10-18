"""Allowlisted CLI diagnostics: never print exception messages, paths or locals."""
from contextlib import contextmanager
import errno
from importlib.metadata import PackageNotFoundError


CONFIG_ERRORS = {
    'results_not_empty': (
        'The results directory already contains files; prior outputs were preserved.',
        'Repeat with --new-run to create a fresh child directory, or choose a new/empty --results-dir.'),
    'results_not_directory': (
        'The results destination exists but is not a directory.',
        'Choose a directory for --results-dir; no existing file will be replaced.'),
    'fixture_directory_missing': (
        'The fixture directory does not exist or is not a directory.',
        'Run from the repository root with --fixture-dir tests/fixtures/llm_bridge_v2.'),
    'fixture_directory_empty': (
        'The fixture directory contains no JSON records.',
        'Use the repository fixture directory tests/fixtures/llm_bridge_v2.'),
    'fixture_symlink': (
        'Symlink fixture files are not permitted.',
        'Use regular synthetic JSON fixtures in the repository fixture directory.'),
    'private_path': (
        'Private data paths are outside this synthetic task.',
        'Use synthetic fixtures and an output directory outside private data locations.'),
}
STAGES = frozenset({'startup', 'network_guard', 'path_policy', 'fixtures', 'results_directory',
                    'synthetic_generation', 'prediction', 'prediction_write',
                    'scoring', 'provenance', 'artifact_write'})


class OfflineConfigurationError(ValueError):
    def __init__(self, code, stage):
        message, _ = CONFIG_ERRORS[code]
        super().__init__(message)
        self.code = code
        self.stage = stage


class OfflineStageError(RuntimeError):
    def __init__(self, stage):
        super().__init__('Offline smoke stage failed')
        self.stage = stage


@contextmanager
def error_stage(stage):
    """Keep the real exception chained for local diagnosis, redact CLI output."""
    try:
        yield
    except (OfflineConfigurationError, OfflineStageError):
        raise
    except Exception as exc:
        raise OfflineStageError(stage) from exc


def safe_failure(exc):
    """Return JSON-safe diagnostics and exit status without inspecting str(exc)."""
    stage = getattr(exc, 'stage', 'startup')
    if stage not in STAGES:
        stage = 'startup'
    cause = exc.__cause__ if isinstance(exc, OfflineStageError) else exc
    if isinstance(cause, OfflineConfigurationError):
        code = cause.code
        message, hint = CONFIG_ERRORS[code]
        kind, status = 'OfflineConfigurationError', 2
    elif isinstance(cause, OSError):
        code, kind, status = 'filesystem_io', 'OSError', 2
        message = 'A filesystem operation failed; no live fallback was attempted.'
        hint = 'Check access permissions, available disk space, and the synthetic fixture/output directories.'
    elif isinstance(cause, (PackageNotFoundError, ImportError)):
        code, kind, status = 'dependency_unavailable', 'ImportError', 2
        message = 'A required local dependency or its package metadata is unavailable.'
        hint = 'Use the project virtual environment and PYTHONPATH="$PWD/src"; verify installed project dependencies.'
    elif isinstance(cause, ValueError):
        code, kind, status = 'invalid_configuration_or_value', 'ValueError', 2
        message = 'Offline validation or serialization failed.'
        hint = 'Check fixture schema/provenance and finite synthetic prediction values at the reported stage.'
    else:
        code, kind, status = 'unexpected_failure', 'Exception', 1
        for known in (RuntimeError, TypeError, KeyError, AttributeError, AssertionError):
            if isinstance(cause, known):
                kind = known.__name__
                break
        message = 'The offline smoke run failed before successful completion.'
        hint = 'Inspect the reported stage in the v2 runner and reproduce locally with synthetic inputs only.'
    result = {'status': 'error', 'error_code': code, 'stage': stage,
              'exception_type': kind, 'message': message, 'hint': hint,
              'live_fallback': False}
    if isinstance(cause, OSError) and isinstance(cause.errno, int):
        # errno names are allowlisted stdlib constants; never include filenames.
        result['errno'] = errno.errorcode.get(cause.errno, 'UNKNOWN')
    return result, status
