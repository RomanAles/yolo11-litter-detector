import pytest

from scripts import complete_v2


def test_generated_lock_pins_local_packages_and_preserves_remote_references(monkeypatch):
    monkeypatch.setattr(complete_v2, 'version', lambda name: {'packaging': '26.3'}[name])
    frozen = ('# Editable install with no version control\n-e D:/local/project\n'
              'packaging @ file:///home/conda/build/work\n'
              'example @ https://example.org/example.whl\nPyYAML==6.0.3\n')
    assert complete_v2.portable_requirements(frozen) == (
        '-e .\npackaging==26.3\nexample @ https://example.org/example.whl\nPyYAML==6.0.3\n')


def test_generated_lock_requires_editable_project():
    with pytest.raises(RuntimeError, match='Editable project'):
        complete_v2.portable_requirements('packaging==26.3\n')
