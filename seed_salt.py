"""Secret randomization salt shared by the RL and query handlers.

Every Bernoulli draw uses a seed hashed from ``[salt, user_id, week, ...]``.
Without a secret salt the allocation is a public function of the participant
ID, so the salt must be set before deployment and never changed during the
study.

Lookup order (first hit wins):

1. environment variable ``ADAPT_SEED_SALT`` (recommended on the server);
2. the file named by config ``seed_salt_path`` (default
   ``RL_agents/seed_salt.local``, git-ignored), first non-empty line;
3. config ``seed`` (legacy; discouraged because config files are committed).

The salt itself is never logged or stored. Its fingerprint (a one-way hash)
is stored in model params and every ``random_state`` so a changed salt is
detectable.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

SALT_ENV_VAR = "ADAPT_SEED_SALT"
DEFAULT_SALT_FILE = Path(__file__).with_name("seed_salt.local")
MIN_SALT_LENGTH = 16


class SeedSaltError(RuntimeError):
    """Raised when a required salt is missing or too short."""


def _from_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return None


def load_seed_salt(config: dict[str, Any]) -> tuple[str | None, str]:
    """Return ``(salt, source)``; ``source`` names where it came from (never the value)."""
    env = os.environ.get(SALT_ENV_VAR)
    if env and env.strip():
        return env.strip(), f"env:{SALT_ENV_VAR}"
    raw_path = config.get("seed_salt_path")
    path = Path(raw_path).expanduser() if raw_path else DEFAULT_SALT_FILE
    if not path.is_absolute():
        path = Path(__file__).parent / path
    salt = _from_file(path)
    if salt:
        return salt, f"file:{path.name}"
    if config.get("seed") is not None:
        return str(config["seed"]), "config:seed"
    return None, "unset"


def salt_fingerprint(salt: str | None) -> str | None:
    if salt is None:
        return None
    return hashlib.sha256(("adapt-seed-salt-fingerprint:" + salt).encode("utf-8")).hexdigest()[:12]


def resolve_seed_salt(config: dict[str, Any]) -> tuple[str | None, str, str | None]:
    """Load and check the salt. Returns ``(salt, source, fingerprint)``.

    With ``require_seed_salt`` true (the shipped default) a missing or short
    salt is an error, so an unsalted deployment cannot start silently.
    """
    salt, source = load_seed_salt(config)
    required = bool(config.get("require_seed_salt", True))
    if salt is None:
        if required:
            raise SeedSaltError(
                f"randomization salt is not set: export {SALT_ENV_VAR} or create "
                f"{DEFAULT_SALT_FILE.name} (see RL_agents/README.md)"
            )
        return None, source, None
    if len(salt) < MIN_SALT_LENGTH:
        raise SeedSaltError(f"randomization salt must have at least {MIN_SALT_LENGTH} characters")
    return salt, source, salt_fingerprint(salt)
