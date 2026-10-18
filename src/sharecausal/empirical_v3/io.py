"""Private aggregate-only artifacts and a process-wide offline CLI guard."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def private_destination(path):
    root = Path(path).expanduser().resolve()
    # macOS /private is a filesystem root alias, not an explicit privacy choice.
    if 'private' not in root.parts[2:]:
        raise ValueError('Use a private directory component for all empirical outputs')
    repo = Path(__file__).resolve().parents[3]
    if root.is_relative_to(repo) and not root.is_relative_to(repo / 'outputs/private'):
        raise ValueError('Repository output location must be outputs/private')
    if root.exists() and any(root.iterdir()):
        raise ValueError('Choose a new empty output directory; earlier results are immutable')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    return root


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    Path(path).chmod(0o600)


def write_csv(path, rows):
    if not rows:
        raise ValueError('Empty table: report a reason row rather than fabricate results')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path(path).chmod(0o600)


def install_offline_guard():
    """Install only in the dedicated CLI; never import transport/client modules."""
    os.umask(0o077)
    def audit(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo', 'http.client.connect',
                     'urllib.Request', 'subprocess.Popen', 'os.system'}:
            raise PermissionError('Network and child processes disabled for empirical analysis')
        if event == 'open' and isinstance(args[0], (str, bytes)):
            name = Path(os.fsdecode(args[0])).name.lower()
            if name == '.env' or name.startswith('.env.') or name.endswith(('.pem', '.key')):
                raise PermissionError('Credential-file access disabled')
    sys.addaudithook(audit)
