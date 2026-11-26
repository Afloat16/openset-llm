"""Read resumable benchmark output and repair only an unfinished final record."""
import json
import sys


def read_rows(path):
    rows = []
    needs_newline = False
    with open(path, "r+b") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            needs_newline = not line.endswith(b"\n")
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if not needs_newline:
                    raise  # a complete corrupt record must not be silently discarded
                stream.seek(offset)
                stream.truncate()
                print(f"Discarded an incomplete final result row in {path}; completed rows were kept.", file=sys.stderr)
                return rows
        if needs_newline:
            stream.write(b"\n")  # do not concatenate the next appended JSON object
    return rows
