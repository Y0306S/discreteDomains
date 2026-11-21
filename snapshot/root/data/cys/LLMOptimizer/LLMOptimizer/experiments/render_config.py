#!/usr/bin/env python3
"""Render a JSON template, including dictionary keys, without launching anything."""
import argparse
import json
import os
from pathlib import Path
import re


def render(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            key = render(key)
            if key in result:
                raise ValueError(f"Duplicate rendered key: {key}")
            result[key] = render(item)
        return result
    if isinstance(value, list):
        return [render(item) for item in value]
    if isinstance(value, str):
        def substitute(match):
            name = match.group(1)
            if name not in os.environ or not os.environ[name]:
                raise ValueError(f"Set environment variable {name} before rendering")
            return os.environ[name]
        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", substitute, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rendered = render(json.loads(args.template.read_text()))
    # Exclusive creation prevents accidental replacement of a campaign identity.
    with args.output.open("x") as stream:
        json.dump(rendered, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
