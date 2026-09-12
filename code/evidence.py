import base64
import json
import mimetypes
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "dataset"

CACHE_PATH = (
    ROOT
    / "evidence_cache.json"
)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


def empty_cache():

    return {
        "image_events": {},
        "messages": {},
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }


def load_evidence_cache():

    if not CACHE_PATH.exists():
        return empty_cache()

    with open(
        CACHE_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        cache = json.load(file)

    cache.setdefault(
        "image_events",
        {},
    )

    cache.setdefault(
        "messages",
        {},
    )

    cache.setdefault(
        "usage",
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    )

    return cache


def save_evidence_cache(cache):

    with open(
        CACHE_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            cache,
            file,
            indent=2,
            ensure_ascii=False,
        )


def find_image_file(image_id):

    matches = []

    for path in ROOT.rglob(
        f"{image_id}.*"
    ):

        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            matches.append(path)

    if not matches:

        raise FileNotFoundError(
            f"No image file found "
            f"for {image_id}"
        )

    if len(matches) > 1:

        raise RuntimeError(
            f"Multiple image files found "
            f"for {image_id}: "
            f"{matches}"
        )

    return matches[0]


def image_to_data_url(path):

    mime_type, _ = (
        mimetypes.guess_type(path)
    )

    if mime_type is None:

        mime_type = (
            "application/octet-stream"
        )

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")

    return (
        f"data:{mime_type};"
        f"base64,{encoded}"
    )


def add_usage(
    cache,
    response,
):

    usage = getattr(
        response,
        "usage",
        None,
    )

    if usage is None:
        return

    input_tokens = (
        getattr(
            usage,
            "input_tokens",
            0,
        )
        or 0
    )

    output_tokens = (
        getattr(
            usage,
            "output_tokens",
            0,
        )
        or 0
    )

    total_tokens = (
        getattr(
            usage,
            "total_tokens",
            0,
        )
        or 0
    )

    cache["usage"][
        "input_tokens"
    ] += input_tokens

    cache["usage"][
        "output_tokens"
    ] += output_tokens

    cache["usage"][
        "total_tokens"
    ] += total_tokens


def apply_image_evidence(events):

    events = events.copy()

    cache = load_evidence_cache()

    resolved = cache.get(
        "image_events",
        {},
    )

    for event_id, evidence in (
        resolved.items()
    ):

        amount = evidence.get(
            "amount"
        )

        if amount is None:
            continue

        mask = (
            events["event_id"]
            == event_id
        )

        if not mask.any():
            continue

        # Only fill missing amounts.
        missing_mask = (
            mask
            & events["amount"].isna()
        )

        events.loc[
            missing_mask,
            "amount",
        ] = float(amount)

    return events