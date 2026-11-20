"""Post-sweep stages. Each is a function from earlier stage outputs to its own JSON in stages/.

select -> tracking -> analyze -> register -> read -> resolve -> measure -> items -> room -> price -> assemble
"""

from claimagent.pipeline.household import CATEGORY
from claimagent.pipeline.stages.books import consensus_text, measure_books, read_tracks, resolve_books
from claimagent.pipeline.stages.frames import (
    analyze_frames,
    check_tracking,
    is_keyframe,
    register_tracks,
    select_frames,
)
from claimagent.pipeline.stages.items import collect_items
from claimagent.pipeline.stages.room import measure_room

__all__ = [
    "CATEGORY",
    "analyze_frames",
    "check_tracking",
    "collect_items",
    "consensus_text",
    "is_keyframe",
    "measure_books",
    "measure_room",
    "read_tracks",
    "register_tracks",
    "resolve_books",
    "select_frames",
]
