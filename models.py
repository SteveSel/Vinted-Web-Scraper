from dataclasses import dataclass


@dataclass
class Listing:
    title: str
    url: str
    image_url: str
    price: str
    source_reference_image: str = ""


@dataclass
class MatchResult:
    listing: Listing
    is_match: bool
    score: float
    reason: str
    raw_model_response: str
