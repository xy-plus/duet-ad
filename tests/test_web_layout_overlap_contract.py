from test_web_h3_contract import APP_JS


STYLES = APP_JS.parent / "styles.css"


def _rule(css: str, selector: str) -> str:
    return css.split(selector + " {", 1)[1].split("}", 1)[0]


def test_featured_keyframe_keeps_intrinsic_height_for_single_image_results():
    css = STYLES.read_text(encoding="utf-8")
    card = _rule(css, ".kf-card")
    featured = _rule(css, ".kf-card:first-child")
    image = _rule(css, ".kf-card img")

    assert "position: relative" in card
    assert "position: absolute" in image
    assert "aspect-ratio: auto" not in featured
    assert "aspect-ratio: 4 / 3" in featured

    source = APP_JS.read_text(encoding="utf-8")
    postprocess = source.split("function ppFramesSection", 1)[1].split(
        "function ppResultDisclosure", 1
    )[0]
    assert "kfGrid(detail, own, pathPrefix" in postprocess


def test_keyframe_grid_tracks_cannot_overflow_narrow_result_cards():
    css = STYLES.read_text(encoding="utf-8")
    grid = _rule(css, ".kf-grid")
    mobile = css.split("@media (max-width: 768px)", 1)[1]
    mobile_grid = _rule(mobile, ".kf-grid")

    assert "repeat(4, minmax(0, 1fr))" in grid
    assert "repeat(2, minmax(0, 1fr))" in mobile_grid


def test_postprocess_chat_children_remain_in_normal_flow_and_wrap_labels():
    css = STYLES.read_text(encoding="utf-8")
    result = _rule(css, ".pp-result-disclosure")
    panel = _rule(css, ".pp-result-panel")
    bubble = _rule(css, ".bubble-user")
    chip = _rule(css, ".pp-chip")

    assert "min-width: 0" in result
    assert "position:" not in panel
    assert "min-width: 0" in bubble
    assert "max-width: 100%" in chip
    assert "overflow-wrap: anywhere" in chip

