from pathlib import Path

import pytest
from PIL import Image

from app import media, product_views
from app.pipeline import Pipeline
from app.research_log import ResearchLog


def sources(project, count, *, size=(200,100)):
    (project / "inputs").mkdir(parents=True)
    paths = [project / "inputs/replacement-image"] + [project / f"inputs/replacement-image-{i}" for i in range(2,count+1)]
    for index, path in enumerate(paths):
        Image.new("RGB",size,(40+index*40,10,20)).save(path, format="PNG")
    return paths


@pytest.mark.parametrize("count,dimensions", [(2,(1536,768)),(3,(2304,768)),(4,(1536,1536))])
def test_composite_layout_preserves_order_and_aspect_ratio(tmp_path, count, dimensions):
    paths = sources(tmp_path,count)
    output = product_views.compose_product_views(tmp_path, paths)
    manifest = product_views.frozen_product_views(tmp_path)
    with Image.open(output) as image:
        assert image.size == dimensions
        for index, view in enumerate(manifest["layout"]["views"]):
            x,y,width,height = view["image_bbox"]
            assert width == height*2
            assert image.getpixel((x+width//2,y+height//2)) == (40+index*40,10,20)
            assert view["source"] == paths[index].relative_to(tmp_path).as_posix()
    assert manifest["interpretation"] == "one_product_multiple_views"


def test_transparent_pixels_use_white_background(tmp_path):
    paths = sources(tmp_path,2)
    Image.new("RGBA",(200,100),(255,0,0,0)).save(paths[0],format="PNG")
    output = product_views.compose_product_views(tmp_path,paths)
    view = product_views.frozen_product_views(tmp_path)["layout"]["views"][0]
    x,y,w,h = view["image_bbox"]
    with Image.open(output) as image:
        assert image.getpixel((x+w//2,y+h//2)) == (255,255,255)


def test_exif_rotation_applies_before_fitting(tmp_path):
    paths = sources(tmp_path,2)
    image = Image.new("RGB",(200,100),(255,0,0))
    exif = image.getexif()
    exif[274] = 6
    image.save(paths[0],format="JPEG",exif=exif)
    product_views.compose_product_views(tmp_path,paths)
    bbox = product_views.frozen_product_views(tmp_path)["layout"]["views"][0]["image_bbox"]
    assert bbox[3] == bbox[2]*2


def test_frozen_reference_is_reused_without_redrawing(tmp_path,monkeypatch):
    paths = sources(tmp_path,3)
    output = product_views.compose_product_views(tmp_path,paths)
    before = output.stat().st_mtime_ns,output.read_bytes()
    monkeypatch.setattr(Image,"open",lambda *_args,**_kwargs: pytest.fail("must reuse frozen composite"))
    assert product_views.compose_product_views(tmp_path,paths) == output
    assert (output.stat().st_mtime_ns,output.read_bytes()) == before


def test_frozen_source_change_and_order_change_are_rejected(tmp_path):
    paths = sources(tmp_path,2)
    output = product_views.compose_product_views(tmp_path,paths)
    before = output.read_bytes()
    with pytest.raises(ValueError,match="order"):
        product_views.compose_product_views(tmp_path,list(reversed(paths)))
    paths[1].write_bytes(b"changed")
    with pytest.raises(ValueError,match="source changed"):
        product_views.compose_product_views(tmp_path,paths)
    assert output.read_bytes() == before


def test_single_image_retains_existing_provider_conversion(tmp_path,monkeypatch):
    paths = sources(tmp_path,1)
    pipeline = object.__new__(Pipeline)
    calls = []
    def convert(source,destination):
        calls.append(source)
        destination.write_bytes(source.read_bytes())
        return destination
    monkeypatch.setattr(media,"provider_reference_png",convert)
    result = pipeline._materialize_canonical_user_reference(tmp_path,paths[0],ResearchLog(tmp_path),replacement_paths=paths)
    assert calls == paths
    assert result == tmp_path / product_views.REFERENCE
    assert not (tmp_path / product_views.MANIFEST).exists()


def test_exact_multiview_mapping_is_in_authoritative_h3_block():
    sheet = {"h3_image_reference":"Image 1","user_image_present":True,
             "upstream_bindings":{"replacement_views":{"view_count":3}},
             "spec":{"tiles":[{"stable_key":"entity-01","tile_id":"TILE_01","row":1,"column":1}]}}
    result = Pipeline._fusion_authority_block({"people":{},"entities":{},"scenes":{},"relations":{}},1,sheet)
    assert "Image 2 is one composite containing 3 uploaded views" in result
    assert "SAME single product" in result
    assert "do not invent missing views" in result
