from pathlib import Path
import hashlib
import json

import pytest
from PIL import Image, ImageOps

from app import media, product_views
from app.pipeline import Pipeline
from app.research_log import ResearchLog


def sources(project, count, *, size=(200,100)):
    (project / "inputs").mkdir(parents=True)
    paths = [project / "inputs/replacement-image"] + [project / f"inputs/replacement-image-{i}" for i in range(2,count+1)]
    for index, path in enumerate(paths):
        Image.new("RGB",size,(40+index*40,10,20)).save(path, format="PNG")
    return paths


@pytest.mark.parametrize("count,dimensions", [(2,(1536,768)),(3,(1536,1536)),(4,(1536,1536))])
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


@pytest.mark.parametrize("count", [1,2,3,4])
@pytest.mark.parametrize("size", [(1,4096),(4096,1),(1,1),(100,4096),(4096,100)])
def test_all_upload_counts_and_extreme_source_sizes_meet_h3_ratio(tmp_path,count,size):
    paths = sources(tmp_path,count,size=size)
    pipeline = object.__new__(Pipeline)
    output = pipeline._materialize_canonical_user_reference(
        tmp_path,paths[0],ResearchLog(tmp_path),replacement_paths=paths,
    )
    with Image.open(output) as image:
        assert .4 <= image.width/image.height <= 2.5
        if count > 1:
            manifest = product_views.frozen_product_views(tmp_path)
            assert len(manifest["layout"]["views"]) == count
            for index,view in enumerate(manifest["layout"]["views"]):
                x,y,w,h = view["image_bbox"]
                assert w >= 1 and h >= 1
                assert 0 <= x < x+w <= image.width
                assert 0 <= y < y+h <= image.height
                assert image.getpixel((x+w//2,y+h//2)) == (40+index*40,10,20)
                scale = min((product_views.CELL_SIZE-24)/size[0],
                            (product_views.CELL_SIZE-product_views.LABEL_HEIGHT-24)/size[1])
                assert (w,h) == (max(1,round(size[0]*scale)),max(1,round(size[1]*scale)))


def test_three_views_have_only_three_numbered_cells_and_blank_fourth(tmp_path):
    output = product_views.compose_product_views(tmp_path,sources(tmp_path,3))
    manifest = product_views.frozen_product_views(tmp_path)
    assert [view["view"] for view in manifest["layout"]["views"]] == [1,2,3]
    assert [view["cell"] for view in manifest["layout"]["views"]] == [
        [0,0,768,768],[768,0,768,768],[0,768,768,768],
    ]
    with Image.open(output) as image:
        assert image.crop((768,768,1536,1536)).getextrema() == ((255,255),(255,255),(255,255))


@pytest.mark.parametrize("size", [(4096,1),(1,4096),(4096,100),(100,4096)])
def test_extreme_views_keep_both_source_edges_without_cropping(tmp_path,size):
    paths = sources(tmp_path,2,size=size)
    source = Image.new("RGB",size,"white")
    wide = size[0] > size[1]
    if wide:
        source.paste((255,0,0),(0,0,size[0]//4,size[1]))
        source.paste((0,0,255),(size[0]*3//4,0,size[0],size[1]))
    else:
        source.paste((255,0,0),(0,0,size[0],size[1]//4))
        source.paste((0,0,255),(0,size[1]*3//4,size[0],size[1]))
    source.save(paths[0],format="PNG")
    output = product_views.compose_product_views(tmp_path,paths)
    x,y,w,h = product_views.frozen_product_views(tmp_path)["layout"]["views"][0]["image_bbox"]
    with Image.open(output) as image:
        first = (x+w//8,y+h//2) if wide else (x+w//2,y+h//8)
        last = (x+w*7//8,y+h//2) if wide else (x+w//2,y+h*7//8)
        assert image.getpixel(first) == (255,0,0)
        assert image.getpixel(last) == (0,0,255)


def test_new_three_view_render_is_deterministic(tmp_path):
    first,second = tmp_path/"first",tmp_path/"second"
    first_output = product_views.compose_product_views(first,sources(first,3))
    second_output = product_views.compose_product_views(second,sources(second,3))
    assert first_output.read_bytes() == second_output.read_bytes()


@pytest.mark.parametrize("size", [(200,100),(747,713),(1000,500),(40,60)])
def test_normal_views_keep_original_contain_geometry_and_pixels(tmp_path,size):
    paths = sources(tmp_path,2,size=size)
    original = Image.linear_gradient("L").resize(size).convert("RGB")
    original.save(paths[0],format="PNG")
    expected = ImageOps.contain(original,(744,712))
    output = product_views.compose_product_views(tmp_path,paths)
    x,y,w,h = product_views.frozen_product_views(tmp_path)["layout"]["views"][0]["image_bbox"]
    assert (w,h) == expected.size
    with Image.open(output) as image:
        assert image.crop((x,y,x+w,y+h)).tobytes() == expected.tobytes()


def test_legacy_frozen_three_by_one_is_not_redrawn_or_rewritten(tmp_path,monkeypatch):
    paths = sources(tmp_path,3)
    output = product_views.compose_product_views(tmp_path,paths)
    manifest_path = tmp_path/product_views.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    # Build a historical receipt entirely inside this test's temporary project.
    Image.new("RGB",(2304,768),(20,40,60)).save(output,format="PNG")
    payload = output.read_bytes()
    manifest["output"].update(sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
    manifest["layout"].update(columns=3,rows=1)
    manifest["layout"]["views"][2]["cell"] = [1536,0,768,768]
    manifest["layout"]["views"][2]["image_bbox"] = [1548,214,744,372]
    manifest_path.write_text(json.dumps(manifest))
    before = payload,output.stat().st_mtime_ns,manifest_path.read_bytes()
    monkeypatch.setattr(Image,"open",lambda *_args,**_kwargs: pytest.fail("must not redraw frozen provider inputs"))
    assert product_views.compose_product_views(tmp_path,paths) == output
    assert (output.read_bytes(),output.stat().st_mtime_ns,manifest_path.read_bytes()) == before
