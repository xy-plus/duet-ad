from pathlib import Path
import hashlib
import json

import pytest
from PIL import Image, ImageOps

from app import media, product_views
from app.config import Settings
from app.pipeline import Pipeline
from app.research_log import ResearchLog


def _composed_pipeline() -> Pipeline:
    pipeline = object.__new__(Pipeline)
    pipeline.settings = Settings(
        access_token="test",
        multiref_mode="composed",
        public_base_url="https://product-views.test",
    )
    return pipeline


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


def test_single_image_defaults_to_user_preprocessing_and_existing_provider_conversion(tmp_path,monkeypatch):
    paths = sources(tmp_path,1)
    pipeline = _composed_pipeline()
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


def test_single_image_white_border_is_trimmed_before_existing_size_tool(tmp_path):
    paths = sources(tmp_path,1,size=(600,400))
    image = Image.new("RGB",(600,400),"white")
    image.paste((30,40,50),(100,60,500,340))
    image.save(paths[0],format="PNG")
    original_input = paths[0].read_bytes()
    pipeline = _composed_pipeline()
    output = pipeline._materialize_canonical_user_reference(tmp_path,paths[0],ResearchLog(tmp_path),replacement_paths=paths)
    with Image.open(output) as result:
        assert result.size == (408,288)
        assert .4 <= result.width/result.height <= 2.5
    assert not (tmp_path/product_views.MANIFEST).exists()
    assert paths[0].read_bytes() == original_input


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
    pipeline = _composed_pipeline()
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


@pytest.mark.parametrize("background", [(255,255,255),(252,252,252)])
def test_trim_uniform_near_white_border_and_keep_internal_white(background):
    image = Image.new("RGB",(300,200),background)
    image.paste((30,40,50),(50,30,250,170))
    image.paste((255,255,255),(110,70,190,130))
    cropped,bbox = product_views._trim_white_border(image,Image.new("L",image.size,255))
    assert bbox == (48,28,252,172)
    assert cropped.getpixel((150-bbox[0],100-bbox[1])) == (255,255,255)
    assert cropped.getpixel((50-bbox[0],30-bbox[1])) == (30,40,50)
    assert cropped.getpixel((249-bbox[0],169-bbox[1])) == (30,40,50)


@pytest.mark.parametrize("kind", ["pure_white","low_contrast","white_product_logo","invisible_white_product_logo","colored_background","sparse_dark_subject"])
def test_ambiguous_or_nonwhite_background_uses_full_image(kind):
    image = Image.new("RGB",(300,200),"white")
    if kind == "low_contrast":
        image.paste((249,249,249),(30,20,270,180))
    elif kind == "white_product_logo":
        image.paste((248,248,248),(30,20,270,180))
        image.paste((10,10,10),(140,90,160,110))
    elif kind == "invisible_white_product_logo":
        image.paste((10,10,10),(140,90,160,110))
    elif kind == "colored_background":
        image.paste((20,80,160),(0,0,300,200))
        image.paste((10,10,10),(30,20,270,180))
    elif kind == "sparse_dark_subject":
        image.paste((10,10,10),(130,80,170,120))
    cropped,bbox = product_views._trim_white_border(image,Image.new("L",image.size,255))
    assert bbox == (0,0,300,200)
    assert cropped.tobytes() == image.tobytes()


@pytest.mark.parametrize("opaque_subject", [True,False])
def test_alpha_support_preserves_white_subject_or_falls_back_when_empty(opaque_subject):
    image = Image.new("RGB",(300,200),"white")
    alpha = Image.new("L",image.size,0)
    if opaque_subject:
        alpha.paste(255,(50,30,250,170))
    cropped,bbox = product_views._trim_white_border(image,alpha)
    assert bbox == ((48,28,252,172) if opaque_subject else (0,0,300,200))
    assert cropped.getextrema() == ((255,255),(255,255),(255,255))


@pytest.mark.parametrize("count", [2,3,4])
def test_trim_then_fit_enlarges_product_and_keeps_h3_ratio(tmp_path,count):
    paths = sources(tmp_path,count,size=(300,200))
    for path in paths:
        image = Image.new("RGB",(300,200),"white")
        image.paste((30,40,50),(50,30,250,170))
        image.save(path,format="PNG")
    output = product_views.compose_product_views(tmp_path,paths)
    manifest = product_views.frozen_product_views(tmp_path)
    with Image.open(output) as image:
        assert .4 <= image.width/image.height <= 2.5
    for view in manifest["layout"]["views"]:
        assert view["source_size"] == [300,200]
        assert view["source_crop_bbox"] == [48,28,252,172]
        assert view["image_bbox"][2] / 204 > 744/300
