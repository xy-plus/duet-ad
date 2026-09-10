import cv2
import numpy as np
import pytest
import struct
import zlib

from app import media


def write_image(path, image):
    ok, data = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(data.tobytes())


def test_user_image_trim_reuses_existing_provider_resize(tmp_path,monkeypatch):
    image = np.full((400,600,3),255,np.uint8)
    image[60:340,100:500] = (10,20,30)
    source,destination = tmp_path/"source.png",tmp_path/"output.png"
    write_image(source,image)
    calls = []
    original = media.provider_reference_png
    def tracked(path,target):
        calls.append(cv2.imread(str(path)).shape[:2])
        return original(path,target)
    monkeypatch.setattr(media,"provider_reference_png",tracked)
    media.user_reference_png(source,destination)
    assert calls == [(288,408)]
    output = cv2.imread(str(destination))
    assert .4 <= output.shape[1]/output.shape[0] <= 2.5
    assert not list(tmp_path.glob(".user-reference-trim-*"))


@pytest.mark.parametrize("kind", ["white","white_logo","low_contrast","color"])
def test_ambiguous_user_image_matches_previous_provider_output(tmp_path,kind):
    image = np.full((400,600,3),255,np.uint8)
    if kind == "white_logo": image[180:220,280:320] = 10
    if kind == "low_contrast": image[40:360,60:540] = 248
    if kind == "color": image[:] = (20,70,150)
    source = tmp_path/"source.png"
    write_image(source,image)
    media.provider_reference_png(source,tmp_path/"old.png")
    media.user_reference_png(source,tmp_path/"new.png")
    assert (tmp_path/"old.png").read_bytes() == (tmp_path/"new.png").read_bytes()


@pytest.mark.parametrize("empty", [False,True])
def test_alpha_white_subject_is_retained_and_flattened_to_white(tmp_path,empty):
    image = np.zeros((400,600,4),np.uint8)
    image[:,:,:3] = 255
    if not empty: image[60:340,100:500,3] = 255
    source = tmp_path/"source.png"
    write_image(source,image)
    media.user_reference_png(source,tmp_path/"out.png")
    output = cv2.imread(str(tmp_path/"out.png"))
    assert np.all(output == 255)


def test_internal_white_pixels_and_both_edges_survive_crop():
    image = np.full((200,300,3),255,np.uint8)
    image[30:170,50:250] = (20,30,40)
    image[70:130,110:190] = 255
    cropped,bbox = media.trim_user_reference_border(image)
    assert bbox == (48,28,252,172)
    assert np.array_equal(cropped[72,102],(255,255,255))
    assert np.array_equal(cropped[2,2],(20,30,40))
    assert np.array_equal(cropped[141,201],(20,30,40))


@pytest.mark.parametrize("size", [(1,4096),(4096,1),(1,1)])
def test_extreme_user_images_use_existing_bounds_without_rejection(tmp_path,size):
    source = tmp_path/"source.png"
    write_image(source,np.full((size[1],size[0],3),50,np.uint8))
    media.user_reference_png(source,tmp_path/"out.png")
    image = cv2.imread(str(tmp_path/"out.png"))
    assert .4 <= image.shape[1]/image.shape[0] <= 2.5


def orientation_exif():
    return b"II*\x00\x08\x00\x00\x00"+struct.pack("<H",1)+struct.pack("<HHIHH",274,3,1,6,0)+b"\x00"*4


def test_jpeg_exif_is_applied_before_user_border_crop(tmp_path):
    image = np.full((400,600,3),255,np.uint8)
    image[60:340,100:500] = 20
    ok,payload = cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,100])
    assert ok
    exif = b"Exif\x00\x00"+orientation_exif()
    source = tmp_path/"rotated.jpg"
    source.write_bytes(payload.tobytes()[:2]+b"\xff\xe1"+struct.pack(">H",len(exif)+2)+exif+payload.tobytes()[2:])
    media.user_reference_png(source,tmp_path/"out.png")
    output = cv2.imread(str(tmp_path/"out.png"))
    assert output.shape[0] > output.shape[1]
    assert .4 <= output.shape[1]/output.shape[0] <= 2.5


def test_alpha_exif_keeps_existing_decoder_path_without_orientation_guess(tmp_path):
    image = np.zeros((400,600,4),np.uint8)
    image[60:340,100:500] = (20,40,60,255)
    ok,payload = cv2.imencode(".png",image)
    assert ok
    exif = orientation_exif()
    chunk = b"eXIf"+exif
    data = payload.tobytes()
    source = tmp_path/"alpha-exif.png"
    source.write_bytes(data[:33]+struct.pack(">I",len(exif))+chunk+struct.pack(">I",zlib.crc32(chunk))+data[33:])
    media.provider_reference_png(source,tmp_path/"old.png")
    media.user_reference_png(source,tmp_path/"new.png")
    assert (tmp_path/"old.png").read_bytes() == (tmp_path/"new.png").read_bytes()
