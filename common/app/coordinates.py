from pathlib import Path

from PIL import Image


# coordinates are sent as slightly weird URL parameters (e.g. 0.png?214,243)
# parse them, will crash server if they are coming in unexpected format
def parse_coordinates(args):
    keys = list(args.keys())
    assert len(keys) == 1
    coordinates = keys[0]

    assert len(coordinates.split(",")) == 2
    x, y = coordinates.split(",")
    x = int(x)
    y = int(y)
    return x, y


# image was not displayed in original size -> need to convert the coordinates
def init_scale_coordinates(image_dir: str, scaled_height: int):
    image_dir = Path(image_dir)

    def perform_scaling(image: str, x: int, y: int):
        image = Image.open(str(image_dir / image))
        original_width, original_height = image.size

        scale = original_height / scaled_height
        return int(x * scale), int(y * scale)

    return perform_scaling
