from pathlib import Path

from app.annotations import AnnotationManager


class ImageManager:
    def __init__(self, image_dir: str, annotation_manager):
        self.image_dir = Path(image_dir)
        all_images = self._index_images(self.image_dir)

        self.images_to_annotate = [
            img for img in all_images if not annotation_manager.annotation_exists(img)
        ]

        self.num_images = len(all_images)
        self.num_previously_annotated = len(all_images) - len(self.images_to_annotate)

        # dictionary of {image: previous_image} (first image will be mapped to None)
        shifted_right = [None] + self.images_to_annotate[:-1]
        self._previous = dict(zip(self.images_to_annotate, shifted_right))

        # dictionary of {image: next_image} (last image will be mapped to None)
        shifted_left = self.images_to_annotate[1:] + [None]
        self._next = dict(zip(self.images_to_annotate, shifted_left))

    def get_first_image(self):
        if len(self.images_to_annotate) == 0:
            return None
        return self.images_to_annotate[0]

    def get_next_image(self, current_image):
        return self._next[current_image]

    def get_previous_image(self, current_image):
        return self._previous[current_image]

    @staticmethod
    def _index_images(image_dir):
        assert image_dir.exists()
        assert image_dir.is_dir()

        images = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        images = sorted(list(images))
        assert len(images) > 0

        return [img.name for img in images]
