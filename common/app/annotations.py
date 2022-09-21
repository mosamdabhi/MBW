import json
import math
from datetime import datetime
from pathlib import Path


class AnnotationManager:
    def __init__(self, annotations_dir: str, landmark_name: str):
        self.annotations_dir = Path(annotations_dir)
        self.annotations_dir.mkdir(exist_ok=True, parents=True)

        self.landmark_filename = landmark_name + ".json"
        self.num_annotations_written = 0

        self._annotation_started = datetime.now()

    def annotation_exists(self, image: str):
        return self._get_annotation_path(image).exists()

    def write_coordinates(self, image: str, x: int, y: int):
        data = {"coordinates": {"x": x, "y": y}, "status": "ok"}
        self._write_annotation(image, data)

    def mark_occluded(self, image: str):
        data = {"status": "occluded/missing"}
        self._write_annotation(image, data)

    def _get_annotation_path(self, image_name: str):
        return self.annotations_dir / image_name / self.landmark_filename

    def _write_annotation(self, image, data):
        outfile = self._get_annotation_path(image)
        outfile.parent.mkdir(exist_ok=True)

        # might already exist if user pressed the back button, don't count as new annotation in this case
        if not self.annotation_exists(image):
            self.num_annotations_written += 1
            self._print_annotation_time_metrics()

        outfile.write_text(json.dumps(data))

    def _print_annotation_time_metrics(self):
        # only print every 100 images
        if self.num_annotations_written == 0 or self.num_annotations_written % 100 != 0:
            return

        delta = int((datetime.now() - self._annotation_started).total_seconds())
        minutes = math.floor(delta / 60)
        seconds = delta % 60

        print(
            "Generated {} annotations in {}min{}sec".format(
                self.num_annotations_written, minutes, seconds
            )
        )
