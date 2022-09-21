import json
import sys

from app.annotations import AnnotationManager
from app.coordinates import init_scale_coordinates, parse_coordinates
from app.images import ImageManager
from flask import Flask, redirect, render_template, request, send_from_directory, url_for

sys.path.append("common")

app = Flask(__name__)

with open("common/app/args_annotation.json", "r") as fid:
    args_json = json.load(fid)

ANNOTATION_DIR = args_json["ANNOTATION_DIR"]
LANDMARK = args_json["LANDMARK"]
IMAGE_DIR = args_json["IMAGE_DIR"]
IMAGE_HEIGHT = args_json["IMAGE_HEIGHT"]
TOTAL_ANNOTATIONS = args_json["TOTAL_ANNOTATIONS"]


annotations = AnnotationManager(ANNOTATION_DIR, LANDMARK)
images = ImageManager(IMAGE_DIR, annotations)
scale_coordinates = init_scale_coordinates(IMAGE_DIR, IMAGE_HEIGHT)

########################################################################
# sending tasks
########################################################################


def shutdown_server():
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()


def select_next_task(current_image=None):
    if current_image is None:
        next_image = images.get_first_image()
    else:
        next_image = images.get_next_image(current_image)

    if next_image is None:
        return redirect(url_for("finished")), shutdown_server()

    else:
        return redirect(url_for("send_task", image=next_image))


@app.route("/")
def send_first_task():
    return select_next_task(current_image=None)


@app.route("/tasks/<image>")
def send_task(image):
    previous_image = images.get_previous_image(image)
    if previous_image is None:
        url_back = url_for(
            "send_task", image=image
        )  # can't go back so sending the same image again - improve?
    else:
        url_back = url_for("send_task", image=previous_image)

    return render_template(
        "landmark.html",
        landmark_name=LANDMARK,
        image_name=image,
        image_height=IMAGE_HEIGHT,
        url_back=url_back,
        done=images.num_previously_annotated + annotations.num_annotations_written,
        total=TOTAL_ANNOTATIONS,
    )


@app.route("/finished")
def finished():
    # check on the filesystem again to make sure really everything annotated
    # (annotations can be missing if user skipped images or deleted annotation files)
    annotations_now = AnnotationManager(ANNOTATION_DIR, LANDMARK)
    images_now = ImageManager(IMAGE_DIR, annotations_now)

    num_not_annotated = images_now.num_images - images_now.num_previously_annotated
    return render_template("finished.html", num_not_annotated=num_not_annotated)


@app.route("/restart")
def restart():
    global annotations
    annotations = AnnotationManager(ANNOTATION_DIR, LANDMARK)
    global images
    images = ImageManager(IMAGE_DIR, annotations)
    return redirect(url_for("send_first_task"))


########################################################################
# storing results
########################################################################


@app.route("/results/<image>/coordinates")
def store_result(image):
    # somehow there seems to be some margin around the image that is still clickable but does not send any coordinates
    # probably was an error that the user clicked there -> send the same image again
    if len(request.args) == 0:
        return redirect(url_for("send_task", image=image))

    x, y = parse_coordinates(request.args)
    x, y = scale_coordinates(image, x, y)
    annotations.write_coordinates(image, x, y)

    return select_next_task(current_image=image)


@app.route("/results/<image>/occluded")
def mark_occluded(image):
    annotations.mark_occluded(image)
    return select_next_task(current_image=image)


@app.route("/results/<image>/skipped")
def skip(image):
    # currently not storing any information about skipped images, just redirect to the next page
    return select_next_task(current_image=image)


########################################################################
# images
########################################################################

# also serve the actual image files here, for simplicity
@app.route("/images/<filename>")
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)
