"""Object tracking in a video stored on GCS.
It also gives the bounding boxes.

Object tracking tracks multiple objects detected in an input video.
To make an object tracking request, call the annotate method and specify
OBJECT_TRACKING in the features field.

An object tracking request annotates a video with labels (tags) for entities
that are detected in the video or video segments provided. For example, a video
of vehicles crossing a traffic signal might produce labels such as "car",
"truck", "bike," "tires", "lights", "window" and so on. Each label can include
a series of bounding boxes, with each bounding box having an associated time
segment containing a time offset (timestamp) that indicates the duration offset
from the beginning of the video. The annotation also contains additional entity
information including an entity id that you can use to find more information
about the entity in the Google Knowledge Graph Search API.

Note: There is a limit on the size of the detected objects. Very small objects
in the video might not get detected.

Object Tracking vs. Label Detection

Object tracking differs from label detection in that label detection provides
labels without bounding boxes, while object tracking detects the presence of
individual boxable objects in a given video along with the bounding box for
each.
"""
from google.cloud import videointelligence

video_client = videointelligence.VideoIntelligenceServiceClient()
features = [videointelligence.enums.Feature.OBJECT_TRACKING]
operation = video_client.annotate_video(
    input_uri='gs://deeplens-videos/my_desk.mp4',
    features=features)
print('\nProcessing video for object annotations:')

result = operation.result(timeout=300)
print('\nFinished processing.\n')

# The first result is retrieved because a single video was processed.
object_annotations = result.annotation_results[0].object_annotations

for object_annotation in object_annotations:
    print('Entity description: {}'.format(
        object_annotation.entity.description))
    if object_annotation.entity.entity_id:
        print('Entity id: {}'.format(object_annotation.entity.entity_id))

    print('Segment: {}s to {}s'.format(
        object_annotation.segment.start_time_offset.seconds +
        object_annotation.segment.start_time_offset.nanos / 1e9,
        object_annotation.segment.end_time_offset.seconds +
        object_annotation.segment.end_time_offset.nanos / 1e9))

    print('Confidence: {}'.format(object_annotation.confidence))

    # Here we print only the bounding box of the first frame in the segment
    frame = object_annotation.frames[0]
    box = frame.normalized_bounding_box
    print('Time offset of the first frame: {}s'.format(
        frame.time_offset.seconds + frame.time_offset.nanos / 1e9))
    print('Bounding box position:')
    print('\tleft  : {}'.format(box.left))
    print('\ttop   : {}'.format(box.top))
    print('\tright : {}'.format(box.right))
    print('\tbottom: {}'.format(box.bottom))
    print('\n')
