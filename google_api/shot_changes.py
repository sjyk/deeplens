"""
Shot change analysis detects shot changes in a video.

This section demonstrates a few ways to analyze a video for shot changes.

Here is an example of performing video analysis for shot changes on a file
located in Google Cloud Storage.
"""
from google.cloud import videointelligence

""" Detects camera shot changes. """
video_client = videointelligence.VideoIntelligenceServiceClient()
features = [videointelligence.enums.Feature.SHOT_CHANGE_DETECTION]
operation = video_client.annotate_video(
    input_uri='gs://deeplens-videos/my_desk.mp4', features=features)
print('\nProcessing video for shot change annotations:')

result = operation.result(timeout=90)
print('\nFinished processing.')

# first result is retrieved because a single video was processed
for i, shot in enumerate(result.annotation_results[0].shot_annotations):
    start_time = (shot.start_time_offset.seconds +
                  shot.start_time_offset.nanos / 1e9)
    end_time = (shot.end_time_offset.seconds +
                shot.end_time_offset.nanos / 1e9)
    print('\tShot {}: {} to {}'.format(i, start_time, end_time))