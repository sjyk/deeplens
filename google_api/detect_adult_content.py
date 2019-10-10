"""
Detecting explicit content in videos

Explicit Content Detection detects adult content within a video.
Adult content is content generally appropriate for 18 years of age and older,
including but not limited to nudity, sexual activities, and pornography
(including cartoons or anime).

The response includes a bucketized likelihood value, from VERY_UNLIKELY to
VERY_LIKELY.

When Explicit Content Detection evaluates a video, it does so on a per-frame
basis and considers visual content only. The audio component of the video is
not used to evaluate the explicit content level. Google does not guarantee the
accuracy of its Explicit Content Detection predictions.

Here is an example of performing video analysis for Explicit Content Detection
features on a file located in Google Cloud Storage.
"""


from google.cloud import videointelligence

""" Detects explicit content from the GCS path to a video. """
video_client = videointelligence.VideoIntelligenceServiceClient()
features = [videointelligence.enums.Feature.EXPLICIT_CONTENT_DETECTION]

operation = video_client.annotate_video(
    input_uri='gs://deeplens-videos/my_desk.mp4', features=features)
print('\nProcessing video for explicit content annotations:')

result = operation.result(timeout=90)
print('\nFinished processing.')

# first result is retrieved because a single video was processed
for frame in result.annotation_results[0].explicit_annotation.frames:
    likelihood = videointelligence.enums.Likelihood(
        frame.pornography_likelihood)
    frame_time = frame.time_offset.seconds + frame.time_offset.nanos / 1e9
    print('Time: {}s'.format(frame_time))
    print('\tpornography: {}'.format(likelihood.name))
