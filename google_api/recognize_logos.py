from google.cloud import videointelligence_v1p3beta1
from google.cloud.videointelligence_v1p3beta1 import enums


def sample_annotate_video():
    client = videointelligence_v1p3beta1.VideoIntelligenceServiceClient()

    input_uri = 'gs://deeplens-videos/my_desk.mp4'
    features_element = enums.Feature.LOGO_RECOGNITION
    features = [features_element]

    operation = client.annotate_video(input_uri=input_uri, features=features)

    print(u'Waiting for operation to complete...')
    response = operation.result()

    # Get the first response, since we sent only one video.
    annotation_result = response.annotation_results[0]
    # Annotations for list of logos detected, tracked and recognized in video.
    for logo_recognition_annotation in annotation_result.logo_recognition_annotations:
        entity = logo_recognition_annotation.entity
        # Opaque entity ID. Some IDs may be available in [Google Knowledge Graph Search API](https://developers.google.com/knowledge-graph/).
        print(u'Entity Id : {}'.format(entity.entity_id))
        # Textual description, e.g. `Google`.
        print(u'Description : {}'.format(entity.description))
        # All logo tracks where the recognized logo appears. Each track corresponds to one logo instance appearing in consecutive frames.
        for track in logo_recognition_annotation.tracks:
            # Video segment of a track.
            segment = track.segment
            segment_start_time_offset = segment.start_time_offset
            print(u'\n\tStart Time Offset : {}.{}'.format(
                segment_start_time_offset.seconds,
                segment_start_time_offset.nanos))
            segment_end_time_offset = segment.end_time_offset
            print(u'\tEnd Time Offset : {}.{}'.format(
                segment_end_time_offset.seconds, segment_end_time_offset.nanos))
            print(u'\tConfidence : {}'.format(track.confidence))
            # The object with timestamp and attributes per frame in the track.
            for timestamped_object in track.timestamped_objects:
                # Normalized Bounding box in a frame, where the object is located.
                normalized_bounding_box = timestamped_object.normalized_bounding_box
                print(u'\n\t\tLeft : {}'.format(normalized_bounding_box.left))
                print(u'\t\tTop : {}'.format(normalized_bounding_box.top))
                print(u'\t\tRight : {}'.format(normalized_bounding_box.right))
                print(u'\t\tBottom : {}'.format(normalized_bounding_box.bottom))
                # Optional. The attributes of the object in the bounding box.
                for attribute in timestamped_object.attributes:
                    print(u'\n\t\t\tName : {}'.format(attribute.name))
                    print(u'\t\t\tConfidence : {}'.format(attribute.confidence))
                    print(u'\t\t\tValue : {}'.format(attribute.value))
            # Optional. Attributes in the track level.
            for track_attribute in track.attributes:
                print(u'\n\t\tName : {}'.format(track_attribute.name))
                print(u'\t\tConfidence : {}'.format(track_attribute.confidence))
                print(u'\t\tValue : {}'.format(track_attribute.value))
        # All video segments where the recognized logo appears. There might be multiple instances of the same logo class appearing in one VideoSegment.
        for logo_recognition_annotation_segment in logo_recognition_annotation.segments:
            logo_recognition_annotation_segment_start_time_offset = logo_recognition_annotation_segment.start_time_offset
            print(u'\n\tStart Time Offset : {}.{}'.format(
                logo_recognition_annotation_segment_start_time_offset.seconds,
                logo_recognition_annotation_segment_start_time_offset.nanos))
            logo_recognition_annotation_segment_end_time_offset = logo_recognition_annotation_segment.end_time_offset
            print(u'\tEnd Time Offset : {}.{}'.format(
                logo_recognition_annotation_segment_end_time_offset.seconds,
                logo_recognition_annotation_segment_end_time_offset.nanos))


if __name__ == "__main__":
    sample_annotate_video()
