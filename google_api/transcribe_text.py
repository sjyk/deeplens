"""Transcribe speech from a video stored on GCS.
Audio track transcription.

The Video Intelligence API can transcribe speech to text from supported
video files.

Video Intelligence speech transcription supports the following features:

    Alternative words: Use the maxAlternatives option to specify the maximum
    number of options for recognized text translations to include in the
    response. This value can be an integer from 1 to 30. The default is 1.
    The API returns multiple transcriptions in descending order based on the
    confidence value for the transcription. Alternative transcriptions do not
    include word-level entries.

    Profanity filtering: Use the filterProfanity option to filter out known
    profanities in transcriptions. Matched words are replaced with the leading
    character of the word followed by asterisks. The default is false.

    Transcription hints: Use the speechContexts option to provide common or
    unusual phrases in your audio. Those phrases are then used to assist the
    transcription service to create more accurate transcriptions. You provide a
    transcription hint as a SpeechContext object.

    Audio track selection: Use the audioTracks option to specify which track to
    transcribe from multi-track audio. This value can be an integer from 0 to 2.
    Default is 0.

    Automatic punctuation: Use the enableAutomaticPunctuation option to include
    punctuation in the transcribed text. The default is false.

    Multiple speakers: Use the enableSpeakerDiarization option to identify
    different speakers in a video. In the response, each recognized word
    includes a speakerTag field that identifies which speaker the recognized
    word is attributed to.

Note: Speech Transcription only supports English (en-US) language transcription
at this time.
"""
from google.cloud import videointelligence

video_client = videointelligence.VideoIntelligenceServiceClient()
features = [videointelligence.enums.Feature.SPEECH_TRANSCRIPTION]

config = videointelligence.types.SpeechTranscriptionConfig(
    language_code='en-US',
    enable_automatic_punctuation=True)
video_context = videointelligence.types.VideoContext(
    speech_transcription_config=config)

operation = video_client.annotate_video(
    input_uri='gs://deeplens-videos/introduction.mp4',
    features=features,
    video_context=video_context)

print('\nProcessing video for speech transcription.')

result = operation.result(timeout=600)

# There is only one annotation_result since only
# one video is processed.
annotation_results = result.annotation_results[0]
for speech_transcription in annotation_results.speech_transcriptions:

    # The number of alternatives for each transcription is limited by
    # SpeechTranscriptionConfig.max_alternatives.
    # Each alternative is a different possible transcription
    # and has its own confidence score.
    for alternative in speech_transcription.alternatives:
        print('Alternative level information:')

        print('Transcript: {}'.format(alternative.transcript))
        print('Confidence: {}\n'.format(alternative.confidence))

        print('Word level information:')
        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time
            print('\t{}s - {}s: {}'.format(
                start_time.seconds + start_time.nanos * 1e-9,
                end_time.seconds + end_time.nanos * 1e-9,
                word))