"""
serializers

factories reuseable helpers
"""


def feed_conversation(samples, limit=5, threshold=.85):
    """helper function to feed result of classifier to Conversation module."""
    try:
        iter(samples)
        assert not any(not isinstance(sub, tuple) for sub in samples)
    except (AssertionError, TypeError) as e:
        raise TypeError('samples must be an iterable contains tuples.')

    samples.sort(key=lambda x: x[1], reverse=True)
    samples = samples[:limit]

    if samples[0][1] > threshold:
        return samples[0]
    elif samples[0][1] > threshold / 3:
        return samples
    else:
        return None


def jsonify_corpus(corpus_dict):
    return [{"url": key,
             "label": corpus_dict[key][0],
             "doc": corpus_dict[key][1:]} for key in corpus_dict]


def jsonify_response(default):
    pass
