import numpy as np
from instalooter.looters import ProfileLooter
from datetime import datetime


def getData(profile):
    looter = ProfileLooter(profile)
    looter.login("apexshopcz", "Dhcepic11")
    looter.jobs = 1
    mediasIterator = looter.medias()

    scrapedMedia = np.array([[0, 0, 0]])

    for x in mediasIterator:
        time = datetime.utcfromtimestamp(x['taken_at_timestamp'])
        minutes_since_midnight = (
            time - time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()/60
        minutes_since_noon = (
            time - time.replace(hour=12, minute=0, second=0, microsecond=0)).total_seconds()/60
        scrapedMedia = np.append(scrapedMedia, [[minutes_since_midnight,
                                                 x['edge_media_preview_like']['count'], x['edge_media_to_comment']['count']]], axis=0)

    return scrapedMedia
