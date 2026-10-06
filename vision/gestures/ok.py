import math

def is_ok_qesture(hand):
    thumb_tip = hand[4]
    index_tip = hand[8]

    distance = math.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 +
        (thumb_tip.y - index_tip.y) ** 2
    )

    return distance < 0.03