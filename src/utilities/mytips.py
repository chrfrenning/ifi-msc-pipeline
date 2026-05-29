import random

#
# Ah how bored I got waiting for processing and training...
#

def get_life_tip():

    suggestions = [
        "How about a coffee?",
        "Maybe get yourself a snack?",
        "Jumping up and down is good for your health!",
        "Maybe read a few pages of a book?",
        "Listen to your favorite song?",
        "Try a short meditation?",
        "Drink a glass of water?",
        "Take a short walk?",
        "Do some quick stretches?",
        "Call your mom?",
        "Do you like Tetris?"
    ]

    return random.choice(suggestions)
