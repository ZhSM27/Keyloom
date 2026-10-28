"""Directional gameplay simplifications, separate from detector label ancestry."""

# Only documented same-family simplifications are automatic. Siblings are not
# interchangeable; explicit secondary requests are still separate intent roles.
LOWER_STYLES = {
    "Quadstream": ("Handstream", "Jumpstream", "Stream"),
    "Chordstream": ("Handstream", "Jumpstream", "Stream"),
    "Handstream": ("Jumpstream", "Stream"),
    "Jumpstream": ("Stream",),
    "Longjacks": ("Chordjacks", "Jacks", "Minijacks"),
    "Gluts": ("Chordjacks", "Jacks", "Minijacks"),
    "Chordjacks": ("Minijacks",),
    "Jacks": ("Minijacks",),
    "Trills": ("Minitrills",),
    "Rolls": ("Stream",),
}


def compatible_styles(label):
    return (label, *LOWER_STYLES.get(label, ()))
