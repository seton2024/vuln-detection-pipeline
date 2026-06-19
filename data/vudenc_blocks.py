"""
VUDENC context-block extraction.

the same blockbuilding logic as in VUDENC project by Laura Wartschinski

creating windows of code
"""

# Characters that VUDENC treats as token boundaries. Context windows always
# start and end on one of these so we never cut a token in half.
SPLITCHARS = [
    " ", "\t", "\n", ".", ":", "(", ")", "[", "]", "<", ">", "+", "-", "=",
    "\"", "\'", "*", "/", "\\", "~", "{", "}", "!", "?", ";", ",", "%", "&",
]


def nextsplit(sourcecode: str, focus: int) -> int:
    #Return the index of the next split character after `focus`, or -1.
    for pos in range(focus + 1, len(sourcecode)):
        if sourcecode[pos] in SPLITCHARS:
            return pos
    return -1


def previoussplit(sourcecode: str, focus: int) -> int:
    #Return the index of the previous split character before `focus`, or -1.
    pos = focus - 1
    while pos >= 0:
        if sourcecode[pos] in SPLITCHARS:
            return pos
        pos -= 1
    return -1


def getcontextPos(sourcecode: str, focus: int, fulllength: int):
    """
    Grow a window outward from character index `focus` until the slice between
    its start and end is longer than `fulllength` characters.
    """
    startcontext = focus
    endcontext = focus
    if focus > len(sourcecode) - 1:
        return None

    start = True
    while not len(sourcecode[startcontext:endcontext]) > fulllength:
        if previoussplit(sourcecode, startcontext) == -1 and nextsplit(sourcecode, endcontext) == -1:
            return None
        if start:
            if previoussplit(sourcecode, startcontext) > -1:
                startcontext = previoussplit(sourcecode, startcontext)
            start = False
        else:
            if nextsplit(sourcecode, endcontext) > -1:
                endcontext = nextsplit(sourcecode, endcontext)
            start = True

    return [startcontext, endcontext]


def findposition(badpart: str, sourcecode: str):
    #Find where `badpart` sits inside `sourcecode`.

    splitchars = SPLITCHARS
    pos = 0
    matchindex = 0
    inacomment = False
    startfound = -1
    endfound = -1
    position = []
    end = False
    last = 0

    # Drop any trailing comment from the badpart itself.
    while "#" in badpart:
        f = badpart.find("#")
        badpart = badpart[:f]

    b = badpart.lstrip()
    if len(b) < 1:
        return [-1, -1]

    while not end:
        if not inacomment:
            last = pos - 1

        if pos >= len(sourcecode):
            end = True
            break

        if sourcecode[pos] == "\n":
            inacomment = False

        if sourcecode[pos] == "\n" and (sourcecode[pos - 1] == "\n" or sourcecode[last] == " "):
            pos = pos + 1
            continue

        if sourcecode[pos] == " " and (sourcecode[pos - 1] == " " or sourcecode[last] == "\n"):
            pos = pos + 1
            continue

        if sourcecode[pos] == "#":
            inacomment = True

        if not inacomment:
            a = sourcecode[pos]
            if a == "\n":
                a = " "
            b = badpart[matchindex]

            c = ""
            if matchindex > 0:
                c = badpart[matchindex - 1]

            d = ""
            if matchindex < len(badpart) - 2:
                d = badpart[matchindex + 1]

            if (a != b) and (a == " " or a == "\n") and ((b in splitchars) or (c in splitchars)):
                pos = pos + 1
                continue

            if (a != b) and (b == " " or b == "\n"):
                if (c in splitchars or d in splitchars):
                    if matchindex < len(badpart) - 1:
                        matchindex = matchindex + 1
                        continue

            if a == b:
                if matchindex == 0:
                    startfound = pos
                matchindex = matchindex + 1
            else:
                matchindex = 0
                startfound = -1

            if matchindex == len(badpart):
                endfound = pos
                break

        if pos == len(sourcecode):
            end = True
        pos = pos + 1

    position.append(startfound)
    position.append(endfound)

    if endfound < 0:
        startfound = -1
    if endfound < 0 and startfound < 0:
        return [-1, -1]
    return position


def findpositions(badparts, sourcecode: str):
#Run findposition for every badpart, keeping only the ones actually located.
#Returns a list of [start, end] character-index pairs.
    positions = []
    for bad in badparts:
        if "#" in bad:
            bad = bad[:bad.find("#")]
        place = findposition(bad, sourcecode)
        if place != [-1, -1]:
            positions.append(place)
    return positions


def getblocks(sourcecode: str, badpositions, step: int, fulllength: int):
    """
    Slide a context window across `sourcecode` and emit labeled blocks.

    Args:
        sourcecode:   full source text of the file.
        badpositions: list of [start, end] char ranges of vulnerable code,
                      from findpositions().
        step:         how far (in characters) to advance the focus each step.
        fulllength:   target context length in characters per window.

    Returns:
        A list of [code_snippet, label] pairs, where label is
        1 (vulnerable) if the snippet overlaps any badposition, else 0 (safe).
        Duplicate snippets are dropped.

    """
    blocks = []

    focus = 0
    lastfocus = 0
    while True:
        if focus > len(sourcecode):
            break

        focusarea = sourcecode[lastfocus:focus]

        if not (focusarea == "\n"):
            middle = lastfocus + round(0.5 * (focus - lastfocus))
            context = getcontextPos(sourcecode, middle, fulllength)

            if context is not None:
                vulnerable = False
                for bad in badpositions:
                    if (context[0] > bad[0] and context[0] <= bad[1]) or \
                       (context[1] > bad[0] and context[1] <= bad[1]) or \
                       (context[0] <= bad[0] and context[1] >= bad[1]):
                        vulnerable = True

                label = 1 if vulnerable else 0   # our convention: 1=vuln, 0=safe
                snippet = sourcecode[context[0]:context[1]]

                if not any(existing[0] == snippet for existing in blocks):
                    blocks.append([snippet, label])

        # Advance the focus: prefer jumping to the next line break nearby,
        # otherwise jump `step` chars ahead to the next token boundary.
        if "\n" in sourcecode[focus + 1:focus + 7]:
            lastfocus = focus
            focus = focus + sourcecode[focus + 1:focus + 7].find("\n") + 1
        else:
            if nextsplit(sourcecode, focus + step) > -1:
                lastfocus = focus
                focus = nextsplit(sourcecode, focus + step)
            else:
                if focus < len(sourcecode):
                    lastfocus = focus
                    focus = len(sourcecode)
                else:
                    break

    return blocks
