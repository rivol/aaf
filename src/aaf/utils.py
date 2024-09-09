def extract_xml_fragment(source: str, tag: str) -> str:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"

    try:
        start = source.index(start_tag) + len(start_tag)
    except ValueError:
        return source

    try:
        end = source.index(end_tag, start)
    except ValueError:
        end = len(source)

    return source[start:end]


def truncate_text(text: str, max_length: int) -> str:
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text
