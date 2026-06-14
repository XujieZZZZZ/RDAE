import re
import textwrap

def load_code_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_edit_block(code: str,
                       start_tag="# === BEGIN CARBON-AWARE LOGIC ===",
                       end_tag="# === END CARBON-AWARE LOGIC ==="):
    pattern = re.compile(
        f"{re.escape(start_tag)}(.*?){re.escape(end_tag)}",
        re.DOTALL
    )
    match = pattern.search(code)
    if not match:
        raise ValueError("Editable block not found")

    return match.group(1).strip(), match.span()


def apply_modified_block(original_code: str,
                         span: tuple,
                         new_block: str,
                         start_tag="# === BEGIN CARBON-AWARE LOGIC ===",
                         end_tag="# === END CARBON-AWARE LOGIC ==="):
    start, end = span

    # IMPORTANT:
    # The editable block lives inside a Python class. We must indent *every* line
    # of the generated block to match the indentation level of the marker line.
    # Indenting only the first line can break method definitions and cause many bugs.
    prefix = original_code[:start]
    last_line = prefix.splitlines()[-1]
    base_indent = last_line[:len(last_line) - len(last_line.lstrip())]

    # 只缩进第一行
    prefix = original_code[:start]
    last_line = prefix.splitlines()[-1]
    indent = last_line[:len(last_line) - len(last_line.lstrip())]
    # 拆分 new_block
    lines = new_block.strip().splitlines()
    if lines:
        # 只给第一行加缩进
        lines[0] = indent + lines[0]
    indented_block = "\n".join(lines)

    return (
            original_code[:start]
            + start_tag + "\n"
            + indented_block + "\n"
            + indent + end_tag
            + original_code[end:]
    )
    # #全都不缩进
    # return (
    #        original_code[:start]
    #        + start_tag + "\n"
    #        + new_block.strip() + "\n"
    #        + end_tag
    #        + original_code[end:] )


# def validate_scheduler_class(code: str, class_name: str):
#     namespace = {}
#     exec(code, namespace)
#     cls = namespace.get(class_name)
#     if cls is None:
#         raise RuntimeError(f"{class_name} not found")
#     if not hasattr(cls, "get_action"):
#         raise RuntimeError(f"{class_name}.get_action not implemented")


# import json
# with open("/Users/dingch/Desktop/Projections/AI4Research/aideml-main-FIFO/aideml-main/carbon_aware_decompose/logs/10-carbon_aware_optimization/journal.json", "r") as rf:
#     obj_dict = json.load(rf)
# print(obj_dict.keys())