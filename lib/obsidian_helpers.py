import os
import re
from pathlib import Path


def extract_sections(file_path: str) -> dict[str, str]:
    # For a given markdown note, make a dict mapping headers to content.
    sections = {}
    with open(file_path, "r") as file:
        content = file.read().split("\n")
        section = ""
        sections[section] = ""
        for line in content:
            if line.startswith("##"):
                if sections[section]:
                    section = line.lstrip("#").strip()
                    sections[section] = ""
            else:
                sections[section] += line + "\n"
    return sections


def clean_section(txt: str) -> str:
    txt = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", txt)

    repl = ["[[", "]]", "*"]
    for r in repl:
        txt = txt.replace(r, "")
    repl_space = ["\n", "\t", "\xa0", "  "]
    for r in repl_space:
        txt = txt.replace(r, " ")
    txt = txt.replace("\\\\", "\\")

    txt = txt.strip()
    return txt


def read_markdown_notes(folder_path: Path) -> dict[tuple[str, str], str]:
    # Iterate through vault, making a dictionary of {(filename, section): text}
    notes: dict[tuple[str, str], str] = {}
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for skip_dir_name in [
            ".git",
            ".obsidian",
            ".trash",
            "_attachments",
            "R-templates",
            "R-excalidraw",
        ]:
            if skip_dir_name in dirnames:
                dirnames.remove(skip_dir_name)
        for file in filenames:
            if file.endswith(".md"):
                file_path = os.path.join(dirpath, file)
                relative_path = os.path.relpath(file_path, folder_path)
                # Clean files
                sections = extract_sections(file_path)
                for section_id, section_contents in sections.items():
                    cleaned_txt = clean_section(section_contents)
                    if cleaned_txt == "":
                        continue
                    notes[(relative_path, section_id)] = cleaned_txt
    return notes
