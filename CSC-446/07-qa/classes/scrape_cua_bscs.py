import re
import json
import requests
from bs4 import BeautifulSoup

URL = "https://engineering.catholic.edu/academics/undergraduate/computer-science/comp-sci-curriculum/index.html"

# ---------- Helpers to build the data model ----------

programs = []
semesters = []
semester_courses = []
courses = {}  # keyed by course code


def add_program(program_id, name, catalog_start_term, catalog_end_term):
    programs.append({
        "id": program_id,
        "name": name,
        "catalog_start_term": catalog_start_term,
        "catalog_end_term": catalog_end_term
    })


def add_course(code, title, credits):
    """Create/update a course if not already present."""
    code = code.strip()
    if code not in courses:
        courses[code] = {
            "code": code,
            "title": title.strip(),
            "credits_default": credits,
            "department": code.split()[0] if " " in code else None
        }


def add_semester(program_id, year, term, total_credits):
    sem_id = f"{program_id}-year{year}-{term.lower()}"
    semesters.append({
        "id": sem_id,
        "program_id": program_id,
        "year": year,
        "term": term,
        "total_credits": total_credits
    })
    return sem_id


def add_semester_courses(semester_id, lines):
    """
    Parse the course lines for one semester.

    `lines` should be everything AFTER the "Code Course Credits" header
    and including the "Total Credits" line.
    """
    total_credits = None
    pos = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Total credits row
        if line.startswith("Total Credits"):
            # last token is the number
            try:
                total_credits = int(line.split()[-1])
            except ValueError:
                total_credits = None
            continue

        # Anything else is assumed to be a course row: "... ... <credits>"
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        left, credits_str = parts
        try:
            credits = int(credits_str)
        except ValueError:
            # Skip lines that don't end with an integer
            continue

        # Code is first token, title is the rest
        left_parts = left.split(" ", 1)
        if len(left_parts) == 1:
            code = left_parts[0]
            title = ""
        else:
            code, title = left_parts

        add_course(code, title, credits)

        semester_courses.append({
            "semester_id": semester_id,
            "course_code": code,
            "credits": credits,
            "position": pos
        })
        pos += 1

    # Patch in total_credits for this semester
    for s in semesters:
        if s["id"] == semester_id:
            s["total_credits"] = total_credits
            break


# ---------- Scrape & parse ----------

resp = requests.get(URL)
resp.raise_for_status()

soup = BeautifulSoup(resp.text, "html.parser")

# Get plain text and split into cleaned lines
text = soup.get_text("\n")
lines = [l.strip() for l in text.splitlines() if l.strip()]

# Find the curriculum section indices
idx_2017 = None
idx_2023 = None

for i, line in enumerate(lines):
    if line.startswith("Curriculum for students entering in Fall 2017"):
        idx_2017 = i
    if line.startswith("Curriculum for students entering in Fall 2023"):
        idx_2023 = i

if idx_2017 is None or idx_2023 is None:
    raise RuntimeError("Could not locate curriculum markers in the page.")

# Assume everything after 2023 block (until end of file) is part of 2023+
end_2017 = idx_2023
end_2023 = len(lines)

# Register programs
add_program(
    "bscs-2017-2023",
    "B.S. Computer Science Course Sequence (2017–2023)",
    "2017-08",
    "2023-05",
)
add_program(
    "bscs-2023+",
    "B.S. Computer Science Course Sequence (2023+)",
    "2023-08",
    None,
)

sem_header_re = re.compile(r"Year\s+(\d+)\s+(Fall|Spring)", re.IGNORECASE)


def parse_program_block(program_id, start_idx, end_idx):
    i = start_idx + 1  # skip the "Curriculum for..." line

    while i < end_idx:
        line = lines[i]

        # Detect a "Year X Fall/Spring" header
        m = sem_header_re.search(line)
        if m:
            year = int(m.group(1))
            term = m.group(2).title()  # "Fall" or "Spring"

            # Collect lines for this semester
            # First, skip any "Code Course Credits" header line
            j = i + 1
            if j < end_idx and "Code" in lines[j] and "Credits" in lines[j]:
                j += 1

            semester_lines = []
            while j < end_idx:
                l2 = lines[j]
                # Stop at next semester header or next curriculum block
                if sem_header_re.search(l2) or l2.startswith("Curriculum for students entering"):
                    break
                semester_lines.append(l2)
                j += 1

            # Create semester and its courses
            sem_id = add_semester(program_id, year, term, total_credits=None)
            add_semester_courses(sem_id, semester_lines)

            i = j
        else:
            i += 1


# Parse each program block
parse_program_block("bscs-2017-2023", idx_2017, end_2017)
parse_program_block("bscs-2023+", idx_2023, end_2023)

# ---------- Build final JSON and save ----------

data = {
    "source_url": URL,
    "programs": programs,
    "semesters": semesters,
    "courses": list(courses.values()),
    "semester_courses": semester_courses,
}

with open("cua_bscs_curriculum.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Saved curriculum to cua_bscs_curriculum.json")
