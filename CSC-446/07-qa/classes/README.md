✅ 1. System Prompt (JSON directly in context)
Use this when you include the JSON at the end of the prompt, as a long data blob (e.g., in the first message).

✅ 2. System Prompt (with tool-based data access)
Use this when you define functions/tools like:
get_programs()
get_semesters(program_id)
get_semester(program_id, year, term)
get_course(course_code)
list_semesters_for_course(course_code)

✅ 3. System Prompt (RAG + structured JSON reasoning)
Use this if you're mixing RAG chunks with structured JSON (best for semantic questions).


Questions:
“Show me Year 2 Fall for the 2023+ curriculum.”
“Where does CSC 323 appear?”
“Compare senior year between 2017–2023 and 2023+.”
“How many total CS credits do I complete before Year 3?”
“Create a 4-year plan if I am behind in math.”



huggingface-cli login
