import re

def parse_and_format_references():
    with open(r'd:\ares\milestones\M15_release\references.bib', 'r', encoding='utf-8') as f:
        text = f.read()

    entries = re.findall(r'@(\w+)\{([^,]+),\s*(.*?)\n\}', text, re.DOTALL)
    formatted = []
    
    for i, (kind, key, fields_str) in enumerate(entries[:65]):
        # Extract title
        m_t = re.search(r'title\s*=\s*\{+(.*?)\}+,', fields_str, re.DOTALL)
        title = m_t.group(1).replace('\n', ' ').strip() if m_t else 'Untitled'
        title = re.sub(r'[\{\}]', '', title)
        
        # Extract author
        m_a = re.search(r'author\s*=\s*\{+(.*?)\}+,', fields_str, re.DOTALL)
        author = m_a.group(1).replace('\n', ' ').strip() if m_a else 'Anonymous'
        authors = author.split(' and ')
        if len(authors) > 2:
            author_str = f"{authors[0].split()[-1]} et al."
        elif len(authors) == 2:
            author_str = f"{authors[0].split()[-1]} and {authors[1].split()[-1]}"
        else:
            author_str = authors[0]
            
        # Extract venue
        m_j = re.search(r'(?:journal|booktitle)\s*=\s*\{+(.*?)\}+,', fields_str, re.DOTALL)
        venue = m_j.group(1).replace('\n', ' ').strip() if m_j else 'Preprint'
        venue = re.sub(r'[\{\}]', '', venue)
        
        # Extract year
        m_y = re.search(r'year\s*=\s*\{*(\d{4})\}*', fields_str)
        year = m_y.group(1) if m_y else '2024'
        
        # Extract DOI
        m_d = re.search(r'doi\s*=\s*\{+(.*?)\}+', fields_str)
        doi = m_d.group(1).strip() if m_d else ''
        if doi.startswith('http'):
            doi_str = f"url: {doi}"
        elif doi:
            doi_str = f"doi: {doi}"
        else:
            doi_str = ""
            
        ref_line = f"[{i+1}] {author_str}, \"{title},\" in {venue}, {year}. {doi_str}".strip()
        formatted.append(ref_line)
        
    print(f"Successfully formatted {len(formatted)} references.")
    for r in formatted[:5]:
        print(r)
    return formatted

if __name__ == "__main__":
    parse_and_format_references()
