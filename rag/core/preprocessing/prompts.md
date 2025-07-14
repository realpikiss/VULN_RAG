Voici la version complète en **Markdown** :

```markdown
# 🎯 **Prompt Unique Optimisé (1 Appel LLM)**

```

**VULNERABILITY SEARCH KEYWORD EXTRACTION**

Code to analyze:

```c
{code}
```

Generate concise search keywords for vulnerability database lookup:

**Instructions:**

1. Identify the main security-relevant purpose (2-4 words max)
2. Extract dangerous functions and operations (3-6 technical terms)
3. Focus on terms that help find similar vulnerability patterns

**Output Format (STRICT):**
Purpose: \[2-4 keywords separated by spaces]
Functions: \[3-6 keywords separated by spaces]

**Examples:**
Purpose: buffer copy validation
Functions: strcpy user input bounds

Purpose: memory allocation tracking
Functions: malloc free pointer arithmetic

Purpose: format string printing
Functions: printf user format control

**Your Response:**
Purpose:
Functions:

````

---

## 🧪 **Parser la réponse (Python)**

```python
def parse_response(response):
    purpose = re.search(r'Purpose:\s*(.+)', response).group(1).strip()
    functions = re.search(r'Functions:\s*(.+)', response).group(1).strip()
    return purpose, functions
````

**C'est tout.** 🎯

```
```
