from bs4 import BeautifulSoup
from markdownify import markdownify as md
import requests
import os
import json
from tqdm import tqdm
import fire
import re
import html2text

# Extract and print the main text from each card
def get_blog_urls():
    # get all blog
    base_url = "https://www.llamaindex.ai"
    response = requests.get(f"{base_url}/blog")
    html = response.text
    
    soup = BeautifulSoup(html, 'html.parser')
    # Find all blog post cards
    blog_cards = soup.find_all('div', class_='CardBlog_card__mm0Zw')
    blog_data = []
    for card in blog_cards:
        # Extract title
        title_element = card.find('p', class_='CardBlog_title__qC51U').find('a')
        title = title_element.text.strip()
        url = base_url + title_element['href']
    
        # Extract publication date
        date = card.find('p', class_='Text_text__zPO0D Text_text-size-16__PkjFu').text.strip()
    
        # Print the extracted information
        print(f"Title: {title}")
        print(f"Date: {date}")
        print(f"URL: {url}")
        print("---")

        blog_data.append({
            "title": title,
            "date": date,
            "url": url
        })
        
    return blog_data

def extract_page(page_url):
    response = requests.get(page_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove script and style tags
    for script in soup(['script', 'style']):
        script.extract()

    # Get rid of empty tags
    for tag in soup.find_all():
        if not tag.text.strip():
            tag.extract()

    # only get main blog content, ignore related blogs
    blog_content = soup.find("main").find("div", class_="BlogPost_htmlPost__Z5oDL")

    return blog_content.prettify()

def remove_emojis(text):
    # Regular expression to match emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002700-\U000027BF"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed Characters
        "]+", 
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)
    
def html_to_md(html_content: str) -> str:
    """
    Preprocess the extracted text: convert HTML to Markdown, remove unwanted sections,
    clean up the text, and add the full domain to relative image URLs.

    Args:
        text (str): The input HTML text to preprocess.

    Returns:
        str: The preprocessed Markdown text.
    """
    
    # Remove emojis from the HTML content
    cleaned_html = remove_emojis(html_content)
    
    # Create an instance of HTML2Text object
    h = html2text.HTML2Text()
    # Ignore links, images, and other content
    h.ignore_links = False
    h.ignore_images = False
    
    # Convert HTML to Markdown
    markdown = h.handle(cleaned_html)
    
    return markdown


def main(html_dir, md_dir, metadata_path="./data/llama_blogs_metadata.json", debug=False):
    os.makedirs(html_dir, exist_ok=True)
    os.makedirs(md_dir, exist_ok=True)
    
    # extract all blog page and save to folder data
    blogs_metadata = get_blog_urls()
    with open(metadata_path, "w") as f:
        f.write(json.dumps(blogs_metadata))

    for blog_data in tqdm(blogs_metadata, desc="Crawling data"):
        blog_url = blog_data["url"]
        
        html_doc = extract_page(blog_url)
        md_doc = html_to_md(html_doc)
        
        save_name =  blog_url.split("/")[-1]

        print("Saving:", save_name) 
        
        with open(f'{html_dir}/{save_name}.html', 'w', encoding='utf-8') as f:
            f.write(html_doc)

        with open(f'{md_dir}/{save_name}.md', 'w', encoding='utf-8') as f:
            f.write(md_doc)

        if debug:
            break

        
if __name__ == "__main__":
    fire.Fire(main)